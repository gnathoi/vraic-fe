"""Age/sex cohort-component engine (reduced model, data gate B).

State N[draw, sex, age] at each year end; sex 0=F, 1=M; ages 0..99 and open-ended 100+.
Annual order follows the official methodology: ageing -> deaths -> births -> net migration.
The same code runs on NumPy (reference) or CuPy (GPU) via the `xp` module argument.
"""
import hashlib
import json
import os
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from .schemas import FIRST_YEAR, Scenario, default_scenario

MODEL_VERSION = "jfe-cohort-1.4"  # 1.1: unbiased migration rounding, blank base-year flows; 1.2: migration variability (AR(1), 2001-2025)
CODES = ["L2", "L1", "MD", "H1", "H2"]
LTC_BANDS = [(65, 74), (75, 84), (85, 94), (95, 100)]
METRICS = {
    "population_total": "Total population",
    "population_0_15": "Population aged 0-15",
    "population_16_64": "Population aged 16-64",
    "population_65_plus": "Population aged 65+",
    "population_80_plus": "Population aged 80+",
    "old_age_ratio": "People aged 65+ per 100 aged 16-64",
    "dependency_ratio": "People aged 0-15 and 65+ per 100 aged 16-64",
    "ltc_claims_index_65plus": "65+ long-term-care claim index at unchanged 2024 claim rates (2025 = 100)",
    "school_age_primary": "Primary school-age residents (Reception-Year 6, school year starting that September)",
    "school_age_secondary": "Secondary school-age residents (Years 7-13, school year starting that September)",
    "pupils_primary_proxy": "Primary pupils at unchanged 2024 participation (illustrative, not capacity)",
    "pupils_secondary_proxy": "Secondary pupils incl. sixth form at unchanged 2024 participation (illustrative, not capacity)",
    "hospital_bed_days": "Acute hospital bed days at 2018-22 rates by age and sex",
    "gp_appointments": "GP surgery appointments at 2021-23 rates by age and sex",
    "nurses_fte_proxy": "Hospital nurses and midwives (FTE) at constant FTE per bed day (illustrative)",
    "doctors_fte_proxy": "Hospital doctors (FTE) at constant FTE per bed day (illustrative)",
    "homes_needed_additional": "Additional homes needed since end-2025 (official housing-needs method, to 2040)",
    "births": "Births in year",
    "deaths": "Deaths in year",
    "net_migration": "Net migration applied in year",
}
UNITS = {k: "people" for k in METRICS} | {"old_age_ratio": "per 100", "dependency_ratio": "per 100", "ltc_claims_index_65plus": "index",
                                         "hospital_bed_days": "bed days", "gp_appointments": "appointments", "nurses_fte_proxy": "FTE", "doctors_fte_proxy": "FTE",
                                         "pupils_primary_proxy": "pupils", "pupils_secondary_proxy": "pupils",
                                         "homes_needed_additional": "homes"}
QUANTILES = {"p05": 5, "p25": 25, "p50": 50, "p75": 75, "p95": 95}


class Pack:
    def __init__(self, path):
        path = Path(path)
        z = np.load(path / "pack.npz")
        self.__dict__.update({k: z[k] for k in z.files})
        self.meta = json.loads((path / "pack.json").read_text())
        self.id = self.meta["pack_id"]
        self.sex_ratio = float(self.sex_ratio)

    def ti(self, year):
        """Index into rate arrays, which start at FIRST_YEAR (2026)."""
        return year - FIRST_YEAR


@lru_cache(maxsize=1)
def load_pack(path=None):
    return Pack(path or os.environ.get("JFE_PACK", "data/processed"))


MIG = [0, 200, 400, 600, 800]


def official_match(pack, sc: Scenario):
    """Published projection matching the scenario's assumption codes (and constant net if published).
    -> (mig_index, f, m, label, exact) where exact means the scenario IS a published combination."""
    f, m = CODES.index(sc.fertility.assumption), CODES.index(sc.mortality.assumption)
    nets = sc.net_by_year()
    exact = len(set(nets)) == 1 and nets[0] in MIG and not sc.fertility.multiplier and not sc.mortality.multiplier
    mi = MIG.index(nets[0]) if exact else (MIG.index(nets[0]) if nets[0] in MIG else 2)
    net_txt = "net nil" if MIG[mi] == 0 else f"+{MIG[mi]}"
    label = f"Official projection {net_txt}, fertility {sc.fertility.assumption}, life expectancy {sc.mortality.assumption} (Statistics Jersey, Feb 2026)"
    return mi, f, m, label, exact


def experiment_hash(sc: Scenario, pack: Pack) -> str:
    numeric = sc.model_dump(exclude={"title", "parent_scenario_id", "solved_for"})
    blob = json.dumps({"scenario": numeric, "pack": pack.id, "model": MODEL_VERSION, "precision": "float64"}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def age_forward(xp, n):
    o = xp.zeros_like(n)
    o[..., 1:] = n[..., :-1]
    o[..., -1] += n[..., -1]
    return o


def integerise(v):
    """Round a vector to integers preserving its rounded sum (largest remainder)."""
    flat = v.ravel()
    fl = np.floor(flat)
    short = int(round(flat.sum())) - int(fl.sum())
    order = np.argsort(-(flat - fl))
    fl[order[:short]] += 1
    return fl.reshape(v.shape)


def round_systematic(xp, v, rs, draws):
    """Whole people per age and sex, unbiased per cell: randomised systematic rounding with one offset per draw.
    Each cell rounds up with probability equal to its fraction; the draw's total stays within one person of the sum.
    v is one [2, A] array for every draw, or a per-draw [D, 2, A] array."""
    v = xp.asarray(v, dtype=float)
    shape = v.shape[-2:]
    m = v.reshape(-1, shape[0] * shape[1])  # [1 or D, cells]
    fl = xp.floor(m)
    frac = m - fl
    c = xp.cumsum(frac, axis=1)
    u = (rs.random((draws, 1)) if hasattr(rs, "random") and not hasattr(rs, "random_sample") else rs.random_sample((draws, 1)))
    extra = xp.floor(c + u) - xp.floor(c - frac + u)
    return (fl + extra).reshape((draws,) + tuple(shape))


# Fertility and mortality level uncertainty, in units of the official L1/H1 variant spread (L1/H1 at the 5th/95th percentiles x scale).
# Scales calibrated to real-time forecast errors: fertility 1.5 (GFR 1990-2025), mortality 4 (2012-2025).
RATE_SCALE = {"fertility": 1.5, "mortality": 4.0}

# Net migration variability around the assumed path: AR(1) fitted to Jersey's 2001-2025 record.
MIGRATION_AR1 = {"phi": 0.59, "sigma": 338.0}  # AR(1) on 2001-2025 net migration (stationary sd 418)


def migration_deviations(xp, seed, years, draws):
    """[D, Y] deviations e_t = phi*e_{t-1} + sigma*z_t, started from the stationary distribution; mean zero, so the median
    follows the assumed path. One random stream per year (seed, year, 5): the same scenario gets the same deviations for a
    given year whatever its horizon, and scenario and comparator share them (paired differences)."""
    phi, sigma = MIGRATION_AR1["phi"], MIGRATION_AR1["sigma"]
    e = xp.empty((draws, years))
    for t in range(years):
        z = _rng(xp, seed, FIRST_YEAR + t, 5).standard_normal(draws)
        e[:, t] = z * sigma / (1 - phi * phi) ** 0.5 if t == 0 else phi * e[:, t - 1] + sigma * z
    return e


def school_inputs(pack, xp=np):
    """Weights for derived service indicators (schools and health), on the array backend in use."""
    return (xp.asarray(pack.w_primary), xp.asarray(pack.w_secondary), float(pack.school_ratio[0]), float(pack.school_ratio[1]),
            xp.asarray(pack.w_bed), xp.asarray(pack.w_gp))


def metric_block(xp, N, rates, school):
    """N[..., 2, A] -> dict of arrays over leading dims."""
    wp, ws, rp, rs, wbed, wgp = school
    both = N.sum(-2)
    prim, sec = both @ wp, both @ ws
    tot = N.sum((-1, -2))
    young, work, old = N[..., :16].sum((-1, -2)), N[..., 16:65].sum((-1, -2)), N[..., 65:].sum((-1, -2))
    ltc = sum(rates[b, s] * N[..., s, lo:hi + 1].sum(-1) for b, (lo, hi) in enumerate(LTC_BANDS) for s in range(2))
    return {
        "population_total": tot, "population_0_15": young, "population_16_64": work, "population_65_plus": old,
        "population_80_plus": N[..., 80:].sum((-1, -2)), "old_age_ratio": 100 * old / work,
        "dependency_ratio": 100 * (young + old) / work, "ltc_claims": ltc,
        "school_age_primary": prim, "school_age_secondary": sec, "pupils_primary_proxy": rp * prim, "pupils_secondary_proxy": rs * sec,
        "hospital_bed_days": (N * wbed).sum((-1, -2)), "gp_appointments": (N * wgp).sum((-1, -2)),
    }


def homes_needed(pack, nets_from_2026, years):
    """Additional homes needed since end-2025, total [len(years)] and by dwelling type [len(years), types].
    Exact for constant net migration (equals the official 'change since 2025'); a time-varying path uses the
    average net migration to date. None beyond the published horizon (2040)."""
    hy = [int(y) for y in pack.house_years]
    tot, by = [], []
    for y in years:
        if y not in hy:
            tot.append(None)
            by.append(None)
            continue
        k = y - FIRST_YEAR + 1
        nbar = float(np.mean(nets_from_2026[:k])) if k > 0 else 0.0
        i = hy.index(y)
        d = (pack.house_nil[i] - pack.house_nil[0]) + nbar * (pack.house_per[i] - pack.house_per[0])
        tot.append(float(d.sum()))
        by.append([round(float(v), 3) for v in d])
    return tot, by


def compile_rates(pack: Pack, sc: Scenario):
    years = list(range(FIRST_YEAR, sc.end_year + 1))
    f, m = CODES.index(sc.fertility.assumption), CODES.index(sc.mortality.assumption)
    fm, mm, nets = sc.schedule(sc.fertility), sc.schedule(sc.mortality), sc.net_by_year()
    q = np.stack([pack.q[m, pack.ti(y)] * mm[i] for i, y in enumerate(years)])
    if (q > 1).any():
        raise ValueError("The mortality multiplier pushes an annual death probability above 1; choose a smaller multiplier.")
    c = np.array([pack.fert_intensity[f, pack.ti(y)] * fm[i] for i, y in enumerate(years)])
    M0 = np.stack([pack.M0[f, m, pack.ti(y)] for y in years])
    P = np.stack([pack.P[f, m, pack.ti(y)] for y in years])
    M = M0 + np.asarray(nets)[:, None, None] * P
    return years, q, c, M, M0, P


def simulate(pack: Pack, sc: Scenario, xp=np, base=None):
    """Return per-draw metrics [D, Y+1] (year 0 = baseline), mean age structure [Y+1, 2, A], flows and checks."""
    years, q, c, M, M0, P = compile_rates(pack, sc)
    stochastic = sc.uncertainty.mode == "process"
    D = sc.uncertainty.draws
    # per-draw deviations of net migration from the assumed path (shared with the comparator through the seed)
    dev = migration_deviations(xp, sc.seed, len(years), D) if stochastic and sc.uncertainty.migration == "historical" else None
    # per-draw fertility and mortality levels: the official low/high (L1/H1) variants are the 5th/95th percentiles, interpolated on the log
    # scale year by year around the scenario's own rates; same seed -> same levels for scenario and comparator
    rate_draws = stochastic and sc.uncertainty.rates == "official_range"
    if rate_draws:
        zr = _rng(np, sc.seed, 0, 6).standard_normal((2, D)) / 1.645 * np.array([[RATE_SCALE["fertility"]], [RATE_SCALE["mortality"]]])
        md, h1 = CODES.index("MD"), CODES.index("H1")
        f_ratio = np.array([pack.fert_intensity[h1, pack.ti(y)] / pack.fert_intensity[md, pack.ti(y)] for y in years])
        with np.errstate(divide="ignore", invalid="ignore"):
            q_ratio = np.stack([np.where(pack.q[md, pack.ti(y)] > 0, pack.q[h1, pack.ti(y)] / pack.q[md, pack.ti(y)], 1.0) for y in years])
        zf, zm = xp.asarray(zr[0]), xp.asarray(zr[1])
    if base is None:
        if sc.baseline_id == "observed_end2025":
            base = pack.base_2025
        else:  # the official projected end-2025 state for the matching published combination
            mi, f, m, _, _ = official_match(pack, sc)
            base = pack.off_N[mi, f, m, 0]
    base = np.asarray(base, dtype=float)
    if stochastic:
        if not np.allclose(base, np.round(base)):
            raise ValueError("Process-variation runs need an integer baseline; use the observed end-2025 baseline.")
        base = np.round(base)
    shape, rates, school = xp.asarray(pack.fert_shape), xp.asarray(pack.ltc_rate), school_inputs(pack, xp)
    pm = pack.sex_ratio / (1 + pack.sex_ratio)
    N = xp.broadcast_to(xp.asarray(base), (D, 2, 101)).copy()
    mean_age = [N.mean(0)]
    per = {k: [v] for k, v in metric_block(xp, N, rates, school).items()}
    flows = {"births": [xp.zeros(D)], "deaths": [xp.zeros(D)], "net_migration": [xp.zeros(D)]}
    max_acct_err, clipped = 0.0, 0.0
    for i, y in enumerate(years):
        prev_total = N.sum((-1, -2))
        aged = age_forward(xp, N)
        qy = xp.asarray(q[i])
        if stochastic:
            rs = _rng(xp, sc.seed, y, 1)
            qd = xp.minimum(qy[None] * xp.asarray(q_ratio[i])[None] ** zm[:, None, None], 1.0) if rate_draws else xp.broadcast_to(qy, aged.shape)
            deaths = rs.binomial(aged.astype(xp.int64), qd).astype(float)
        else:
            deaths = aged * qy
        post = aged - deaths
        exp_births = c[i] * (post[:, 0, :] @ shape) * (float(f_ratio[i]) ** zf if rate_draws else 1.0)
        if stochastic:
            rs = _rng(xp, sc.seed, y, 2)
            births = rs.poisson(exp_births).astype(float)
            bm = _rng(xp, sc.seed, y, 3).binomial(births.astype(xp.int64), pm).astype(float)
            post[:, 0, 0], post[:, 1, 0] = births - bm, bm
            target = M[i] if dev is None else xp.asarray(M0[i])[None] + (float(sc.net_by_year()[i]) + dev[:, i])[:, None, None] * xp.asarray(P[i])[None]
            mig = round_systematic(xp, target, _rng(xp, sc.seed, y, 4), D)
        else:
            births = exp_births
            post[:, 0, 0], post[:, 1, 0] = births / (1 + pack.sex_ratio), births * pm
            mig = xp.asarray(M[i])
        N = post + mig
        neg = xp.minimum(N, 0)
        if float(neg.sum()) < -1e-6:
            if not stochastic:
                raise ValueError(f"Infeasible scenario: net migration in {y} removes more people than exist in some age groups.")
            clipped += float(-neg.sum())
            N = N - neg
        applied_mig = N.sum((-1, -2)) - post.sum((-1, -2))
        acct = N.sum((-1, -2)) - (prev_total + births - deaths.sum((-1, -2)) + applied_mig)
        max_acct_err = max(max_acct_err, float(xp.abs(acct).max()))
        for k, v in metric_block(xp, N, rates, school).items():
            per[k].append(v)
        flows["births"].append(births)
        flows["deaths"].append(deaths.sum((-1, -2)))
        flows["net_migration"].append(applied_mig)
        mean_age.append(N.mean(0))
    to_np = (lambda a: a.get()) if xp is not np else (lambda a: a)
    per = {k: to_np(xp.stack(v, 1)) for k, v in (per | flows).items()}
    ltc = per.pop("ltc_claims")
    per["ltc_claims_index_65plus"] = 100 * ltc / ltc[:, :1]
    bed_index = per["hospital_bed_days"] / per["hospital_bed_days"][:, :1]
    per["nurses_fte_proxy"], per["doctors_fte_proxy"] = pack.staff_fte[0] * bed_index, pack.staff_fte[1] * bed_index
    checks = {"max_accounting_error_people": max_acct_err, "people_clipped_by_infeasible_migration": clipped}
    return [FIRST_YEAR - 1] + years, per, to_np(xp.stack(mean_age)), checks


def _rng(xp, seed, year, event):
    """Independent stream per (seed, year, event); the draw index is the position within the batch."""
    ss = np.random.SeedSequence([seed, year, event])
    if xp is np:
        return np.random.Generator(np.random.PCG64(ss))
    return xp.random.RandomState(int(ss.generate_state(1)[0]))


def summarise(per, stochastic, xp=np):
    if not stochastic:
        return {k: {"value": v[0].tolist()} for k, v in per.items()}
    # every metric and quantile in one pass (on the GPU when there is one): per-metric CPU sorts dominated run time
    keys = list(per)
    arr = xp.asarray(np.stack([per[k] for k in keys]))  # [metric, draw, year]
    qs = xp.percentile(arr, list(QUANTILES.values()), axis=1)  # [quantile, metric, year]
    mean = arr.mean(1)
    # share of simulations below the base-year value (for a difference: below zero, i.e. scenario below its baseline)
    below = (arr < arr[:, :, :1] - 1e-9).mean(1)
    if xp is not np:
        qs, mean, below = qs.get(), mean.get(), below.get()
    return {k: {"mean": mean[i].tolist(), "p_below_base": below[i].round(4).tolist()} | {q: qs[j, i].tolist() for j, q in enumerate(QUANTILES)}
            for i, k in enumerate(keys)}


def get_xp(backend):
    if backend == "gpu":
        import cupy
        return cupy
    return np


def run_experiment(sc: Scenario, pack: Pack | None = None, backend="cpu"):
    """Scenario and its comparator are simulated with identical seeds (common random numbers),
    so differences are paired per draw and summarised after subtraction."""
    pack = pack or load_pack()
    xp = get_xp(backend)
    comp = default_scenario(end_year=sc.end_year, baseline_id=sc.baseline_id, uncertainty=sc.uncertainty, seed=sc.seed)
    t0 = time.perf_counter()
    years, per_s, age_s, chk_s = simulate(pack, sc, xp)
    _, per_c, age_c, chk_c = simulate(pack, comp, xp)
    if xp is not np:
        xp.cuda.Stream.null.synchronize()
    compute = time.perf_counter() - t0
    t1 = time.perf_counter()
    stochastic = sc.uncertainty.mode == "process"
    per_d = {k: per_s[k] - per_c[k] for k in per_s}
    mi, fi, mo, off_label, exact = official_match(pack, sc)
    if not exact:
        mi, fi, mo = 2, 2, 2
        off_label = "Official projection +400, mid-range fertility and life expectancy (Statistics Jersey, Feb 2026)"
    off = pack.off_N[mi, fi, mo]
    off_years = [int(y) for y in pack.years if y <= sc.end_year]
    off_m = metric_block(np, off[: len(off_years)], pack.ltc_rate, school_inputs(pack))
    ltc = off_m.pop("ltc_claims")
    off_m["ltc_claims_index_65plus"] = 100 * ltc / ltc[0]
    off_m["nurses_fte_proxy"] = pack.staff_fte[0] * off_m["hospital_bed_days"] / off_m["hospital_bed_days"][0]
    off_m["doctors_fte_proxy"] = pack.staff_fte[1] * off_m["hospital_bed_days"] / off_m["hospital_bed_days"][0]
    obs_years = [int(y) for y in pack.obs_years]
    obs = np.nan_to_num(pack.obs)  # suppressed cells are tiny 97+ male counts; history line only
    obs_m = metric_block(np, obs, pack.ltc_rate, school_inputs(pack))
    for k in ("ltc_claims", "hospital_bed_days", "gp_appointments"):  # no observed service-use series: never show modelled values as observed
        obs_m.pop(k)
    obs_m = {k: v.tolist() for k, v in obs_m.items()}
    # observed pupils are the actual January census counts (census Y+1 <-> year end Y), never modelled values
    census = {int(y) - 1: p for y, p in zip(pack.edu_years, pack.edu_pupils)}
    for i, k in enumerate(("pupils_primary_proxy", "pupils_secondary_proxy")):
        obs_m[k] = [float(census[y][i]) if y in census else None for y in obs_years]
    # homes: the official method is linear in average net migration to date, so each simulation's own migration path gives its own figure
    h_s, hb_s = homes_needed(pack, sc.net_by_year(), years)
    h_c, hb_c = homes_needed(pack, comp.net_by_year(), years)
    aggs = ["value"] if not stochastic else ["mean", *QUANTILES]
    homes = {"scenario": {a: h_s for a in aggs}, "comparator": {a: h_c for a in aggs},
             "difference": {a: [None if x is None else x - y for x, y in zip(h_s, h_c)] for a in aggs}}
    if stochastic and sc.uncertainty.migration == "historical":
        dev = migration_deviations(xp, sc.seed, len(years) - 1, sc.uncertainty.draws)
        dev = dev.get() if xp is not np else dev
        hy = [int(y) for y in pack.house_years]
        per_unit = {y: float((pack.house_per[hy.index(y)] - pack.house_per[0]).sum()) for y in years if y in hy}
        for key, det in (("scenario", h_s), ("comparator", h_c)):
            band = {a: [] for a in aggs}
            for t, (y, v) in enumerate(zip(years, det)):
                if v is None:
                    for a in aggs:
                        band[a].append(None)
                    continue
                draws = v + (dev[:, :t].mean(1) * per_unit[y] if t else np.zeros(dev.shape[0]))
                band["mean"].append(float(draws.mean()))
                for q, pq in QUANTILES.items():
                    band[q].append(float(np.percentile(draws, pq)))
            homes[key] = band
    result = {
        "model_version": MODEL_VERSION,
        "pack_id": pack.id,
        "experiment_hash": experiment_hash(sc, pack),
        "scenario": sc.model_dump(),
        "comparator": comp.model_dump(),
        "comparator_label": comp.title,
        "years": years,
        "aggregation": "deterministic expected values" if not stochastic else "ensemble of process-variation draws: mean and pointwise percentiles; differences are paired by draw",
        "metrics": {k: summarise(v, stochastic, xp) | {"homes_needed_additional": homes[k]} for k, v in (("scenario", per_s), ("comparator", per_c), ("difference", per_d))},
        "housing": {"types": [str(t) for t in pack.house_types], "years": [y for y, v in zip(years, hb_s) if v is not None],
                    "scenario": [v for v in hb_s if v is not None], "comparator": [v for v in hb_c if v is not None],
                    "note": "Official housing-needs method (S39), exact for constant net migration; time-varying paths use average net migration to date; to 2040."},
        "age": {
            "scenario": np.round(age_s, 3).tolist(), "comparator": np.round(age_c, 3).tolist(),
            "aggregation": "deterministic expectation" if not stochastic else "ensemble mean (additive)",
        },
        "official": {
            "label": off_label, "matches_scenario_assumptions": exact,
            "years": off_years, "metrics": {k: v.tolist() for k, v in off_m.items()} | {"homes_needed_additional": homes_needed(pack, [MIG[mi]] * (off_years[-1] - FIRST_YEAR + 1), off_years)[0]},
            "age": np.round(off[: len(off_years)], 3).tolist(),
        },
        "observed": {"label": "Observed estimates (Statistics Jersey, Sept 2026 release; 2025 provisional); pupils from the January school census", "years": obs_years, "metrics": obs_m},
        "ltc_contributions": {
            "bands": ["65-74", "75-84", "85-94", "95+"],
            "scenario": [[float(sum(pack.ltc_rate[b, s] * age_s[t, s, lo:hi + 1].sum() for s in range(2))) for b, (lo, hi) in enumerate(LTC_BANDS)] for t in range(len(years))],
        },
        "checks": {"scenario": chk_s, "comparator": chk_c},
        "reproduction": ({
            "note": "Scenario starts from the official projected end-2025 state with a published assumption combination: engine minus published official values.",
            "max_abs_difference_people": {k: float(np.abs(np.array(per_s[k][0]) - np.array(off_m[k][: len(years)])).max()) for k in ("population_total", "population_16_64", "population_65_plus")},
        } if exact and sc.baseline_id == "official_state_2025" and not stochastic else None),
        "warnings": sc.warnings() + ([f"Net migration would remove more people than exist in some age groups: on average {per_sim:.0f} people per simulation were not removed."]
                                     if (per_sim := chk_s["people_clipped_by_infeasible_migration"] / max(sc.uncertainty.draws, 1)) >= 1 else []),
        "receipt": {
            "backend": "cuda (CuPy)" if xp is not np else "cpu (NumPy)",
            "rng": "numpy PCG64 / SeedSequence(seed, year, event)" if xp is np else "cupy RandomState seeded from SeedSequence(seed, year, event)",
            "precision": "float64", "requested_draws": sc.uncertainty.draws, "completed_draws": sc.uncertainty.draws,
            "seed": sc.seed, "compute_seconds": round(compute, 4), "summary_seconds": round(time.perf_counter() - t1, 4),
            "cells_per_draw": 2 * 101, "years": len(years) - 1, "hardware": _hardware(xp),
        },
    }
    for block in result["metrics"].values():  # flows have no simulated value in the base year: blank, not zero
        for k in ("births", "deaths", "net_migration"):
            for q in block.get(k, {}):
                block[k][q][0] = None
    return result


def _hardware(xp):
    if xp is np:
        return f"CPU, {os.cpu_count()} logical cores"
    props = xp.cuda.runtime.getDeviceProperties(0)
    return props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
