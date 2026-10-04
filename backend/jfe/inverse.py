"""Inverse simulation: which value of ONE assumption makes a metric reach a target, under this model?

Deterministic and bounded: the free parameter is searched only inside its validated range. The response is sampled
across the whole range in one vectorised batch, checked for monotonicity, bracketed and refined. Outcomes are
"solved", "unreachable" (with the closest achievable value) or "not_monotone" (refused). Never a recommendation."""
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .engine import CODES, age_forward, compile_rates, load_pack, metric_block, official_match, school_inputs, simulate
from .schemas import (CODE_LABEL, FERT_MULT, FIRST_YEAR, MAX_END_YEAR, MORT_MULT, NET_LIMITS, Migration, MigrationSegment, RateAssumption,
                      SchedulePoint, Scenario, Uncertainty)

PARAMS = {"net_migration": (NET_LIMITS, 50, "Net migration (people per year)"),
          "fertility_multiplier": (FERT_MULT, 0.025, "Fertility rate multiplier"),
          "mortality_multiplier": (MORT_MULT, 0.02, "Death rate multiplier")}
TARGET_METRICS = ["population_total", "population_0_15", "population_16_64", "population_65_plus", "population_80_plus", "old_age_ratio",
                  "dependency_ratio", "pupils_primary_proxy", "pupils_secondary_proxy", "ltc_claims_index_65plus", "nurses_fte_proxy",
                  "hospital_bed_days", "gp_appointments", "doctors_fte_proxy", "homes_needed_additional"]


class TargetSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameter: Literal["net_migration", "fertility_multiplier", "mortality_multiplier"]
    metric: Literal[tuple(TARGET_METRICS)]  # type: ignore[valid-type]
    year: int = Field(ge=FIRST_YEAR + 1, le=MAX_END_YEAR)
    target_kind: Literal["hold_base", "value", "change_pct"] = "hold_base"
    target_value: float | None = None
    from_year: int = Field(default=FIRST_YEAR, ge=FIRST_YEAR, le=MAX_END_YEAR)
    probability: float | None = Field(default=None, ge=0.5, le=0.99)  # "an 80% chance of …": solved on the outcome distribution
    direction: Literal["at_least", "at_most"] | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.from_year > self.year:
            raise ValueError("The assumption must start changing before the target year.")
        if self.target_kind != "hold_base" and self.target_value is None:
            raise ValueError("A target value is needed.")
        if self.metric == "homes_needed_additional":
            if self.year > 2040:
                raise ValueError("Homes needed are available to 2040 only.")
            if self.target_kind == "change_pct":
                raise ValueError("For homes, give a number of additional homes (for example 2,000 by 2040).")
        return self


def _extend(parent: Scenario, end: int) -> Scenario:
    """Parent with its horizon extended to `end` (last migration level and multipliers held), deterministic."""
    d = parent.model_dump()
    segs = d["migration"]["segments"]
    if end > parent.end_year:
        segs[-1]["to_year"] = end
    d.update(end_year=max(end, parent.end_year), uncertainty={"mode": "deterministic", "draws": 1}, solved_for=None)
    return Scenario.model_validate(d)


def _homes(pack, sc: Scenario, spec: TargetSpec, values):
    """Homes depend only on the net migration path (average to date), so they are evaluated in closed form."""
    from .engine import homes_needed
    base = sc.net_by_year()
    yrs = list(range(FIRST_YEAR, sc.end_year + 1))
    out = []
    for v in values:
        nets = [float(v) if (spec.parameter == "net_migration" and y >= spec.from_year) else n for y, n in zip(yrs, base)]
        out.append(homes_needed(pack, nets, [spec.year])[0][0])
    return np.array(out, dtype=float), 0.0, np.ones(len(values), dtype=bool)


def batch(pack, parent: Scenario, spec: TargetSpec, values, xp=np):
    """Deterministic simulation of len(values) variants of `parent`, each with the free parameter set to one value
    from spec.from_year. Returns (metric at target year [N], metric at base year (scalar), feasible mask [N])."""
    sc = _extend(parent, spec.year)
    if spec.metric == "homes_needed_additional":
        return _homes(pack, sc, spec, values)
    years, q, c, M, _, _ = compile_rates(pack, sc)
    f, m = CODES.index(sc.fertility.assumption), CODES.index(sc.mortality.assumption)
    if sc.baseline_id == "observed_end2025":
        base = pack.base_2025
    else:
        mi, fo, mo, _, _ = official_match(pack, sc)
        base = pack.off_N[mi, fo, mo, 0]
    v = xp.asarray(np.asarray(values, dtype=float))
    n = len(values)
    shape, rates, school = xp.asarray(pack.fert_shape), xp.asarray(pack.ltc_rate), school_inputs(pack, xp)
    pm = pack.sex_ratio / (1 + pack.sex_ratio)
    N = xp.broadcast_to(xp.asarray(base, dtype=float), (n, 2, 101)).copy()
    first = metric_block(xp, N, rates, school)
    feasible = xp.ones(n, dtype=bool)
    for i, y in enumerate(years):
        on = y >= spec.from_year
        t = pack.ti(y)
        qy = xp.asarray(q[i])[None] * xp.ones((n, 1, 1))
        if on and spec.parameter == "mortality_multiplier":
            qy = xp.asarray(pack.q[m, t])[None] * v[:, None, None]
            feasible &= (qy <= 1).all((1, 2))
            qy = xp.minimum(qy, 1.0)
        aged = age_forward(xp, N)
        post = aged - aged * qy
        cy = pack.fert_intensity[f, t] * v if (on and spec.parameter == "fertility_multiplier") else c[i]
        births = cy * (post[:, 0, :] @ shape)
        post[:, 0, 0], post[:, 1, 0] = births * (1 - pm), births * pm
        if on and spec.parameter == "net_migration":
            mig = xp.asarray(pack.M0[f, m, t])[None] + v[:, None, None] * xp.asarray(pack.P[f, m, t])[None]
        else:
            mig = xp.asarray(M[i])[None]
        N = post + mig
        feasible &= (N >= -1e-6).all((1, 2))
        if y == spec.year:
            break
    last = metric_block(xp, N, rates, school)
    ltc, bed = last.pop("ltc_claims"), last["hospital_bed_days"]
    last["ltc_claims_index_65plus"] = 100 * ltc / first["ltc_claims"]
    last["nurses_fte_proxy"] = pack.staff_fte[0] * bed / first["hospital_bed_days"]
    last["doctors_fte_proxy"] = pack.staff_fte[1] * bed / first["hospital_bed_days"]
    first["ltc_claims_index_65plus"], first["nurses_fte_proxy"], first["doctors_fte_proxy"] = 100.0, float(pack.staff_fte[0]), float(pack.staff_fte[1])
    to_np = (lambda a: a.get()) if xp is not np else np.asarray
    base_val = first[spec.metric]
    base_val = float(base_val if np.isscalar(base_val) else to_np(base_val)[0])
    return to_np(last[spec.metric]), base_val, to_np(feasible)


def target_value(spec: TargetSpec, base_val: float) -> float:
    if spec.target_kind == "hold_base":
        return base_val
    if spec.target_kind == "change_pct":
        return base_val * (1 + spec.target_value / 100)
    return float(spec.target_value)


def apply(parent: Scenario, spec: TargetSpec, value: float, solved_for: str | None = None) -> Scenario:
    """The parent scenario with the free parameter set to `value` from spec.from_year (a normal, validated Scenario)."""
    sc = _extend(parent, spec.year)
    d = sc.model_dump()
    if spec.parameter == "net_migration":
        nets = sc.net_by_year()
        yrs = list(range(FIRST_YEAR, sc.end_year + 1))
        nets = [int(value) if y >= spec.from_year else n for y, n in zip(yrs, nets)]
        segs = []
        for y, n in zip(yrs, nets):
            if segs and segs[-1]["net_per_year"] == n:
                segs[-1]["to_year"] = y
            else:
                segs.append({"from_year": y, "to_year": y, "net_per_year": n})
        d["migration"] = {"kind": "fixed_annual_net", "segments": segs}
    else:
        key = "fertility" if spec.parameter == "fertility_multiplier" else "mortality"
        ra = getattr(sc, key)
        before = [{"year": p.year, "value": p.value} for p in ra.multiplier if p.year < spec.from_year - 1]
        pts = [{"year": spec.from_year, "value": round(float(value), 4)}]
        if spec.from_year > FIRST_YEAR:  # the parent's own schedule holds until the year before the solved value applies
            prev = sc.schedule(ra)[spec.from_year - 1 - FIRST_YEAR]
            pts = before + [{"year": spec.from_year - 1, "value": round(float(prev), 6)}] + pts
        d[key] = {"assumption": d[key]["assumption"], "multiplier": pts}
    d["parent_scenario_id"] = parent.parent_scenario_id
    d["solved_for"] = solved_for
    d["uncertainty"] = parent.uncertainty.model_dump()
    return Scenario.model_validate(d)


def _grid(param):
    (lo, hi), step, _ = PARAMS[param]
    return np.round(np.arange(lo, hi + step / 2, step), 6)


def solve(parent: Scenario, spec: TargetSpec, pack=None, sensitivity=True) -> dict:
    pack = pack or load_pack()
    grid = _grid(spec.parameter)
    vals, base_val, ok = batch(pack, parent, spec, grid)
    tgt = target_value(spec, base_val)
    g, y = grid[ok], vals[ok]
    curve = [[float(a), float(b) if k else None] for a, b, k in zip(grid, vals, ok)]
    out = {"spec": spec.model_dump(), "target": tgt, "base_value": base_val, "curve": curve, "parameter_label": PARAMS[spec.parameter][2],
           "range": list(PARAMS[spec.parameter][0])}
    d = np.diff(y)
    if len(y) >= 2 and float(np.ptp(y)) < 1e-6:
        return out | {"status": "independent"}
    if len(y) < 3 or not (np.all(d >= -1e-9) or np.all(d <= 1e-9)):
        return out | {"status": "not_monotone"}
    fv = y - tgt
    cross = np.where(np.sign(fv[:-1]) * np.sign(fv[1:]) <= 0)[0]
    if not len(cross):
        k = int(np.argmin(np.abs(fv)))
        return out | {"status": "unreachable", "closest_value": float(g[k]), "closest_metric": float(y[k])}
    i = int(cross[0])
    lo, hi = g[i], g[i + 1]
    if spec.parameter == "net_migration":
        fine = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1)
        fv2, _, _ = batch(pack, parent, spec, fine)
        k = int(np.argmin(np.abs(fv2 - tgt)))
        value, achieved = float(fine[k]), float(fv2[k])
    else:
        fine = np.linspace(lo, hi, 201)
        fv2, _, _ = batch(pack, parent, spec, fine)
        value = float(np.interp(0.0, fv2 - tgt, fine) if fv2[-1] > fv2[0] else np.interp(0.0, (fv2 - tgt)[::-1], fine[::-1]))
        value = round(value, 3)
        achieved = float(batch(pack, parent, spec, [value])[0][0])
    out |= {"status": "solved", "value": value, "achieved": achieved}
    if sensitivity:
        out["sensitivity"] = _sensitivity(parent, spec, pack)
    return out


def _sensitivity(parent: Scenario, spec: TargetSpec, pack) -> list[dict]:
    """Re-solve under the official low/high variants of the assumptions that are not being solved."""
    rows = []
    for key, label in (("fertility", "fertility"), ("mortality", "life expectancy")):
        if (key == "fertility" and spec.parameter == "fertility_multiplier") or (key == "mortality" and spec.parameter == "mortality_multiplier"):
            continue
        for code in ("L1", "H1"):
            d = parent.model_dump()
            d[key] = {"assumption": code, "multiplier": d[key]["multiplier"]}  # switch only the official code; keep any multiplier
            r = solve(Scenario.model_validate(d), spec, pack, sensitivity=False)
            rows.append({"variant": f"official {CODE_LABEL[code]} {label} ({code})", "status": r["status"], "value": r.get("value")})
    return rows


def frontier(parent: Scenario, spec: TargetSpec, pack=None) -> dict:
    """Metric at the target year over a 2-D grid: net migration x (fertility or death-rate) multiplier, in one batch."""
    pack = pack or load_pack()
    ypar = "mortality_multiplier" if spec.parameter == "mortality_multiplier" else "fertility_multiplier"
    nets = np.arange(NET_LIMITS[0], NET_LIMITS[1] + 1, 25)
    ys = _grid(ypar)
    sc = _extend(parent, spec.year)
    if spec.metric == "homes_needed_additional":  # independent of fertility and death rates by construction
        row = _homes(pack, sc, spec.model_copy(update={"parameter": "net_migration"}), nets)[0]
        z = np.tile(row, (len(ys), 1))
        return {"x": nets.tolist(), "x_label": PARAMS["net_migration"][2], "y": ys.tolist(), "y_label": PARAMS[ypar][2], "z": np.round(z, 3).tolist(),
                "target": target_value(spec, 0.0), "scenarios": int(z.size)}
    years, q, c, M, _, _ = compile_rates(pack, sc)
    f, m = CODES.index(sc.fertility.assumption), CODES.index(sc.mortality.assumption)
    base = pack.base_2025 if sc.baseline_id == "observed_end2025" else pack.off_N[(lambda o: o[0])(official_match(pack, sc)), f, m, 0]
    nn, yy = np.meshgrid(nets, ys)
    nv, yv = nn.ravel().astype(float), yy.ravel()
    n = len(nv)
    shape, rates, school = pack.fert_shape, pack.ltc_rate, school_inputs(pack)
    pm = pack.sex_ratio / (1 + pack.sex_ratio)
    N = np.broadcast_to(np.asarray(base, dtype=float), (n, 2, 101)).copy()
    first = metric_block(np, N[:1], rates, school)
    for i, y in enumerate(years):
        on, t = y >= spec.from_year, pack.ti(y)
        qy = (pack.q[m, t][None] * yv[:, None, None]) if (on and ypar == "mortality_multiplier") else q[i][None]
        aged = age_forward(np, N)
        post = aged - aged * np.minimum(qy, 1.0)
        cy = pack.fert_intensity[f, t] * yv if (on and ypar == "fertility_multiplier") else c[i]
        births = cy * (post[:, 0, :] @ shape)
        post[:, 0, 0], post[:, 1, 0] = births * (1 - pm), births * pm
        mig = (pack.M0[f, m, t][None] + nv[:, None, None] * pack.P[f, m, t][None]) if on else M[i][None]
        N = post + mig
        if y == spec.year:
            break
    last = metric_block(np, N, rates, school)
    last["ltc_claims_index_65plus"] = 100 * last.pop("ltc_claims") / first["ltc_claims"][0]
    last["nurses_fte_proxy"] = pack.staff_fte[0] * last["hospital_bed_days"] / first["hospital_bed_days"][0]
    last["doctors_fte_proxy"] = pack.staff_fte[1] * last["hospital_bed_days"] / first["hospital_bed_days"][0]
    first_val = {"ltc_claims_index_65plus": 100.0, "nurses_fte_proxy": float(pack.staff_fte[0]), "doctors_fte_proxy": float(pack.staff_fte[1])}.get(spec.metric)
    base_val = first_val if first_val is not None else float(first[spec.metric][0])
    z = last[spec.metric].reshape(len(ys), len(nets))
    return {"x": nets.tolist(), "x_label": PARAMS["net_migration"][2], "y": ys.tolist(), "y_label": PARAMS[ypar][2], "z": np.round(z, 3).tolist(),
            "target": target_value(spec, base_val), "scenarios": int(n)}


if __name__ == "__main__":  # self-check: batch equals the engine; known solutions behave
    from .schemas import default_scenario
    pack = load_pack()
    par = default_scenario()
    spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040)
    v, b, ok = batch(pack, par, spec, [400, 593])
    ref = simulate(pack, par)[1]["population_16_64"][0][-1]
    assert abs(v[0] - ref) < 1e-6, (v[0], ref)
    r = solve(par, spec, pack)
    print(r["status"], r.get("value"), round(r["target"]), r.get("sensitivity"))
    r2 = solve(par, TargetSpec(parameter="net_migration", metric="old_age_ratio", year=2040), pack, sensitivity=False)
    print(r2["status"], r2.get("closest_value"), r2.get("closest_metric"))
    import time
    t = time.perf_counter()
    fr = frontier(par, spec, pack)
    print("frontier", fr["scenarios"], "scenarios", round(time.perf_counter() - t, 2), "s")


UPPER_IS_WORSE = {"old_age_ratio", "dependency_ratio", "ltc_claims_index_65plus", "hospital_bed_days", "gp_appointments", "nurses_fte_proxy",
                  "doctors_fte_proxy", "homes_needed_additional"}


def _outcomes(pack, sc: Scenario, spec: TargetSpec, draws: int) -> np.ndarray:
    """Per-simulation values of the target metric in the target year (net migration varying as in 2001-2025)."""
    from .engine import get_xp
    proc = sc.uncertainty.mode == "process"
    mig = sc.uncertainty.migration if proc else "historical"  # "with fixed migration" keeps it fixed
    rates = sc.uncertainty.rates if proc else "official_range"
    ens = sc.model_copy(update={"uncertainty": Uncertainty(mode="process", draws=draws, migration=mig, rates=rates)})
    try:
        years, per, _, _ = simulate(pack, ens, get_xp("gpu"))
    except Exception:  # no GPU in this process: the CPU path gives the same distribution, more slowly
        years, per, _, _ = simulate(pack, ens)
    return np.asarray(per[spec.metric][:, years.index(spec.year)], dtype=float)


def solve_probability(parent: Scenario, spec: TargetSpec, probability: float, direction: str | None = None, pack=None, draws: int = 10000) -> dict:
    """Value of the free parameter that gives `probability` of the metric being at least (or at most) the target in the target year.

    Population responds linearly to the assumption, so an outcome is the deterministic path plus a deviation whose distribution
    hardly depends on the parameter. The median must therefore reach target - q (q = the relevant quantile of the deviation);
    the answer is then checked with a full ensemble at the solved value."""
    pack = pack or load_pack()
    direction = direction or ("at_most" if spec.metric in UPPER_IS_WORSE else "at_least")
    first = solve(parent, spec, pack, sensitivity=False)
    if first["status"] != "solved" or spec.metric == "homes_needed_additional":
        return first | {"probability": probability, "direction": direction, "probabilistic": False}
    tgt = first["target"]
    x = _outcomes(pack, apply(parent, spec, first["value"]), spec, draws)
    dev = np.quantile(x, 1 - probability if direction == "at_least" else probability) - np.median(x)
    spec2 = spec.model_copy(update={"target_kind": "value", "target_value": float(tgt - dev)})
    second = solve(parent, spec2, pack, sensitivity=False)
    if second["status"] != "solved":
        return second | {"target": tgt, "probability": probability, "direction": direction, "probabilistic": True, "median_value": first["value"]}
    y = _outcomes(pack, apply(parent, spec, second["value"]), spec, draws)
    hit = float(np.mean(y >= tgt) if direction == "at_least" else np.mean(y <= tgt))
    if abs(hit - probability) > 0.01:  # the spread depends on the parameter (fertility, ratios): one correction step at the new value
        dev2 = np.quantile(y, 1 - probability if direction == "at_least" else probability) - np.median(y)
        third = solve(parent, spec.model_copy(update={"target_kind": "value", "target_value": float(tgt - dev2)}), pack, sensitivity=False)
        if third["status"] == "solved":
            y3 = _outcomes(pack, apply(parent, spec, third["value"]), spec, draws)
            hit3 = float(np.mean(y3 >= tgt) if direction == "at_least" else np.mean(y3 <= tgt))
            if abs(hit3 - probability) < abs(hit - probability):
                second, y, hit, dev = third, y3, hit3, dev2
    return second | {"target": tgt, "base_value": first["base_value"], "curve": first["curve"], "deviation": float(dev), "probability": probability, "direction": direction,
                     "probabilistic": True, "achieved_probability": round(hit, 3), "median_outcome": float(np.median(y)), "median_value": first["value"]}
