"""Reproducible evidence. Writes into $JFE_EVAL_OUT (default evaluation/); `prompts` reads prompts.json from there.

python -m jfe.evidence prompts   # held-out natural-language evaluation (needs the LLM)
python -m jfe.evidence bench     # CPU vs GPU timing on identical workloads
python -m jfe.evidence backtest  # retrospective 2018-2025 check + official-reference emulation
"""
import asyncio
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .engine import CODES, age_forward, load_pack, metric_block, run_experiment, school_inputs, simulate
from .schemas import FIRST_YEAR, Scenario, Uncertainty, default_scenario

OUT = Path(os.environ.get("JFE_EVAL_OUT", "evaluation"))


# ------------------------------------------------------------------------------- prompts
def _check(exp, r):
    sc = r.get("scenario")
    errs = []
    st = r["status"]
    if "status" in exp and st != exp["status"]:
        errs.append(f"status {st} != {exp['status']}")
    if "status_in" in exp and st not in exp["status_in"]:
        errs.append(f"status {st} not in {exp['status_in']}")
    if any(k in exp for k in ("net", "fertility_code", "mortality_code", "fertility_mult", "mortality_mult", "end_year", "mode")) and not sc:
        return errs + ["no scenario"]
    if "solve" in exp and not sc:
        sc = None
    if "net" in exp:
        nets = dict(zip(range(FIRST_YEAR, sc.end_year + 1), sc.net_by_year()))
        for y, v in exp["net"].items():
            if nets.get(int(y)) != v:
                errs.append(f"net {y}: {nets.get(int(y))} != {v}")
    if "end_year" in exp and sc.end_year != exp["end_year"]:
        errs.append(f"end_year {sc.end_year} != {exp['end_year']}")
    if "mode" in exp and sc.uncertainty.mode != exp["mode"]:
        errs.append(f"mode {sc.uncertainty.mode}")
    for k, attr in (("fertility_code", "fertility"), ("mortality_code", "mortality")):
        if k in exp and getattr(sc, attr).assumption != exp[k]:
            errs.append(f"{k} {getattr(sc, attr).assumption} != {exp[k]}")
    for k, attr in (("fertility_mult", "fertility"), ("mortality_mult", "mortality")):
        if k in exp:
            sched = dict(zip(range(FIRST_YEAR, sc.end_year + 1), sc.schedule(getattr(sc, attr))))
            for y, v in exp[k].items():
                if abs(sched[int(y)] - v) > 1e-6:
                    errs.append(f"{k} {y}: {sched[int(y)]:.3f} != {v}")
    if "chart_kind" in exp and (r.get("chart") or {}).get("kind") != exp["chart_kind"]:
        errs.append(f"chart kind {(r.get('chart') or {}).get('kind')} != {exp['chart_kind']}")
    if "chart_metric" in exp and exp["chart_metric"] not in ((r.get("chart") or {}).get("metrics") or []):
        errs.append(f"chart metrics {(r.get('chart') or {}).get('metrics')}")
    if "solve" in exp:
        t = r.get("target") or {}
        for k, v in exp["solve"].items():
            if t.get(k) != v:
                errs.append(f"target.{k} {t.get(k)} != {v}")
    if exp.get("parent_kept") and not sc.parent_scenario_id:
        errs.append("parent not kept")
    return errs


async def prompts():
    import httpx

    from . import compiler
    spec = json.loads((OUT / "prompts.json").read_text())
    parents = {k: Scenario.model_validate({"migration": v["migration"], **{x: v[x] for x in v if x != "migration"}}) for k, v in spec["parents"].items()}
    rows = []
    async with httpx.AsyncClient() as c:
        for p in spec["prompts"]:
            parent = parents.get(p.get("parent"))
            t0 = time.perf_counter()
            try:
                r = await compiler.draft(c, p["prompt"], parent, "run_parent" if parent else None, [])
                errs = _check(p["expect"], r)
            except compiler.LLMUnavailable as e:
                r, errs = {"status": "llm_error", "message": str(e)}, ["llm unavailable"]
            lat = time.perf_counter() - t0
            rows.append({"id": p["id"], "category": p["cat"], "prompt": p["prompt"], "status": r["status"], "pass": not errs,
                         "errors": "; ".join(errs), "latency_s": round(lat, 3), "message": r.get("message", "")[:200]})
            print(f"{'PASS' if not errs else 'FAIL'} {p['id']:4} {lat:5.2f}s {r['status']:20} {'; '.join(errs)}", flush=True)
    with open(OUT / "prompt_evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lat = sorted(r["latency_s"] for r in rows)
    cats = {}
    for r in rows:
        c = cats.setdefault(r["category"], [0, 0])
        c[0] += r["pass"]
        c[1] += 1
    summary = {"model": compiler.LLM_MODEL, "date": time.strftime("%Y-%m-%d %H:%M"), "passed": sum(r["pass"] for r in rows), "total": len(rows),
               "by_category": {k: f"{a}/{b}" for k, a_b in cats.items() for a, b in [a_b]},
               "latency_s": {"p50": lat[len(lat) // 2], "p95": lat[int(len(lat) * 0.95) - 1], "max": lat[-1]},
               "method": "Exact match on listed fields after server-side merge and strict validation; temperature 0; prompts written before tuning and absent from few-shot examples."}
    (OUT / "prompt_evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


# ------------------------------------------------------------------------------- bench
def bench():
    pack = load_pack()
    import cupy
    info = {"cpu": platform.processor() or platform.machine(), "cpu_logical_cores": os.cpu_count(), "numpy": np.__version__, "cupy": cupy.__version__,
            "gpu": cupy.cuda.runtime.getDeviceProperties(0)["name"].decode(), "workload": "two simulations per run (scenario + comparator), 2 sexes x 101 ages, 2026-2040, float64, binomial deaths + Poisson births",
            "timing": "warm; median of repeats; includes host-device transfer and summarisation inside run_experiment compute_seconds (synchronised)", "date": time.strftime("%Y-%m-%d %H:%M")}
    rows = []
    for draws in (1, 64, 256, 1024, 4096, 16384, 65536):
        # model_construct: the benchmark deliberately exceeds the product cap of 4,096 draws
        sc = default_scenario() if draws == 1 else default_scenario(uncertainty=Uncertainty.model_construct(mode="process", draws=draws))
        for backend in ("cpu", "gpu"):
            run_experiment(sc, pack, backend)
            reps = 5 if draws <= 4096 else 3
            ts = sorted(run_experiment(sc, pack, backend)["receipt"]["compute_seconds"] for _ in range(reps))
            rows.append({"draws": draws, "backend": backend, "median_s": ts[len(ts) // 2], "min_s": ts[0], "max_s": ts[-1], "repeats": reps})
            print(rows[-1], flush=True)
    for r in rows:
        cpu = next(x for x in rows if x["draws"] == r["draws"] and x["backend"] == "cpu")["median_s"]
        r["speedup_vs_cpu"] = round(cpu / r["median_s"], 2)
    (OUT / "latency_benchmark.json").write_text(json.dumps({"environment": info, "results": rows}, indent=2))


# ------------------------------------------------------------------------------- backtest
def backtest(write=True):
    pack = load_pack()
    md = CODES.index("MD")
    years = [int(y) for y in pack.comp_years]
    comp = {y: dict(zip(("births", "deaths", "net", "adj", "pop"), pack.comp[i])) for i, y in enumerate(years)}
    oy = [int(y) for y in pack.obs_years]
    start = 2017
    N = pack.obs[oy.index(start)].copy()
    supp = int(np.isnan(N).sum())
    N = np.nan_to_num(N, nan=5.0)  # labelled midpoint assumption for suppressed small cells (<10)
    q, M0, P = pack.q[md, 0], pack.M0[md, md, 0], pack.P[md, md, 0]  # 2026 official mid-range levels, held fixed
    pm = pack.sex_ratio / (1 + pack.sex_ratio)
    out = []
    for variant, netf in (("observed net migration", lambda y: comp[y]["net"] + comp[y]["adj"]), ("assumed +400 every year", lambda y: 400.0)):
        n = N.copy()
        rows = []
        for y in range(start + 1, 2026):
            aged = age_forward(np, n)
            deaths = aged * q
            post = aged - deaths
            post[0, 0], post[1, 0] = comp[y]["births"] * (1 - pm), comp[y]["births"] * pm
            n = post + M0 + netf(y) * P
            m = metric_block(np, n, pack.ltc_rate, school_inputs(pack))
            obs = metric_block(np, np.nan_to_num(pack.obs[oy.index(y)], nan=5.0), pack.ltc_rate, school_inputs(pack))
            rows.append({"year": y, "model_deaths": round(float(deaths.sum()), 1), "observed_deaths": comp[y]["deaths"],
                         **{f"model_{k}": round(float(m[k]), 1) for k in ("population_total", "population_0_15", "population_16_64", "population_65_plus", "population_80_plus")},
                         **{f"observed_{k}": round(float(obs[k]), 1) for k in ("population_total", "population_0_15", "population_16_64", "population_65_plus", "population_80_plus")}})
        last = rows[-1]
        err = {k: round(last[f"model_{k}"] - last[f"observed_{k}"], 1) for k in ("population_total", "population_0_15", "population_16_64", "population_65_plus", "population_80_plus")}
        out.append({"variant": variant, "end_2025_error_model_minus_observed": err,
                    "end_2025_relative_error_pct": {k: round(100 * v / last[f"observed_{k}"], 2) for k, v in err.items()}, "years": rows})
    # one-year check of the official projection itself
    off25 = metric_block(np, pack.off_N[2, md, md, 0], pack.ltc_rate, school_inputs(pack))
    obs25 = metric_block(np, pack.base_2025, pack.ltc_rate, school_inputs(pack))
    official_check = {k: {"official_projection_2025": round(float(off25[k]), 1), "observed_2025": round(float(obs25[k]), 1)} for k in ("population_total", "population_16_64", "population_65_plus")}
    # reference emulation (NOT validation: parameters were derived from these same outputs)
    years_e, per, _, _ = simulate(pack, default_scenario(baseline_id="official_state_2025", end_year=2060))
    off = pack.off_N[2, md, md, : len(years_e)]
    om = metric_block(np, off, pack.ltc_rate, school_inputs(pack))
    emu_years = [{"year": y, **{f"engine_{k}": round(float(per[k][0][i]), 1) for k in ("population_total", "population_16_64", "population_65_plus")},
                  **{f"official_{k}": round(float(om[k][i]), 1) for k in ("population_total", "population_16_64", "population_65_plus")}} for i, y in enumerate(years_e)]
    if write:
        with open(OUT / "reference_comparison.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["year", "metric_id", "engine_emulation", "official_published", "engine_minus_official"])
            for i, y in enumerate(years_e):
                for k in ("population_total", "population_0_15", "population_16_64", "population_65_plus"):
                    w.writerow([y, k, round(float(per[k][0][i]), 3), round(float(om[k][i]), 3), round(float(per[k][0][i] - om[k][i]), 3)])
    emu_max = {k: round(float(np.abs(per[k][0] - om[k]).max()), 2) for k in ("population_total", "population_0_15", "population_16_64", "population_65_plus")}
    sq = pack.meta["quality"]["schools"]["by_year"]
    y0, y1 = min(sq, key=int), max(sq, key=int)
    school_check = {k: {"ratio_held_from": y0, "predicted_pupils": round(sq[y0][f"{k}_ratio"] * sq[y1][f"{k}_pupils"] / sq[y1][f"{k}_ratio"]),
                        "actual_pupils": sq[y1][f"{k}_pupils"], "year": y1} for k in ("primary", "secondary")}
    for v in school_check.values():
        v["error_pct"] = round(100 * (v["predicted_pupils"] - v["actual_pupils"]) / v["actual_pupils"], 1)
    from .schemas import Migration, MigrationSegment
    h = run_experiment(Scenario(title="PHI cross-check", end_year=2043, migration=Migration(segments=[MigrationSegment(from_year=2026, to_year=2043, net_per_year=325)])), pack)
    hy, hb, hg = h["years"], h["metrics"]["scenario"]["hospital_bed_days"]["value"], h["metrics"]["scenario"]["gp_appointments"]["value"]
    health_check = {"note": "Engine (end-2025 base, +325 net migration, mid-range) versus Public Health Intelligence Disease Projection Report central scenario (+325, 2023 base), Figs. 40 and 5.",
                    "bed_days": {"engine": {y: round(hb[hy.index(y)]) for y in (2025, 2033, 2043)}, "phi": {2023: 60300, 2033: 68950, 2043: 78090},
                                 "growth_pct": {"engine_2025_2043": round(100 * (hb[-1] / hb[0] - 1), 1), "phi_2023_2043": round(100 * (78090 / 60300 - 1), 1)}},
                    "gp_appointments": {"engine": {y: round(hg[hy.index(y)]) for y in (2025, 2033, 2043)}, "phi": {2023: 410790, 2033: 430000, 2043: 446860}}}
    # year-group cohort check (S41): pupils by year group, spring 2024/25, vs residents of matching age at end-2024
    man = json.loads(Path(os.environ.get("JFE_MANIFEST", "data/manifests/sources.json")).read_text())
    t41 = next(d for d in man["documents"] if d["id"] == "S41")["table"]
    p24 = np.nan_to_num(pack.obs[oy.index(2024)]).sum(0)
    yg = {}
    for g, (name, pupils) in enumerate(zip(t41["year_groups"], t41["pupils"])):
        a = 4 + g  # Reception starts at age 4 in September
        residents = 8 / 12 * p24[a] + 4 / 12 * p24[a + 1]
        yg[name] = round(pupils / residents, 3)
    # vintage check (S42): previous round (+325) vs observed
    vint = {}
    raw = Path(os.environ.get("JFE_RAW", "data/raw")) / "population-projections-2023-to-2080.csv"
    if raw.exists():
        tot = {}
        for r in csv.DictReader(open(raw, newline="", encoding="utf-8-sig")):
            if r["Scenario"].strip().startswith("325") and int(r["Year"]) in (2023, 2024, 2025):
                tot[int(r["Year"])] = tot.get(int(r["Year"]), 0.0) + float(r["People"])
        vint = {y: {"projected_2023_round_+325": round(v), "observed": round(float(np.nansum(pack.obs[oy.index(y)])))} for y, v in sorted(tot.items())}
    result = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "pack_id": pack.id, "health_cross_check": health_check,
        "school_year_group_check": {"note": "Spring 2024/25 pupils by year group (S41 p.5) / residents of matching school age at end-2024 (month-of-birth weights). Reception-Y11 near 1 supports the weighting; Y12-13 fall with post-16 coverage.", "ratios": yg},
        "vintage_check": {"note": "Previous official round (Dec 2023, +325 scenario) versus observed estimates (Sept 2026 vintage, published cells).", "values": vint},
        "pupil_proxy_check": {"note": f"Pupils predicted for {y1} from the observed school-age population using the {y0} participation ratio, versus actual {y1} pupils (12 years ahead, population known).", "values": school_check},
        "retrospective": {"description": f"Start from observed end-{start}; apply observed births and observed net migration (incl. administrative adjustment) each year 2018-2025, "
                          "with mortality and the migration age/sex profile held at the official 2026 mid-range levels; compare with observed estimates. "
                          "Tests the age-structure mechanics and migration profile; the observed component totals are inputs, so total population matching is expected.",
                          "suppressed_cells_at_start": supp, "variants": out},
        "official_projection_one_year_check": {"note": "Official Feb 2026 projection (+400, from its own end-2024 base) versus the Sept 2026 observed estimate for end-2025; actual 2025 net migration was +182.", "values": official_check},
        "reference_emulation": {"note": "Engine started from the official end-2025 state with official mid-range assumptions, 2026-2060. Mortality and migration are derived from these outputs, so agreement is an emulation check, not independent validation. Remaining differences come from the published 2023-25 fertility age pattern.",
                                "max_abs_difference_people": emu_max, "years": emu_years},
    }
    if not write:
        return result
    (OUT / "backtest.json").write_text(json.dumps(result, indent=2))
    for v in out:
        print(v["variant"], v["end_2025_error_model_minus_observed"], v["end_2025_relative_error_pct"])
    print("deaths model vs observed:", [(r["year"], r["model_deaths"], r["observed_deaths"]) for r in out[0]["years"]])
    print("official one-year check:", official_check)
    print("emulation max abs diff:", emu_max)
    print("pupil proxy check:", school_check)


def datasets():
    """Render docs/DATASETS.md from the checksummed manifest and the built pack (single source of truth)."""
    pack = load_pack()
    man = json.loads(Path(os.environ.get("JFE_MANIFEST", "data/manifests/sources.json")).read_text())
    q = pack.meta["quality"]
    lines = ["# Datasets used by Vraic Futures Engine", "",
             f"Generated from `data/manifests/sources.json` and data pack `{pack.id}` (built {pack.meta['built']}). "
             "Every machine-readable file is downloaded by `python -m jfe.ingest` and refused if its SHA-256 differs from the pinned value. "
             "Nothing is fetched at runtime.", "",
             "## Machine-readable sources (inputs to the data pack)", "",
             "| ID | Dataset | Publisher | Resource ID | Vintage | Licence | Used for | Bytes | SHA-256 |", "|---|---|---|---|---|---|---|---:|---|"]
    for s in man["sources"]:
        lines.append(f"| {s['id']} | [{s['title']}]({s['url']}) | {s['publisher']} | `{s['resource_id']}` | {s['vintage']} | {s['licence']} | {s['use']} | {s.get('bytes') or ''} | `{s.get('sha256') or ''}` |")
    lines += ["", "## Documents (definitions, assumptions and one transcribed table)", "", "| ID | Document | Used for | SHA-256 |", "|---|---|---|---|"]
    for d in man["documents"]:
        lines.append(f"| {d['id']} | [{d['title']}]({d['url']}) | {d['use']} | `{d.get('sha256') or 'not archived'}` |")
        if d.get("table"):
            t = d["table"]
            lines += ["", f"Values transcribed from {d['id']}: `" + json.dumps(t) + "`", ""]
    lines += ["", "## How each dataset enters the model", "",
              "| Parameter | Provenance class | Derivation |", "|---|---|---|",
              "| Baseline population end-2025 | published (+3 imputed cells) | S06 single-year age/sex counts; suppressed male 98/99/100+ cells filled from the official 2025 projected counts (labelled modelling assumption) |",
              "| Death probabilities by year/age/sex/code | derived | S07 Deaths / population aged forward (event order verified against S04) for each life-expectancy code |",
              "| Net migration age/sex profile | derived | Exact decomposition M = M_nil + net x P from S07-nil and S07 (+400); verified on held-out +200/+600/+800 to 4e-5 people |",
              f"| Fertility age pattern | published | S34 Table 2 (2023-25 ASFRs, uniform within 5-year bands) |",
              f"| Fertility level by year/code | derived | Scale factor calibrated to official births (S07 cubes net 0/400/800); held-out births error <= {max(q['fertility']['held_out_max_abs_births_error_2026_2040'].values()):.2f}/yr; implied 2034 TFR {q['fertility']['implied_tfr_2034_MD']:.3f} vs published 1.01 |",
              "| Birth sex ratio | derived | S07 births by sex (1.0408 M per F, constant) |",
              "| LTC claim rates 2024 | published / derived | S08 2024 open claims by 65+ band and sex / S06 2024 population (same vintage) |",
              "| School participation ratios | derived | S36 January pupils / school-age residents (year-group weights); 2024 ratio held constant |",
              "| Hospital bed days and GP appointments | published rates (transcribed) | S37 Figs. 39 and 4 per-person rates by 5-year age band and sex |",
              "| Hospital nurse and doctor FTE proxies | published baseline (transcribed) | S38 Table 1 Health and Care Jersey FTE at 31 Dec 2025, held constant per age-weighted bed day |",
              "| Homes needed by dwelling type | derived | S39 exact decomposition H = H_nil + net x H_per_person per year and type (verified on held-out +200/+600/+800); change since 2025 |",
              "| Year-group and vintage checks | published | S41 pupils by year group 2024/25; S42 previous-round projection (validation only) |",
              "| Observed history and backtest inputs | published | S06 estimates 2011-2025; S35 births, deaths, net migration 2018-2025 (retrospective check only) |",
              "", "## Data-quality checks (all must pass for the pack to build)", ""]
    lines += [f"- {'PASS' if c['pass'] else 'FAIL'}: {c['check']}" for c in q["checks"]]
    lines += [""]
    Path("docs/DATASETS.md").write_text("\n".join(lines))
    print("wrote docs/DATASETS.md")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    cmd = sys.argv[1]
    {"prompts": lambda: asyncio.run(prompts()), "bench": bench, "backtest": backtest, "datasets": datasets}[cmd]()
