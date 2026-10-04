"""Validated chart specifications -> exact observation tables built from stored results (no re-simulation).

The same table drives the on-screen chart, the "Show as table" view and the "Download chart data" CSV."""
import io
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .engine import METRICS, UNITS

Kind = Literal["trajectory", "fan", "difference", "pyramid", "ranking", "reconciliation", "response", "frontier", "housing"]
SERIES = ("scenario", "comparator", "official", "observed")
CHART_METRICS = list(METRICS)
DEFINITIONS = METRICS | {
    "old_age_ratio": "People aged 65+ per 100 people aged 16-64 (demographic convention, not financial dependency)",
    "dependency_ratio": "People aged 0-15 plus 65+ per 100 people aged 16-64 (demographic convention, not financial dependency)",
}
BANDS = [(lo, lo + 4) for lo in range(0, 100, 5)] + [(100, 100)]
SHORT = {  # concise chart titles
    "population_total": "Total population", "population_0_15": "Population aged 0-15", "population_16_64": "Population aged 16-64",
    "population_65_plus": "Population aged 65+", "population_80_plus": "Population aged 80+", "old_age_ratio": "Old-age ratio (65+ per 100 aged 16-64)",
    "dependency_ratio": "Dependency ratio (0-15 and 65+ per 100 aged 16-64)", "ltc_claims_index_65plus": "65+ care claim index (2025 = 100)",
    "school_age_primary": "Primary school-age population", "school_age_secondary": "Secondary school-age population",
    "pupils_primary_proxy": "Primary pupils", "pupils_secondary_proxy": "Secondary pupils", "hospital_bed_days": "Acute hospital bed days",
    "gp_appointments": "GP appointments", "nurses_fte_proxy": "Nurses and midwives (FTE)", "doctors_fte_proxy": "Doctors (FTE)", "births": "Births",
    "homes_needed_additional": "Additional homes needed since end-2025", "deaths": "Deaths", "net_migration": "Net migration",
}


def band_label(lo, hi):
    return "100+" if lo == 100 else f"{lo}-{hi}"


class ChartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Kind
    metrics: list[Literal[tuple(CHART_METRICS)]] = Field(default_factory=list, max_length=3)  # type: ignore[valid-type]
    series: list[Literal[SERIES]] = Field(default_factory=list, max_length=4)  # type: ignore[valid-type]
    years: list[int] = Field(default_factory=list, max_length=2)
    basis: Literal["vs_comparator", "vs_start"] = "vs_start"
    view: Literal["chart", "table"] = "chart"
    run_id: str = "baseline"
    compare_run_ids: list[str] = Field(default_factory=list, max_length=2)
    target: dict | None = None  # inverse-simulation target (response, frontier)
    base_scenario: dict | None = None  # conditions stated with the target ("under low fertility"); else the run's scenario


class ChartError(ValueError):
    pass


def _agg(result):
    return "value" if "value" in result["metrics"]["scenario"]["population_total"] else "p50"


def _provenance(result, run_id):
    r = result["receipt"]
    return {"run_id": run_id, "experiment_hash": result["experiment_hash"], "pack_id": result["pack_id"], "model_version": result["model_version"],
            "backend": r["backend"], "draws": r["completed_draws"], "seed": r["seed"], "aggregation": result["aggregation"],
            "scenario_title": result["scenario"]["title"], "comparator": result["comparator_label"], "official": result["official"]["label"]}


def build(spec: ChartSpec, result: dict, others: dict[str, dict] | None = None, backtest: dict | None = None) -> dict:
    """-> {spec, title, unit, columns, rows, series, caption, definitions, provenance, notes[, extra]}"""
    others = others or {}
    extra = None
    agg = _agg(result)
    years = result["years"]
    notes = []
    if spec.kind in ("trajectory", "fan", "difference"):
        spec.metrics = spec.metrics or ["population_total"]
    if spec.kind == "trajectory":
        if len({UNITS[m] for m in spec.metrics}) > 1:
            raise ChartError("Those metrics have different units; plot them one at a time.")
        series = spec.series or list(SERIES)
        cols, rows = ["year"], {y: [y] for y in sorted(set(years) | set(result["observed"]["years"]) | set(result["official"]["years"]))}
        roles = []
        for m in spec.metrics:
            for s in series:
                if s in ("official", "observed"):
                    vals = dict(zip(result[s]["years"], result[s]["metrics"].get(m, [])))
                else:
                    vals = dict(zip(years, result["metrics"][s][m][agg]))
                name = f"{m}:{s}"
                cols.append(name)
                roles.append({"column": name, "metric": m, "role": s, "label": _label(s, result)})
                for y in rows:
                    rows[y].append(_r(vals.get(y)))
            for rid, other in others.items():
                if other["pack_id"] != result["pack_id"]:
                    raise ChartError("That comparison would mix data vintages, so it is not allowed.")
                vals = dict(zip(other["years"], other["metrics"]["scenario"][m][_agg(other)]))
                name = f"{m}:run:{rid}"
                cols.append(name)
                roles.append({"column": name, "metric": m, "role": "other_run", "label": other["scenario"]["title"]})
                for y in rows:
                    rows[y].append(_r(vals.get(y)))
        rows = [r for r in rows.values() if any(v is not None for v in r[1:])]
        title = " / ".join(SHORT.get(m, METRICS[m]) for m in spec.metrics)
        unit = UNITS[spec.metrics[0]]
        caption = "Observed: Statistics Jersey estimates (Sept 2026). Official: Statistics Jersey projection (Feb 2026)." + (
            f" Scenario and comparator: median of {result['receipt']['completed_draws']:,} simulations." if agg == "p50" else "")
    elif spec.kind == "fan":
        if agg != "p50":
            raise ChartError("A fan chart needs a process-variation run. Re-run this scenario with process variation (256 draws) first.")
        m = spec.metrics[0]
        blk = result["metrics"]["scenario"][m]
        cols = ["year", "p05", "p25", "p50", "p75", "p95", "mean", "comparator_p50"]
        rows = [[y, *(_r(blk[q][i]) for q in ("p05", "p25", "p50", "p75", "p95", "mean")), _r(result["metrics"]["comparator"][m]["p50"][i])] for i, y in enumerate(years)]
        roles = [{"column": c, "metric": m, "role": c, "label": c} for c in cols[1:]]
        title, unit = SHORT.get(m, METRICS[m]), UNITS[m]
        varies = (result["scenario"].get("uncertainty") or {}).get("migration") == "historical"
        unc = result["scenario"].get("uncertainty") or {}
        caption = (f"Median; pointwise 50% and 90% ranges; {result['receipt']['completed_draws']:,} simulations: random births and deaths"
                   + ("; net migration as in 2001-2025" if varies else "") + ("; fertility and mortality levels" if unc.get("rates") == "official_range" else "") + ".")
    elif spec.kind == "difference":
        if len({UNITS[m] for m in spec.metrics}) > 1:
            raise ChartError("Those metrics have different units; show their differences one at a time.")
        cols, roles = ["year"], []
        for m in spec.metrics:
            cols.append(f"{m}:difference")
            roles.append({"column": f"{m}:difference", "metric": m, "role": "difference", "label": METRICS[m]})
            if agg == "p50":
                cols += [f"{m}:difference_p05", f"{m}:difference_p95"]
                roles += [{"column": f"{m}:difference_p05", "metric": m, "role": "band_low", "label": "5th percentile"},
                          {"column": f"{m}:difference_p95", "metric": m, "role": "band_high", "label": "95th percentile"}]
        rows = []
        for i, y in enumerate(years):
            row = [y]
            for m in spec.metrics:
                d = result["metrics"]["difference"][m]
                row.append(_r(d[agg][i]))
                if agg == "p50":
                    row += [_r(d["p05"][i]), _r(d["p95"][i])]
            rows.append(row)
        title, unit = "Difference from baseline: " + " / ".join(SHORT.get(m, METRICS[m]) for m in spec.metrics), UNITS[spec.metrics[0]]
        caption = "Scenario minus baseline" + (", paired simulations." if agg == "p50" else ".")
    elif spec.kind in ("pyramid", "ranking"):
        ys = spec.years or ([years[0], years[-1]] if spec.kind == "pyramid" else [years[-1]])
        bad = [y for y in ys if y not in years]
        if bad:
            raise ChartError(f"{bad[0]} is outside this run ({years[0]}-{years[-1]}).")
        sc, cp = result["age"]["scenario"], result["age"]["comparator"]
        if spec.kind == "pyramid":
            cols = ["age_band", "sex", "year", "scenario", "comparator"]
            rows = [[band_label(lo, hi), sex, y, _r(sum(sc[years.index(y)][s][lo:hi + 1])), _r(sum(cp[years.index(y)][s][lo:hi + 1]))]
                    for y in ys for s, sex in enumerate(("F", "M")) for lo, hi in BANDS]
            roles = [{"column": "scenario", "role": "scenario", "label": "Scenario"}, {"column": "comparator", "role": "comparator", "label": "Comparator"}]
            title, unit = "Population by age band and sex, " + " and ".join(str(y) for y in ys), "people"
            caption = "Common x-axis scale. Bars: scenario; outline: baseline."
        else:
            y = ys[0]
            i, i0 = years.index(y), 0
            cols = ["age_band", "scenario", "comparator", "difference_vs_comparator", f"scenario_{years[0]}", "change_vs_start", "change_vs_start_pct"]
            rows = []
            for lo, hi in BANDS:
                s_ = sum(sum(sc[i][s][lo:hi + 1]) for s in range(2))
                c_ = sum(sum(cp[i][s][lo:hi + 1]) for s in range(2))
                s0 = sum(sum(sc[i0][s][lo:hi + 1]) for s in range(2))
                rows.append([band_label(lo, hi), _r(s_), _r(c_), _r(s_ - c_), _r(s0), _r(s_ - s0), _r(100 * (s_ - s0) / s0 if s0 else None)])
            key = 3 if spec.basis == "vs_comparator" else 5
            rows.sort(key=lambda r: -abs(r[key] or 0))
            roles = [{"column": cols[key], "role": "difference", "label": cols[key]}]
            title = f"Change by age band, {y} " + ("vs baseline" if spec.basis == "vs_comparator" else f"vs {years[0]}")
            unit = "people"
            caption = "Five-year age bands, both sexes."
            if spec.basis == "vs_comparator" and all(abs(r[3] or 0) < 0.5 for r in rows):
                notes.append("Scenario equals baseline.")
    elif spec.kind == "housing":
        h = result.get("housing") or {}
        if not h.get("years"):
            raise ChartError("Homes needed are available for 2026-2040 only.")
        y = spec.years[0] if spec.years else h["years"][-1]
        if y not in h["years"]:
            raise ChartError(f"Homes needed are available for {h['years'][0]}-{h['years'][-1]}.")
        i = h["years"].index(y)
        cols = ["dwelling_type", "scenario", "comparator", "difference"]
        rows = [[t, _r(a), _r(b), _r(a - b)] for t, a, b in zip(h["types"], h["scenario"][i], h["comparator"][i])]
        roles = [{"column": "scenario", "role": "scenario", "label": "Scenario"}, {"column": "comparator", "role": "comparator", "label": "Comparator"}]
        title, unit = f"Additional homes needed since end-2025, by dwelling type, {y}", "homes"
        caption = "Statistics Jersey housing-needs method; depends on net migration only; need, not supply."
        spec.years = [y]
    elif spec.kind in ("response", "frontier"):
        from .inverse import TargetSpec, frontier, solve
        from .schemas import Scenario
        if not spec.target:
            raise ChartError("Ask a target question first, for example: what net migration keeps the working-age population at today's level by 2040?")
        t = TargetSpec.model_validate(spec.target)
        parent = Scenario.model_validate((spec.base_scenario or result["scenario"]) | {"solved_for": None})
        name = SHORT.get(t.metric, METRICS[t.metric])
        unit = UNITS[t.metric]
        if spec.kind == "response" and t.probability:
            from .inverse import solve_probability
            plain = t.model_copy(update={"probability": None, "direction": None})
            r = solve_probability(parent, plain, t.probability, t.direction)
            q = round(100 * (1 - t.probability if r.get("direction", "at_least") == "at_least" else t.probability))
            dv = r.get("deviation", 0.0)
            cols = ["parameter_value", "metric_value", "metric_quantile", "target"]
            rows = [[a, _r(b), _r(b + dv) if b is not None else None, _r(r["target"])] for a, b in r["curve"]]
            roles = [{"column": "metric_value", "role": "comparator", "label": "Median"}, {"column": "metric_quantile", "role": "scenario", "label": f"{q}th percentile"},
                     {"column": "target", "role": "target", "label": "Target"}]
            title = f"{name} in {t.year} by {r['parameter_label'].split(' (')[0].lower()}"
            caption = f"{q}th percentile ({round(100 * t.probability)}% chance); median dashed."
            sol = {"status": r.get("status"), "value": r.get("value"), "achieved": r.get("target") if r.get("status") == "solved" else None, "target": r.get("target"),
                   "closest_value": r.get("closest_value"), "closest_metric": r.get("closest_metric")}
            extra = {"x_label": r["parameter_label"], "solution": sol, "probability": t.probability}
        elif spec.kind == "response":
            r = solve(parent, t, sensitivity=False)
            cols = ["parameter_value", "metric_value", "target"]
            rows = [[a, _r(b), _r(r["target"])] for a, b in r["curve"]]
            roles = [{"column": "metric_value", "role": "scenario", "label": name}, {"column": "target", "role": "target", "label": "Target"}]
            title = f"{name} in {t.year} by {r['parameter_label'].split(' (')[0].lower()}"
            caption = "Deterministic; other assumptions as in the parent scenario."
            extra = {"x_label": r["parameter_label"], "solution": {k: r.get(k) for k in ("status", "value", "achieved", "target", "closest_value", "closest_metric")}}
        else:
            fr = frontier(parent, t)
            ycol = "fertility_multiplier" if "Fertility" in fr["y_label"] else "mortality_multiplier"
            cols = ["net_migration", ycol, "metric_value"]
            rows = [[x, y, z] for j, y in enumerate(fr["y"]) for i, x in enumerate(fr["x"]) for z in [fr["z"][j][i]]]
            roles = [{"column": "metric_value", "role": "scenario", "label": name}]
            title = f"{name} in {t.year}"
            caption = f"Deterministic; {fr['scenarios']:,} scenarios; contour at the target."
            extra = {"x_label": fr["x_label"], "y_label": fr["y_label"], "target": _r(fr["target"]), "x": fr["x"], "y": fr["y"]}
        spec.metrics = [t.metric]
    elif spec.kind == "reconciliation":
        if not backtest:
            raise ChartError("Reconciliation data is unavailable.")
        em = backtest["reference_emulation"]
        groups = [k.removeprefix("engine_") for k in em["years"][0] if k.startswith("engine_")]
        cols = ["year", "age_group", "engine", "official", "engine_minus_official"]
        rows = [[r["year"], g, r[f"engine_{g}"], r[f"official_{g}"], round(r[f"engine_{g}"] - r[f"official_{g}"], 1)] for r in em["years"] for g in groups]
        roles = [{"column": "engine_minus_official", "role": "difference", "label": "Engine minus official"}]
        title, unit = "Engine vs official projection", "people"
        caption = "Engine minus published projection (+400, mid-range), both from the official end-2025 state."
    for m in spec.metrics:
        if m in ("pupils_primary_proxy", "pupils_secondary_proxy"):
            notes.append("Pupils at January-2024 participation; not capacity.")
        if m == "homes_needed_additional":
            notes.append("Depends on net migration only (average to date for changing paths); to 2040.")
        if m == "ltc_claims_index_65plus":
            notes.append("Open LTC claims at 2024 rates; not care need.")
    out = {"spec": spec.model_dump(), "title": title, "unit": unit, "columns": cols, "rows": rows, "series": roles, "caption": caption,
            "definitions": {m: DEFINITIONS[m] for m in spec.metrics}, "provenance": _provenance(result, spec.run_id), "notes": notes,
            "rounding": "Values stored to 3 decimal places; the display may round further."}
    if spec.kind in ("response", "frontier"):
        out["extra"] = extra
    return out


def _label(s, result):
    return {"scenario": result["scenario"]["title"], "comparator": "Baseline", "official": result["official"]["label"],
            "observed": "Observed estimates"}[s]


def _r(v):
    return None if v is None else round(float(v), 3)


def to_csv(chart: dict) -> str:
    from .exports import write_csv
    p = chart["provenance"]
    head = "".join(f"# {k}: {v}\n" for k, v in [("title", chart["title"]), ("unit", chart["unit"]), ("caption", chart["caption"]), ("run_id", p["run_id"]),
                                                 ("experiment_hash", p["experiment_hash"]), ("data_pack", p["pack_id"]), ("model", p["model_version"]),
                                                 ("backend", p["backend"]), ("draws", p["draws"]), ("seed", p["seed"]), ("aggregation", p["aggregation"]),
                                                 ("rounding", chart["rounding"])] + [(f"definition {k}", v) for k, v in chart["definitions"].items()])
    body = write_csv(([("" if v is None else v) for v in r] for r in chart["rows"]), chart["columns"])
    return head + body


if __name__ == "__main__":  # quick self-check against a fresh deterministic run
    from .engine import run_experiment
    from .schemas import default_scenario
    res = run_experiment(default_scenario())
    for k in ("trajectory", "difference", "pyramid", "ranking"):
        c = build(ChartSpec(kind=k, metrics=["population_16_64"] if k in ("trajectory", "difference") else []), res)
        print(k, c["title"], len(c["rows"]), c["columns"][:4])
    print(io.StringIO(to_csv(build(ChartSpec(kind="trajectory", metrics=["population_16_64"]), res))).read()[:400])
