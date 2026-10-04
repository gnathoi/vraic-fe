"""Portable evidence: tidy CSVs, scenario/result JSON, a self-contained HTML briefing with SVG plots,
and the experiment ZIP. Everything is rendered from the stored result object, never recomputed."""
import csv
import hashlib
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .engine import METRICS, UNITS, load_pack  # noqa: E402
from .schemas import Scenario, describe  # noqa: E402

ART = Path(os.environ.get("JFE_ARTIFACTS", "artifacts"))
KINDS = {
    "report.html": "text/html; charset=utf-8", "metrics.csv": "text/csv; charset=utf-8",
    "age_distribution.csv": "text/csv; charset=utf-8", "scenario.json": "application/json",
    "result.json": "application/json", "zip": "application/zip", "manifest.json": "application/json",
    "assumptions.csv": "text/csv; charset=utf-8", "comparisons.csv": "text/csv; charset=utf-8",
}
COL = {"scenario": "#0066cc", "comparator": "#6a6e73", "official": "#8476d1", "observed": "#151515"}
CAVEATS = [
    "Conditional scenario from a reduced cohort model; not a forecast or an official projection.",
    "Mortality and migration profile derived from the official Feb 2026 projection; fertility pattern from 2023-25 Jersey rates.",
    "Net migration follows the stated path (in ensembles it varies around the path as in 2001-2025) with a fixed age/sex profile; gross flows are not modelled.",
    "Pupils, LTC claims, bed days, staff and homes hold observed rates or the official method constant; they are not capacity, need or supply.",
    "No housing, labour-market or policy feedback.",
]
env = Environment(loader=FileSystemLoader(Path(__file__).parent / "templates"), autoescape=select_autoescape())
LOGO = (Path(__file__).parent / "templates" / "logo-64.b64").read_text().strip()


def safe_text(v):
    """Neutralise spreadsheet formula injection in text cells; numbers stay numeric."""
    if isinstance(v, str) and v[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + v
    return v


def write_csv(rows, header):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    for r in rows:
        w.writerow([safe_text(x) for x in r])
    return buf.getvalue()


def metrics_rows(run_id, result):
    agg = list(result["metrics"]["scenario"]["population_total"].keys())
    for series in ("scenario", "comparator", "difference"):
        for mid, block in result["metrics"][series].items():
            for a in agg:
                for y, v in zip(result["years"], block.get(a, ())):  # homes have no per-draw probability
                    if v is not None:
                        yield [run_id, series, mid, UNITS[mid], y, a, round(v, 6)]
    for series in ("official", "observed"):
        src = result[series]
        for mid, vals in src["metrics"].items():
            for y, v in zip(src["years"], vals):
                if v is not None:
                    yield [run_id, series, mid, UNITS.get(mid, ""), y, "published" if series == "official" else "observed", round(v, 6)]


def age_rows(run_id, result):
    for series in ("scenario", "comparator"):
        for y, grid in zip(result["years"], result["age"][series]):
            for s, sex in enumerate(("F", "M")):
                for a, v in enumerate(grid[s]):
                    yield [run_id, series, y, sex, a if a < 100 else "100+", result["age"]["aggregation"], v]


def svg(fig):
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    s = buf.getvalue()
    return s[s.index("<svg"):]


def _style(ax, title, unit):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_ylabel(unit)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6)


def plot_pyramid(result):
    years, end = result["years"], result["years"][-1]
    s, c = result["age"]["scenario"][-1], result["age"]["comparator"][-1]
    bands = [(lo, min(lo + 4, 100)) for lo in range(0, 100, 5)] + [(100, 100)]
    lab = [f"{lo}-{hi}" if lo < 100 else "100+" for lo, hi in bands]
    grp = lambda g, sex: [sum(g[sex][lo:hi + 1]) for lo, hi in bands]  # noqa: E731
    fig, ax = plt.subplots(figsize=(7, 5.4))
    y = range(len(bands))
    ax.barh(y, [-v for v in grp(s, 0)], color=COL["scenario"], height=0.85, label="Scenario")
    ax.barh(y, grp(s, 1), color=COL["scenario"], height=0.85)
    ax.step([-v for v in grp(c, 0)], y, where="mid", color="#151515", linewidth=1, label="Baseline")
    ax.step(grp(c, 1), y, where="mid", color="#151515", linewidth=1)
    ax.set_yticks(list(y), lab, fontsize=7)
    m = max(max(grp(s, 0)), max(grp(s, 1)), max(grp(c, 0)), max(grp(c, 1))) * 1.08
    ax.set_xlim(-m, m)
    ax.set_xticks(ax.get_xticks(), [f"{abs(int(t)):,}" for t in ax.get_xticks()])
    ax.text(0.02, 0.98, "Female", transform=ax.transAxes, va="top")
    ax.text(0.98, 0.98, "Male", transform=ax.transAxes, va="top", ha="right")
    ax.axvline(0, color="#151515", linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _style(ax, f"Population by five-year age band, end-{end}", "")
    ax.set_xlabel(f"People ({result['age']['aggregation']}; source sex categories)")
    return svg(fig)


def plot_trajectories(result):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    agg = "value" if "value" in result["metrics"]["scenario"]["population_total"] else "mean"
    for ax, mid in zip(axes, ("population_total", "population_16_64", "population_65_plus")):
        ob, of = result["observed"], result["official"]
        ax.plot(ob["years"], [v if v is not None else float("nan") for v in ob["metrics"][mid]], "o", ms=2.5, color=COL["observed"], label="Observed")
        ax.plot(of["years"], of["metrics"][mid], color=COL["official"], linewidth=1.2, label="Official +400 (Feb 2026)")
        ax.plot(result["years"], result["metrics"]["comparator"][mid][agg], "--", color=COL["comparator"], linewidth=1.2, label="Baseline")
        sm = result["metrics"]["scenario"][mid]
        if agg == "mean":
            ax.fill_between(result["years"], sm["p05"], sm["p95"], color=COL["scenario"], alpha=0.15, linewidth=0, label="90% process interval")
        ax.plot(result["years"], sm[agg], color=COL["scenario"], linewidth=2, label="Scenario")
        _style(ax, METRICS[mid], "people")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return svg(fig)


def plot_difference(result):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    d = result["metrics"]["difference"]
    agg = "value" if "value" in d["population_total"] else "mean"
    for mid, ls in (("population_total", "-"), ("population_16_64", "--"), ("population_65_plus", ":")):
        ax.plot(result["years"], d[mid][agg], ls, linewidth=2, color=COL["scenario"], label=METRICS[mid])
    ax.axhline(0, color="#151515", linewidth=0.8)
    _style(ax, "Scenario minus baseline", "people")
    ax.legend(frameon=False, fontsize=8)
    return svg(fig)


def plot_ltc(result):
    fig, ax = plt.subplots(figsize=(7, 3))
    agg = "value" if "value" in result["metrics"]["scenario"]["population_total"] else "mean"
    ax.plot(result["years"], result["metrics"]["comparator"]["ltc_claims_index_65plus"][agg], "--", color=COL["comparator"], label="Baseline")
    ax.plot(result["years"], result["metrics"]["scenario"]["ltc_claims_index_65plus"][agg], color=COL["scenario"], linewidth=2, label="Scenario")
    _style(ax, "65+ LTC claim-pressure index (2025 = 100)", "index")
    ax.legend(frameon=False, fontsize=8)
    return svg(fig)


def findings(result):
    """At most three factual statements, numbers inserted from the result object."""
    agg = "value" if "value" in result["metrics"]["scenario"]["population_total"] else "p50"
    y0, y1 = result["years"][0], result["years"][-1]
    s, d = result["metrics"]["scenario"], result["metrics"]["difference"]
    out = []
    for mid in ("population_total", "population_16_64", "population_65_plus"):
        a, b, dd = s[mid][agg][0], s[mid][agg][-1], d[mid][agg][-1]
        out.append(f"{METRICS[mid]}: {a:,.0f} at end-{y0} to {b:,.0f} at end-{y1} ({dd:+,.0f} versus the baseline).")
    return out


def build(run, result, cache_status):
    sc = Scenario.model_validate(result["scenario"])
    rid = run["id"]
    out = ART / rid
    tmp = ART / f".{rid}.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "plots").mkdir(exist_ok=True)
    (tmp / "validation").mkdir(exist_ok=True)
    pack = load_pack()
    plots = {"population-pyramid.svg": plot_pyramid(result), "trajectories.svg": plot_trajectories(result),
             "scenario-differences.svg": plot_difference(result), "ltc-claim-pressure.svg": plot_ltc(result)}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    files = {
        "scenario.json": json.dumps(result["scenario"], indent=2),
        "result.json": json.dumps(result),
        "metrics.csv": write_csv(metrics_rows(rid, result), ["run_id", "series", "metric_id", "unit", "year", "aggregation", "value"]),
        "age_distribution.csv": write_csv(age_rows(rid, result), ["run_id", "series", "year", "sex_source_category", "age", "aggregation", "value"]),
        "comparisons.csv": write_csv(([rid, "difference", m, UNITS[m], y, a, round(v, 6)] for m, blk in result["metrics"]["difference"].items() for a in blk for y, v in zip(result["years"], blk[a]) if v is not None),
                                     ["run_id", "comparison", "metric_id", "unit", "year", "aggregation", "scenario_minus_comparator"]),
        "assumptions.csv": write_csv(([r["field"], r["value"], r["provenance"]] for r in describe(sc)), ["field", "value", "provenance_class"]),
        "data_dictionary.csv": write_csv([[k, v, UNITS[k]] for k, v in METRICS.items()] + [
            ["series", "scenario | comparator | difference (scenario minus comparator) | official | observed", ""],
            ["aggregation", "value = deterministic expectation; mean/p05..p95 = ensemble summaries computed from per-draw totals; p_below_base = share of simulations below the base-year value (for a difference: share with scenario below comparator)", ""],
            ["sex_source_category", "Sex categories as published by the source; not a model of gender identity", ""]], ["field", "definition", "unit"]),
        "sources.json": json.dumps(pack.meta["sources"], indent=2),
        "validation/run-checks.json": json.dumps({"checks": result["checks"], "data_quality": pack.meta["quality"]["checks"]}, indent=2),
        "plots/chart-specifications.json": json.dumps({"plot_spec_version": "1", "plots": list(plots), "colours": COL}, indent=2),
    } | {f"plots/{k}": v for k, v in plots.items()}
    report = env.get_template("report.html").render(
        run=run, result=result, sc=sc, assumptions=describe(sc), findings=findings(result), plots=plots,
        caveats=CAVEATS, cache_status=cache_status, generated=now, sources=pack.meta["sources"], metrics=METRICS,
        quality=pack.meta["quality"], logo=LOGO)
    files["report.html"] = report
    files["README.md"] = README.format(rid=rid, title=sc.title, generated=now, pack=result["pack_id"], model=result["model_version"])
    manifest = {
        "run_id": rid, "experiment_hash": result["experiment_hash"], "schema_version": sc.schema_version,
        "model_version": result["model_version"], "data_pack_id": result["pack_id"], "code_commit": os.environ.get("JFE_GIT_SHA", "unknown"),
        "image": os.environ.get("JFE_IMAGE", "unknown"), "receipt": result["receipt"], "cache_status": cache_status,
        "queue_seconds": run.get("queue_seconds"), "generated": now, "files": sorted(files),
        "licence_note": "Source data: Open Government Licence - Jersey v1.0; attribution: Statistics Jersey / Government of Jersey.",
    }
    files["manifest.json"] = json.dumps(manifest, indent=2)
    files["checksums.sha256"] = "".join(f"{hashlib.sha256(v.encode()).hexdigest()}  {k}\n" for k, v in sorted(files.items()))
    for k, v in files.items():
        (tmp / k).write_text(v, encoding="utf-8")
    with zipfile.ZipFile(tmp / "experiment.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in sorted(files.items()):
            z.writestr(f"vraic-fe-{rid[-8:]}/{k}", v)
    if out.exists():  # idempotent publication
        return
    tmp.rename(out)


def path_for(run_id, kind):
    if kind not in KINDS:
        raise KeyError(kind)
    return ART / run_id / ("experiment.zip" if kind == "zip" else kind)


README = """# Vraic Futures Engine - experiment {rid}

**{title}**

Generated {generated}. Data pack `{pack}`, model `{model}`.

This package is an *experimental scenario* from a reduced age/sex cohort model. It is a conditional
"what if" calculation, not an official projection or forecast. See `report.html` for the briefing,
assumptions and limitations.

| File | Contents |
|---|---|
| report.html | Self-contained briefing (open offline in any browser) |
| scenario.json | The validated scenario that was executed |
| result.json | Full result contract, including run receipt and checks |
| metrics.csv | Tidy annual metrics for scenario, comparator, difference, official reference and observed history |
| age_distribution.csv | Single-year age by sex for scenario and comparator |
| comparisons.csv | Scenario minus comparator, paired by draw where stochastic |
| assumptions.csv | Assumptions with provenance class (published / derived / assumed / user assumption) |
| data_dictionary.csv | Metric definitions and units |
| sources.json | Every source dataset with resource ID, URL, licence and SHA-256 |
| validation/run-checks.json | Accounting checks for this run and the data-pack quality checks |
| plots/ | SVG charts used in the report |
| checksums.sha256 | SHA-256 of every file (`sha256sum -c checksums.sha256`) |

## Replay

From the project repository (container image built from the same commit):

    podman run --rm -v ./scenario.json:/in/scenario.json:ro,Z localhost/jfe-app:latest \\
        python -m jfe.cli run --scenario /in/scenario.json --output /tmp/out

The replay recomputes the scenario from the archived data pack and compares it with `result.json`.
Deterministic runs reproduce to floating-point tolerance. Process-variation runs reproduce exactly on the
same backend (NumPy vs CuPy use different random streams).
"""
