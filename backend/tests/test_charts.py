import csv
import io

import pytest

from jfe.charts import ChartError, ChartSpec, build, to_csv
from jfe.engine import run_experiment
from jfe.schemas import Migration, MigrationSegment, Scenario, Uncertainty, default_scenario

det = run_experiment(Scenario(title="t", migration=Migration(segments=[MigrationSegment(from_year=2026, to_year=2030, net_per_year=250), MigrationSegment(from_year=2031, to_year=2040, net_per_year=0)])))
proc = run_experiment(default_scenario(uncertainty=Uncertainty(mode="process", draws=64)))


def test_trajectory_rows_equal_result_values():
    c = build(ChartSpec(kind="trajectory", metrics=["population_16_64"], series=["scenario", "comparator"]), det)
    col = c["columns"].index("population_16_64:scenario")
    got = {r[0]: r[col] for r in c["rows"]}
    for y, v in zip(det["years"], det["metrics"]["scenario"]["population_16_64"]["value"]):
        assert got[y] == pytest.approx(v, abs=1e-3)


def test_csv_is_the_same_table():
    c = build(ChartSpec(kind="difference", metrics=["population_total", "population_16_64"]), det)
    lines = [ln for ln in to_csv(c).splitlines() if not ln.startswith("#")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    assert rows[0] == c["columns"]
    flat = [float(x) for r in rows[1:] for x in r]
    assert flat == pytest.approx([float(x) for r in c["rows"] for x in r])
    assert "experiment_hash" in to_csv(c)


def test_fan_requires_process_run_and_orders_quantiles():
    with pytest.raises(ChartError):
        build(ChartSpec(kind="fan", metrics=["population_total"]), det)
    c = build(ChartSpec(kind="fan", metrics=["population_total"]), proc)
    for r in c["rows"]:
        assert r[1] <= r[2] <= r[3] <= r[4] <= r[5]


def test_pyramid_years_and_totals():
    c = build(ChartSpec(kind="pyramid", years=[2025, 2040]), det)
    tot40 = sum(r[3] for r in c["rows"] if r[2] == 2040)
    assert tot40 == pytest.approx(det["metrics"]["scenario"]["population_total"]["value"][-1], rel=1e-6)
    with pytest.raises(ChartError):
        build(ChartSpec(kind="pyramid", years=[2070]), det)


def test_ranking_sorted_by_absolute_change():
    c = build(ChartSpec(kind="ranking", basis="vs_start"), det)
    k = c["columns"].index("change_vs_start")
    vals = [abs(r[k]) for r in c["rows"]]
    assert vals == sorted(vals, reverse=True)


def test_mixed_units_refused():
    with pytest.raises(ChartError):
        build(ChartSpec(kind="trajectory", metrics=["population_total", "old_age_ratio"]), det)


def test_cached_result_overlay_keeps_own_scenario():
    other = det | {"scenario": det["scenario"] | {"title": "SECRET"}}
    mine = other | {"scenario": det["scenario"] | {"title": "bravo"}}
    c = build(ChartSpec(kind="trajectory", metrics=["population_total"]), mine)
    assert "SECRET" not in str(c)
