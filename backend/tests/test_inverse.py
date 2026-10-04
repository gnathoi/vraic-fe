import re
import numpy as np
import pytest

from jfe.charts import ChartSpec, build
from jfe.engine import load_pack, run_experiment, simulate
from jfe.inverse import TargetSpec, apply, batch, frontier, solve
from jfe.schemas import RateAssumption, Scenario, default_scenario

pack = load_pack()
par = default_scenario()


@pytest.mark.parametrize("param,value", [("net_migration", 700), ("fertility_multiplier", 1.2), ("mortality_multiplier", 0.9)])
def test_batch_equals_engine(param, value):
    spec = TargetSpec(parameter=param, metric="population_total", year=2040, from_year=2030)
    v, _, ok = batch(pack, par, spec, [value])
    sc = apply(par, spec, value)
    ref = simulate(pack, sc)[1]["population_total"][0][-1]
    assert ok[0] and v[0] == pytest.approx(ref, abs=1e-6)


def test_solved_scenario_achieves_target_in_the_engine():
    spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040)
    r = solve(par, spec, pack)
    assert r["status"] == "solved" and 500 < r["value"] < 700
    sc = apply(par, spec, r["value"], solved_for="test")
    got = run_experiment(sc, pack)["metrics"]["scenario"]["population_16_64"]["value"][-1]
    assert got == pytest.approx(r["target"], abs=40)  # one person per year of migration moves 16-64 by < 40
    assert len(r["sensitivity"]) == 4 and all(s["status"] == "solved" for s in r["sensitivity"])


def test_fertility_solution_and_unreachable_target():
    r = solve(par, TargetSpec(parameter="fertility_multiplier", metric="population_0_15", year=2040), pack, sensitivity=False)
    assert r["status"] == "solved" and 1.2 < r["value"] < 1.6
    u = solve(par, TargetSpec(parameter="net_migration", metric="old_age_ratio", year=2040), pack, sensitivity=False)
    assert u["status"] == "unreachable" and u["closest_value"] == 1500


def test_solved_for_is_descriptive_only():
    from jfe.engine import experiment_hash
    spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040)
    a, b = apply(par, spec, 600), apply(par, spec, 600, solved_for="x")
    assert experiment_hash(a, pack) == experiment_hash(b, pack)


def test_response_and_frontier_charts():
    res = run_experiment(par, pack)
    t = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040).model_dump()
    c = build(ChartSpec(kind="response", target=t), res)
    assert c["columns"] == ["parameter_value", "metric_value", "target"] and c["extra"]["solution"]["status"] == "solved"
    xs = [r[0] for r in c["rows"]]
    assert xs == sorted(xs) and xs[0] == -400 and xs[-1] == 1500
    f = build(ChartSpec(kind="frontier", target=t), res)
    assert len(f["rows"]) == len(f["extra"]["x"]) * len(f["extra"]["y"])
    vals = np.array([r[2] for r in f["rows"]]).reshape(len(f["extra"]["y"]), len(f["extra"]["x"]))
    assert np.all(np.diff(vals, axis=1) > 0)  # more net migration -> more people aged 16-64


def test_target_spec_validation():
    with pytest.raises(ValueError):
        TargetSpec(parameter="net_migration", metric="population_total", year=2040, target_kind="value")
    with pytest.raises(ValueError):
        TargetSpec(parameter="net_migration", metric="population_total", year=2030, from_year=2035)


def test_solve_messages_for_every_target_kind():
    from jfe.api import _solve_text
    for kind, val in (("hold_base", None), ("value", 66000), ("change_pct", -5)):
        spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040, target_kind=kind, target_value=val)
        msg, solved_for = _solve_text(spec, solve(par, spec, pack, sensitivity=False))
        assert msg and solved_for


def test_sensitivity_keeps_parent_multipliers():
    from jfe.schemas import SchedulePoint
    mult = [SchedulePoint(year=2026, value=1.37)]
    spec = TargetSpec(parameter="net_migration", metric="population_0_15", year=2040)
    r = solve(default_scenario(fertility=RateAssumption(multiplier=mult)), spec, pack)
    low = next(x for x in r["sensitivity"] if "(L1)" in x["variant"] and "fertility" in x["variant"])
    explicit = solve(default_scenario(fertility=RateAssumption(assumption="L1", multiplier=mult)), spec, pack, sensitivity=False)
    assert low["value"] == explicit["value"]  # variant switches only the official code; the x1.37 multiplier is kept


def test_homes_match_official_and_solve():
    from jfe.engine import homes_needed
    hq = pack.meta["quality"]["housing"]["additional_2040_by_scenario"]
    for net in (0, 400, 800):
        assert homes_needed(pack, [net] * 15, [2040])[0][0] == pytest.approx(hq[str(net)], abs=0.5)
    r = run_experiment(par, pack)
    h = r["metrics"]["scenario"]["homes_needed_additional"]["value"]
    assert h[0] == 0 and h[-1] == pytest.approx(hq["400"], abs=0.5)
    s = solve(par, TargetSpec(parameter="net_migration", metric="homes_needed_additional", year=2040, target_kind="value", target_value=2000), pack, sensitivity=False)
    assert s["status"] == "solved" and 300 < s["value"] < 400
    c = build(ChartSpec(kind="housing"), r)
    assert len(c["rows"]) == 8 and sum(x[1] for x in c["rows"]) == pytest.approx(hq["400"], abs=1)
    long = run_experiment(default_scenario(end_year=2050), pack)["metrics"]["scenario"]["homes_needed_additional"]["value"]
    assert long[-1] is None and long[15] is not None


def test_solve_from_year_keeps_parent_multiplier_before():
    from jfe.schemas import SchedulePoint
    boosted = default_scenario(fertility=RateAssumption(multiplier=[SchedulePoint(year=2026, value=1.2)]))
    spec = TargetSpec(parameter="fertility_multiplier", metric="population_0_15", year=2040, from_year=2030)
    r = solve(boosted, spec, pack, sensitivity=False)
    sc = apply(boosted, spec, r["value"])
    got = run_experiment(sc, pack)["metrics"]["scenario"]["population_0_15"]["value"][-1]
    assert got == pytest.approx(r["target"], abs=5)  # H1 from the system test: the x1.2 for 2026-2029 must be kept
    assert sc.schedule(sc.fertility)[:4] == pytest.approx([1.2] * 4)


def test_explanation_year_check():
    from jfe.compiler import fill, receipt
    rec = receipt(run_experiment(par, pack))
    ok = [{"text": "At unchanged 2024 rates there are {v0} primary pupils in 2030.", "values": [{"series": "scenario", "metric_id": "pupils_primary_proxy", "year": 2030}]}]
    assert fill(ok, rec)[0]
    bad = [{"text": "In 2033 there are {v0} people aged 80+.", "values": [{"series": "scenario", "metric_id": "population_80_plus", "year": 2035}]}]
    with pytest.raises(ValueError):
        fill(bad, rec)


def test_explanation_direction_words():
    from jfe.compiler import fill, receipt
    rec = receipt(run_experiment(par, pack))
    val = rec["values"]["difference"]["population_total"][2035]
    s = [{"text": "In 2035 the scenario has {v0} fewer residents than the comparator.", "values": [{"series": "difference", "metric_id": "population_total", "year": 2035}]}]
    text = fill(s, rec)[0][0]
    assert "-" not in text and "comparator" not in text and "the baseline" in text
    assert (" fewer " in text) == (val < 0) and f"{abs(val):,.0f}" in text


def test_explanation_comparison_needs_value():
    from jfe.compiler import fill, receipt
    rec = receipt(run_experiment(par, pack))
    with pytest.raises(ValueError):
        fill([{"text": "Children aged 0-15 are lower in 2040 compared with the baseline.", "values": []}], rec)
    out, _ = fill([{"text": "In 2040 there are {v0} people aged 65+.", "values": [{"series": "scenario", "metric_id": "population_65_plus", "year": 2040}]},
                   {"text": "That is {v1} more.", "values": []}], rec)
    assert len(out) == 1


def test_explanation_comparative_follows_sign():
    from jfe.compiler import fill, receipt
    from jfe.schemas import Migration, MigrationSegment
    rec = receipt(run_experiment(default_scenario().model_copy(update={"migration": Migration(segments=[MigrationSegment(from_year=2026, to_year=2040, net_per_year=900)])}), pack))
    d = rec["values"]["difference"]["population_0_15"][2040]
    s = [{"text": "By 2040 the scenario has fewer children aged 0-15 than the baseline, a difference of {v0} people.",
          "values": [{"series": "difference", "metric_id": "population_0_15", "year": 2040}]}]
    text = fill(s, rec)[0][0]
    assert d > 0 and "more children" in text


def test_probabilistic_goal_seek_reaches_its_probability():
    from jfe.inverse import solve_probability
    spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040)
    r = solve_probability(default_scenario(), spec, 0.8, pack=pack, draws=4000)
    plain = solve(default_scenario(), spec, pack, sensitivity=False)
    assert r["status"] == "solved" and r["value"] > plain["value"]  # more migration is needed to be 80% sure than to hit the median
    assert abs(r["achieved_probability"] - 0.8) < 0.04


def test_receipt_with_migration_variability():
    from jfe.compiler import receipt
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=64, migration="historical")}), pack)
    rec = receipt(r)
    assert 0 <= rec["values"]["chance_below_end_2025_pct"]["population_16_64"][2040] <= 100


def test_probabilistic_response_chart():
    from jfe import charts
    spec = TargetSpec(parameter="net_migration", metric="population_16_64", year=2040, probability=0.8)
    data = charts.build(charts.ChartSpec(kind="response", target=spec.model_dump(), run_id="baseline"), run_experiment(par, pack))
    assert "metric_quantile" in data["columns"] and data["extra"]["solution"]["status"] == "solved"
    plain = solve(par, TargetSpec(parameter="net_migration", metric="population_16_64", year=2040), pack, sensitivity=False)
    assert data["extra"]["solution"]["value"] > plain["value"]


def test_probability_series_and_wording():
    from jfe.compiler import fill, receipt
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=200, migration="historical")}), pack)
    rec = receipt(r)
    assert rec["values"]["chance_above_end_2025_pct"]["population_16_64"][2040] == 100 - rec["values"]["chance_below_end_2025_pct"]["population_16_64"][2040]
    assert rec["values"]["p05"]["population_total"][2040] < rec["values"]["p95"]["population_total"][2040]
    ok = [{"text": "There is a {v0}% chance that the population aged 16-64 is above its end-2025 level in 2040.",
           "values": [{"series": "chance_above_end_2025_pct", "metric_id": "population_16_64", "year": 2040}]}]
    assert fill(ok, rec)[0]
    bad = [{"text": "There is a {v0}% chance that the population aged 16-64 falls below its end-2025 level in 2040.",
            "values": [{"series": "chance_above_end_2025_pct", "metric_id": "population_16_64", "year": 2040}]}]
    import pytest as _p
    with _p.raises(ValueError):
        fill(bad, rec)
    share = [{"text": "In 2040, {v0}% of residents are aged 65 or older.", "values": [{"series": "scenario", "metric_id": "share_65_plus_pct", "year": 2040}]}]
    assert fill(share, rec)[0]


def test_fact_answers():
    from jfe.compiler import fact_answer
    r = run_experiment(par, pack)
    assert "Deaths exceed births from" in fact_answer("Will deaths exceed births, and when?", r)[0]
    assert fact_answer("What is natural change in 2035?", r)[0].startswith("Natural change (births minus deaths) in 2035:")
    assert "peaks in" in fact_answer("When does the population peak?", r)[0]
    assert fact_answer("How many pupils in 2035?", r) is None


def test_chance_fact_answer():
    from jfe.compiler import fact_answer
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=200, migration="historical")}), pack)
    out = fact_answer("What is the chance the working-age population is below today's level in 2040?", r)[0]
    assert "chance that the population aged 16-64 is below its end-2025 level" in out and "2040" in out


def test_attendee_questions():
    from jfe.compiler import fact_answer, strip_preamble
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=400, migration="historical")}), pack)
    assert strip_preamble("I'm a school planner. How many primary pupils in 2035?") == "How many primary pupils in 2035?"
    assert fact_answer("Is this a forecast? Can I trust these numbers?", r)[0].startswith("No, it is not a forecast")
    med = r["metrics"]["scenario"]["population_total"]["p50"][r["years"].index(2040)]
    out = fact_answer(f"What's the chance we get to {med:,.0f} people in 2040?", r)[0]
    assert "chance that the total population is at or above" in out and re.search(r"About (4[5-9]|5[0-5])%", out)
    assert "chance that the population aged 16-64 is below" in fact_answer("Will there be fewer working-age people to hire in 2035?", r)[0]


def test_worst_case_and_default_measure():
    from jfe.compiler import fact_answer
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=200, migration="historical")}), pack)
    assert fact_answer("What's the worst case for that?", r)[0].startswith("For the total population in 2040: worst case (5th percentile)")
    assert "the total population" in fact_answer("what's the chance we actually get to 110,000 in 2040 with that?", r)[0]
    assert "worst case (95th percentile)" in fact_answer("Worst case for the old-age ratio in 2040?", r)[0]


def test_journalist_questions():
    from jfe.compiler import fact_answer
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=200, migration="historical")}), pack)
    sup = fact_answer("how many working age people per pensioner now and in 2040", r)[0]
    assert sup.startswith("People aged 16-64 per person aged 65+: 3.2 in 2025") and "in 2040" in sup
    assert "not an official" in fact_answer("who made this? is it the government", r)[0]
    assert "cannot make" in fact_answer("is the island full?", r)[0]
    out = fact_answer("how many fewer primary kids by 2035", r)[0]
    assert out.startswith("Primary pupils in 2035:") and "fewer than at end-2025" in out
    assert "96%" in fact_answer("has this model been wrong before?", r)[0]


def test_health_planner_questions():
    from jfe.compiler import fact_answer
    from jfe.schemas import Uncertainty
    r = run_experiment(default_scenario().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=200, migration="historical")}), pack)
    assert "+400" in fact_answer("What net migration does the baseline assume?", r)[0]
    multi = fact_answer("Give me a 2040 table of working-age, 65+, bed days and nurses", r)[0]
    assert multi.startswith("In 2040:") and "hospital bed days" in multi and "hospital nurses" in multi
    assert fact_answer("What happens to hospital bed days by 2040?", r)[0].startswith("Hospital bed days")


def test_assumption_rule_not_triggered_by_if_questions():
    from jfe.compiler import fact_answer
    r = run_experiment(par, pack)
    assert fact_answer("How many primary pupils should we plan for in 2035 if fertility follows the official low assumption?", r) is None or \
        not fact_answer("How many primary pupils should we plan for in 2035 if fertility follows the official low assumption?", r)[0].startswith("This run assumes")
    assert fact_answer("What net migration does the baseline assume?", r)[0].startswith("This run assumes")
