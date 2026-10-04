import copy

import numpy as np
import pytest
from pydantic import ValidationError

from jfe.engine import load_pack, run_experiment, simulate
from jfe.schemas import Migration, MigrationSegment, RateAssumption, SchedulePoint, Scenario, Uncertainty, default_scenario, diff

pack = load_pack()


def sc(**kw):
    return default_scenario(title="t", **kw)


def two_stage(a=250, b=400, end=2040):
    return Scenario(title="two-stage", end_year=end, migration=Migration(segments=[
        MigrationSegment(from_year=2026, to_year=2030, net_per_year=a), MigrationSegment(from_year=2031, to_year=end, net_per_year=b)]))


def test_reference_emulation_matches_official_cube():
    """Started from the official end-2025 state with official assumptions, the engine reproduces the
    published +400 projection. Mortality and migration are exact by construction; births come from the
    published ASFR shape, so small differences are expected. This is an emulator check, not validation."""
    years, per, age, chk = simulate(pack, sc(baseline_id="official_state_2025"))
    off = pack.off_N[2, 2, 2, : len(years)].sum((-1, -2))
    assert np.abs(per["population_total"][0] - off).max() < 10  # people, out of ~105k
    assert chk["max_accounting_error_people"] < 1e-6


def test_no_event_fixture_conserves_and_ages():
    p = copy.copy(pack)
    p.q, p.fert_intensity, p.M0, p.P = pack.q * 0, pack.fert_intensity * 0, pack.M0 * 0, pack.P * 0
    years, per, age, chk = simulate(p, sc(end_year=2030))
    tot = per["population_total"][0]
    assert np.allclose(tot, tot[0])
    # a cohort aged 10 at end-2025 is aged 15 at end-2030
    assert np.allclose(age[5][:, 15], pack.base_2025[:, 10])
    # the open-ended 100+ group keeps its members and gains the 99s
    assert np.allclose(age[1][:, 100], pack.base_2025[:, 100] + pack.base_2025[:, 99])


def test_fertility_cannot_change_working_age_before_2042():
    """Multiplier is 1.0 in 2026, so the first extra births are in 2027; they turn 16 at end-2043."""
    hi = sc(end_year=2045, fertility=RateAssumption(multiplier=[SchedulePoint(year=2026, value=1.0), SchedulePoint(year=2034, value=1.1)]))
    r = run_experiment(hi)
    d = np.array(r["metrics"]["difference"]["population_16_64"]["value"])
    years = r["years"]
    assert np.all(d[: years.index(2042) + 1] == 0)
    assert d[years.index(2043)] > 0
    assert r["metrics"]["difference"]["births"]["value"][years.index(2034)] > 0


def test_identity_scenario_has_zero_difference():
    r = run_experiment(sc())
    for m, v in r["metrics"]["difference"].items():
        assert np.allclose([x for x in v["value"] if x is not None], 0), m


def test_migration_path_is_respected():
    r = run_experiment(two_stage())
    net = r["metrics"]["scenario"]["net_migration"]["value"]
    assert np.allclose(net[1:6], 250) and np.allclose(net[6:], 400)
    assert r["metrics"]["difference"]["population_total"]["value"][-1] < 0


def test_process_variation_is_reproducible_paired_and_bounded():
    s = sc(uncertainty=Uncertainty(mode="process", draws=64), fertility=RateAssumption(assumption="H1"))
    r1, r2 = run_experiment(s), run_experiment(s)
    assert r1["metrics"]["scenario"] == r2["metrics"]["scenario"]
    det = run_experiment(sc(fertility=RateAssumption(assumption="H1")))
    m, d = np.array(r1["metrics"]["scenario"]["population_total"]["mean"]), np.array(det["metrics"]["scenario"]["population_total"]["value"])
    assert np.abs(m / d - 1).max() < 0.005
    lo, hi = r1["metrics"]["scenario"]["population_total"]["p05"], r1["metrics"]["scenario"]["population_total"]["p95"]
    assert all(a <= b for a, b in zip(lo, hi)) and hi[-1] > lo[-1]
    # common random numbers: paired difference spread is far narrower than the level spread
    dl, dh = r1["metrics"]["difference"]["population_total"]["p05"][-1], r1["metrics"]["difference"]["population_total"]["p95"][-1]
    assert (dh - dl) < (hi[-1] - lo[-1])
    assert r1["checks"]["scenario"]["max_accounting_error_people"] < 1e-6
    assert min(r1["metrics"]["scenario"]["population_total"]["p05"]) > 0


def test_semantic_validation_rejects_bad_scenarios():
    with pytest.raises(ValidationError, match="gap"):
        Scenario(title="x", migration=Migration(segments=[MigrationSegment(from_year=2026, to_year=2029, net_per_year=1), MigrationSegment(from_year=2031, to_year=2040, net_per_year=1)]))
    with pytest.raises(ValidationError, match="overlap"):
        Scenario(title="x", migration=Migration(segments=[MigrationSegment(from_year=2026, to_year=2031, net_per_year=1), MigrationSegment(from_year=2031, to_year=2040, net_per_year=1)]))
    with pytest.raises(ValidationError, match="can change from"):
        sc(fertility=RateAssumption(multiplier=[SchedulePoint(year=2030, value=3.0)]))
    with pytest.raises(ValidationError):
        Scenario.model_validate({"title": "x", "migration": {"segments": [{"from_year": 2026, "to_year": 2040, "net_per_year": 400}]}, "run_code": "rm -rf /"})
    with pytest.raises(ValidationError):
        sc(uncertainty=Uncertainty(mode="deterministic", draws=5))


def test_extrapolation_warning_and_diff():
    s = two_stage(1200, 1200)
    assert any("extrapolated" in w for w in s.warnings())
    d = diff(None, two_stage())
    assert [c["field"] for c in d["changed"]] == ["migration"]
    assert {u["field"] for u in d["unchanged"]} >= {"fertility", "mortality", "baseline"}


def test_ltc_index_is_100_in_base_year_and_grows_with_ageing():
    r = run_experiment(sc())
    idx = r["metrics"]["scenario"]["ltc_claims_index_65plus"]["value"]
    assert idx[0] == pytest.approx(100)
    assert idx[-1] > 120


def test_negative_migration_is_feasible_or_refused_clearly():
    s = two_stage(-400, -400, end=2060)
    try:
        r = run_experiment(s)
        assert min(r["metrics"]["scenario"]["population_total"]["value"]) > 0
    except ValueError as e:
        assert "Infeasible" in str(e)


def test_reproduces_every_published_combination():
    """Started from the official projected end-2025 state, every published net x fertility x life-expectancy
    combination matches Statistics Jersey's numbers closely (emulation check, not independent validation)."""
    from jfe.schemas import RateAssumption as RA
    worst = 0.0
    for net in (0, 200, 400, 600, 800):
        for f in ("L2", "MD", "H2"):
            for m in ("L2", "MD", "H2"):
                s = sc(baseline_id="official_state_2025", end_year=2060, fertility=RA(assumption=f), mortality=RA(assumption=m))
                s = s.model_copy(update={"migration": Migration(segments=[MigrationSegment(from_year=2026, to_year=2060, net_per_year=net)])})
                r = run_experiment(Scenario.model_validate(s.model_dump()))
                assert r["official"]["matches_scenario_assumptions"]
                worst = max(worst, r["reproduction"]["max_abs_difference_people"]["population_total"])
    assert worst < 25, worst


def test_school_metrics_respond_to_fertility_only_after_school_entry():
    from jfe.schemas import RateAssumption as RA
    r = run_experiment(sc(fertility=RA(multiplier=[SchedulePoint(year=2026, value=0.8)])))
    d = r["metrics"]["difference"]["pupils_primary_proxy"]["value"]
    years = r["years"]
    # births from 2026 enter Reception (age 4 in September) from the school year starting 2030
    assert all(abs(x) < 1e-9 for x in d[: years.index(2029) + 1])
    assert d[years.index(2035)] < 0
    assert all(abs(x) < 1e-9 for x in r["metrics"]["difference"]["pupils_secondary_proxy"]["value"][: years.index(2036) + 1])
    base = r["metrics"]["comparator"]["pupils_primary_proxy"]["value"][0]
    assert 6000 < base < 9000  # ~7,400 primary pupils in January 2024
    # an official fertility code also shifts the official migration profile slightly (emigration depends on population)
    code = run_experiment(sc(fertility=RA(assumption="L1")))["metrics"]["difference"]["pupils_primary_proxy"]["value"]
    assert max(abs(x) for x in code[: years.index(2029) + 1]) < 5


def test_health_proxies_anchor_to_2025_baseline_and_rise_with_ageing():
    r = run_experiment(sc())
    n = r["metrics"]["scenario"]["nurses_fte_proxy"]["value"]
    assert n[0] == pytest.approx(1112) and n[-1] > 1112 * 1.1
    assert r["metrics"]["scenario"]["doctors_fte_proxy"]["value"][0] == pytest.approx(272)
    assert "hospital_bed_days" not in r["observed"]["metrics"]


def test_official_baseline_requires_explicit_reproduce_request():
    from jfe.compiler import EXPLICIT_REPRODUCE
    assert not EXPLICIT_REPRODUCE.search("Use the official high life expectancy assumption with zero net migration from 2031.")
    assert EXPLICIT_REPRODUCE.search("Reproduce the official +600 projection")


def test_summarise_matches_per_metric_percentiles():
    from jfe.engine import QUANTILES, summarise
    rng = np.random.default_rng(1)
    per = {"a": rng.normal(size=(500, 7)), "b": rng.poisson(30, size=(500, 7)).astype(float)}
    out = summarise(per, True)
    for k, v in per.items():
        assert out[k]["mean"] == pytest.approx(v.mean(0).tolist())
        for q, p in QUANTILES.items():
            assert out[k][q] == pytest.approx(np.percentile(v, p, axis=0).tolist())


def test_systematic_rounding_is_unbiased_and_keeps_the_total():
    from jfe.engine import round_systematic
    v = np.array([[0.3, 2.7, -1.4, 0.05], [5.5, 0.0, 0.95, 1.2]])
    out = round_systematic(np, v, np.random.default_rng(3), 20000)
    assert np.all(out == np.round(out))
    assert np.allclose(out.mean(0), v, atol=0.02)
    assert np.all(np.abs(out.sum((1, 2)) - v.sum()) < 1)


def test_migration_variability_keeps_the_median_and_pairs_the_baseline():
    from jfe.engine import migration_deviations
    e = migration_deviations(np, 7, 45, 20000)
    assert abs(e.mean()) < 15 and 250 < e[:, -1].std() < 500  # mean zero, stationary spread of the fitted AR(1)
    unc = Uncertainty(mode="process", draws=2000, migration="historical")
    r = run_experiment(sc().model_copy(update={"uncertainty": unc}))
    d = run_experiment(sc())
    tot, det = r["metrics"]["scenario"]["population_total"], d["metrics"]["scenario"]["population_total"]["value"]
    assert abs(tot["p50"][-1] - det[-1]) < 0.01 * det[-1] and tot["p95"][-1] - tot["p05"][-1] > 2000  # wide band, same centre
    diff = r["metrics"]["difference"]["population_total"]
    assert diff["p05"][-1] == diff["p95"][-1] == 0  # identical assumptions: same random paths, zero difference
    assert 0 <= tot["p_below_base"][-1] <= 1


def test_exports_handle_every_aggregation():
    from jfe.exports import metrics_rows
    r = run_experiment(sc().model_copy(update={"uncertainty": Uncertainty(mode="process", draws=64, migration="historical")}))
    r["official"], r["observed"] = r.get("official", {"years": [], "metrics": {}}), r.get("observed", {"years": [], "metrics": {}})
    rows = list(metrics_rows("r_test", r))
    assert any(x[5] == "p_below_base" for x in rows) and any(x[2] == "homes_needed_additional" for x in rows)


def test_migration_deviations_do_not_depend_on_the_horizon():
    from jfe.engine import migration_deviations
    short, long = migration_deviations(np, 11, 15, 500), migration_deviations(np, 11, 45, 500)
    assert np.allclose(short, long[:, :15])


def test_rate_uncertainty_spans_the_official_variants_and_stays_paired():
    unc = Uncertainty(mode="process", draws=2000, migration="historical", rates="official_range")
    r = run_experiment(sc().model_copy(update={"uncertainty": unc}))
    i = r["years"].index(2035)
    pup = r["metrics"]["scenario"]["pupils_primary_proxy"]
    lo = run_experiment(sc(fertility=RateAssumption(assumption="L1")))["metrics"]["scenario"]["pupils_primary_proxy"]["value"][i]
    hi = run_experiment(sc(fertility=RateAssumption(assumption="H1")))["metrics"]["scenario"]["pupils_primary_proxy"]["value"][i]
    assert pup["p05"][i] < lo and pup["p95"][i] > hi  # the band covers the official low-high fertility spread
    assert r["metrics"]["difference"]["population_total"]["p95"][-1] == 0  # same draws for scenario and comparator
    homes = r["metrics"]["scenario"]["homes_needed_additional"]
    assert homes["p05"][-1] < homes["p50"][-1] < homes["p95"][-1]
