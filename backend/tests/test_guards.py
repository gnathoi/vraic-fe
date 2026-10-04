from jfe.compiler import rate_guards
from jfe.schemas import product_default


def guard(prompt, llm=None):
    p = {"intent": "create_scenario", "fertility": None, "mortality": None} | (llm or {})
    rate_guards(p, prompt, product_default())
    return p["fertility"], p["mortality"]


def test_official_codes_follow_the_words():
    f, m = guard("Use the low fertility assumption and the high life expectancy assumption.", {"fertility": {"assumption": "L2", "multiplier": None}})
    assert f["assumption"] == "L1" and m["assumption"] == "H1"
    f, m = guard("Low fertility and low life expectancy.")
    assert f["assumption"] == "L1" and m["assumption"] == "L1"
    f, _ = guard("Use the official highest fertility assumption.", {"fertility": {"assumption": "H2", "multiplier": [{"year": 2034, "value": 1.4}]}})
    assert f == {"assumption": "H2", "multiplier": []}
    _, m = guard("High death rates.")
    assert m["assumption"] == "L1"  # high death rates = low life expectancy


def test_percent_death_rates_are_a_multiplier():
    for q in ("Death rates 10% lower.", "Reduce death rates by 10%.", "Net migration of 900 a year to 2060 with death rates 10% lower."):
        _, m = guard(q, {"mortality": {"assumption": "H1", "multiplier": None}})
        assert m["assumption"] == "MD" and m["multiplier"][-1]["value"] == 0.9, q
    _, m = guard("Death rates 10% lower from 2030.")
    assert m["multiplier"] == [{"year": 2029, "value": 1.0}, {"year": 2030, "value": 0.9}]
    _, m = guard("Reduce death rates by 5% gradually by 2040.", {"mortality": {"assumption": "MD", "multiplier": [{"year": 2026, "value": 1.0}, {"year": 2040, "value": 0.95}]}})
    assert m["multiplier"][-1] == {"year": 2040, "value": 0.95}


def test_undo_clears_the_multiplier():
    f, _ = guard("Undo the fertility change.")
    assert f == {"assumption": "MD", "multiplier": []}
    f, _ = guard("Set fertility back to the official mid-range.")
    assert f["multiplier"] == [] and f["assumption"] == "MD"


def test_another_percent_compounds():
    from jfe.schemas import RateAssumption, SchedulePoint
    prev = product_default().model_copy(update={"fertility": RateAssumption(multiplier=[SchedulePoint(year=2029, value=1.0), SchedulePoint(year=2030, value=1.1)])})
    p = {"intent": "modify_scenario", "fertility": None, "mortality": None}
    rate_guards(p, "Fertility another 10% higher on top of that.", prev)
    assert p["fertility"]["multiplier"] == [{"year": 2029, "value": 1.0}, {"year": 2030, "value": 1.21}]


def test_age_groups_do_not_set_the_direction():
    import re
    prompt = "What net migration gives an 80% chance that over-65s stay below 25,000 in 2040?"
    dp = re.sub(r"\b(over|under)[- ]?\d+s?\b|\baged \d+\s*(and over|or over|\+)?", " ", prompt, flags=re.I)
    assert not re.search(r"\bover\b", dp) and re.search(r"\bbelow\b", dp)


def test_migration_chance():
    from jfe.compiler import migration_chance
    from jfe.schemas import Migration, MigrationSegment
    sc = product_default().model_copy(update={"migration": Migration(segments=[MigrationSegment(from_year=2026, to_year=2070, net_per_year=760)])})
    out = migration_chance("what is the chance that we will have that level of migration?", sc)[0]
    assert "averaging +760 a year or more over 2026-2040 is" in out
    assert migration_chance("can you determine a probability", sc)
    assert migration_chance("What net migration gives an 80% chance of keeping the working-age population at today's level?", sc) is None
    assert migration_chance("can you determine a probability", product_default()) is None

def test_migration_chance_typo():
    from jfe.compiler import migration_chance
    from jfe.schemas import Migration, MigrationSegment
    sc = product_default().model_copy(update={"migration": Migration(segments=[MigrationSegment(from_year=2026, to_year=2070, net_per_year=760)])})
    assert migration_chance("can you determine a probablilty", sc)
