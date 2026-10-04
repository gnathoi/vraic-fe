"""The one scenario contract shared by chat, form, API, worker, exports and the replay CLI."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

FIRST_YEAR = 2026  # first simulated year; state at end-2025 is the baseline
MAX_END_YEAR = 2080
DEFAULT_END_YEAR = 2070  # product default horizon
OFFICIAL_NET_RANGE = (0, 800)  # published scenario envelope; outside it the profile is extrapolated
NET_LIMITS = (-400, 1500)  # hard envelope: beyond this we refuse rather than extrapolate further
FERT_MULT = (0.5, 1.6)
MORT_MULT = (0.6, 1.4)
Code = Literal["L2", "L1", "MD", "H1", "H2"]
CODE_LABEL = {"L2": "lowest", "L1": "low", "MD": "mid-range", "H1": "high", "H2": "highest"}
PUBLIC_MAX_DRAWS = 16384
DEFAULT_DRAWS = 10000  # product default: every run is a 10,000-draw ensemble unless the user asks otherwise (about 0.25 s on the warm GPU worker)
OPERATOR_MAX_DRAWS = 65536
DEFAULT_MIGRATION = "historical"  # ensembles vary net migration around the assumed path as Jersey's 2001-2025 record did


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class MigrationSegment(Strict):
    from_year: int
    to_year: int
    net_per_year: int = Field(ge=NET_LIMITS[0], le=NET_LIMITS[1])


class Migration(Strict):
    kind: Literal["fixed_annual_net"] = "fixed_annual_net"
    segments: list[MigrationSegment] = Field(min_length=1, max_length=24)


class SchedulePoint(Strict):
    year: int = Field(ge=FIRST_YEAR, le=MAX_END_YEAR)
    value: float


class RateAssumption(Strict):
    """Official assumption code, optionally scaled by a piecewise-linear multiplier.

    Before the first point the first value holds; after the last point the last value holds."""
    assumption: Code = "MD"
    multiplier: list[SchedulePoint] = Field(default_factory=list, max_length=12)


class Uncertainty(Strict):
    mode: Literal["deterministic", "process"] = "deterministic"
    draws: int = Field(default=1, ge=1, le=OPERATOR_MAX_DRAWS)
    # "historical": each simulation's net migration varies around the assumed path like Jersey's record (AR(1)); "fixed": exactly the path
    migration: Literal["fixed", "historical"] = "fixed"
    # "official_range": each simulation draws fertility and mortality levels, treating the official low/high variants as the 5th/95th percentiles
    rates: Literal["fixed", "official_range"] = "fixed"


class Scenario(Strict):
    schema_version: Literal["1.0"] = "1.0"
    title: str = Field(min_length=1, max_length=120)
    parent_scenario_id: str | None = None
    baseline_id: Literal["observed_end2025", "official_state_2025"] = "observed_end2025"
    end_year: int = Field(default=2040, ge=2030, le=MAX_END_YEAR)
    migration: Migration
    fertility: RateAssumption = Field(default_factory=RateAssumption)
    mortality: RateAssumption = Field(default_factory=RateAssumption)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    seed: int = Field(default=20261003, ge=0, le=2**31 - 1)
    solved_for: str | None = Field(default=None, max_length=300)  # set when an assumption was solved for a target

    @model_validator(mode="after")
    def _semantics(self):
        segs = sorted(self.migration.segments, key=lambda s: s.from_year)
        expect = FIRST_YEAR
        for s in segs:
            if s.from_year > s.to_year:
                raise ValueError(f"migration segment {s.from_year}-{s.to_year} ends before it starts")
            if s.from_year != expect:
                kind = "overlap" if s.from_year < expect else "gap"
                raise ValueError(f"migration segments must cover every year from {FIRST_YEAR} to {self.end_year} without gaps or overlaps ({kind} at {s.from_year})")
            expect = s.to_year + 1
        if expect - 1 != self.end_year:
            raise ValueError(f"migration segments end in {expect - 1} but the scenario ends in {self.end_year}")
        self.migration.segments = segs
        for name, ra, (lo, hi) in (("fertility", self.fertility, FERT_MULT), ("mortality", self.mortality, MORT_MULT)):
            years = [p.year for p in ra.multiplier]
            if years != sorted(set(years)):
                raise ValueError(f"{name} multiplier years must be strictly increasing")
            for p in ra.multiplier:
                if not lo <= p.value <= hi:
                    hint = " Jersey's current fertility is about 1.0 child per woman, so ×1.6 is about 1.6." if name == "fertility" else ""
                    raise ValueError(f"{name.capitalize()} can change from ×{lo} to ×{hi} of current rates (asked ×{p.value:g} in {p.year}).{hint}")
                if p.year > self.end_year:
                    raise ValueError(f"{name} multiplier point {p.year} is after the scenario end year {self.end_year}")
        if self.uncertainty.mode == "deterministic" and self.uncertainty.draws != 1:
            raise ValueError("A deterministic run is a single run.")
        if self.uncertainty.mode == "process" and self.uncertainty.draws < 16:
            raise ValueError("Runs use at least 16 simulations, or a single deterministic run.")
        return self

    def schedule(self, ra: RateAssumption):
        """Multiplier per simulated year (list aligned to FIRST_YEAR..end_year)."""
        years = range(FIRST_YEAR, self.end_year + 1)
        if not ra.multiplier:
            return [1.0 for _ in years]
        xs = [p.year for p in ra.multiplier]
        ys = [p.value for p in ra.multiplier]
        out = []
        for y in years:
            if y <= xs[0]:
                out.append(ys[0])
            elif y >= xs[-1]:
                out.append(ys[-1])
            else:
                i = next(i for i in range(len(xs)) if xs[i] >= y)
                w = (y - xs[i - 1]) / (xs[i] - xs[i - 1])
                out.append(ys[i - 1] + w * (ys[i] - ys[i - 1]))
        return out

    def net_by_year(self):
        return [next(s.net_per_year for s in self.migration.segments if s.from_year <= y <= s.to_year) for y in range(FIRST_YEAR, self.end_year + 1)]

    def warnings(self):
        w = []
        lo, hi = OFFICIAL_NET_RANGE
        if any(not lo <= n <= hi for n in self.net_by_year()):
            w.append(f"Net migration outside the published {lo} to +{hi} range: the official age/sex profile is extrapolated linearly.")
        return w


def default_scenario(title="Baseline: official mid-range assumptions", end_year=2040, **kw) -> Scenario:
    return Scenario(title=title, end_year=end_year, migration=Migration(segments=[MigrationSegment(from_year=FIRST_YEAR, to_year=end_year, net_per_year=400)]), **kw)


def product_default(**kw) -> Scenario:
    """The scenario a new conversation starts from: official mid-range, rebased, with the default ensemble."""
    kw.setdefault("end_year", DEFAULT_END_YEAR)
    return default_scenario(uncertainty=Uncertainty(mode="process", draws=DEFAULT_DRAWS, migration=DEFAULT_MIGRATION, rates="official_range"), **kw)


def describe(sc: Scenario) -> list[dict]:
    """Plain-language assumption rows; used for the review diff, the report and assumptions.csv."""
    segs = "; ".join(f"{s.from_year}-{s.to_year}: {s.net_per_year:+d} people/year" for s in sc.migration.segments)

    def rate(ra, what):
        base = f"official {CODE_LABEL[ra.assumption]} ({ra.assumption}) {what} assumption"
        if not ra.multiplier:
            return base
        pts = ", ".join(f"x{p.value:.3g} in {p.year}" for p in ra.multiplier)
        return f"{base}, scaled by {pts} (linear between points, held after the last)"

    return [
        {"field": "baseline", "value": "Observed end-2025 population (Statistics Jersey, Sept 2026, provisional)" if sc.baseline_id == "observed_end2025" else "Reproduce official projection: official projected end-2025 state (Statistics Jersey, Feb 2026, from its end-2024 base)", "provenance": "published"},
        {"field": "horizon", "value": f"{FIRST_YEAR}-{sc.end_year}", "provenance": "default" if sc.end_year == DEFAULT_END_YEAR else "user assumption"},
        {"field": "migration", "value": f"Fixed annual net migration: {segs}", "provenance": "published" if all(s.net_per_year == 400 for s in sc.migration.segments) else "user assumption"},
        {"field": "fertility", "value": rate(sc.fertility, "fertility"), "provenance": "derived" if not sc.fertility.multiplier else "user assumption"},
        {"field": "mortality", "value": rate(sc.mortality, "life-expectancy"), "provenance": "derived" if not sc.mortality.multiplier else "user assumption"},
        {"field": "simulations", "value": "Single deterministic run (expected values)" if sc.uncertainty.mode == "deterministic" else f"{sc.uncertainty.draws:,} simulations (random births and deaths; " + ("net migration varies around the path as in 2001-2025" if sc.uncertainty.migration == "historical" else "migration fixed") + ("; fertility and mortality levels vary, scaled to past forecast errors)" if sc.uncertainty.rates == "official_range" else ")"), "provenance": "default" if (sc.uncertainty.mode == "process" and sc.uncertainty.draws == DEFAULT_DRAWS) else "user assumption"},
    ] + ([{"field": "target", "value": sc.solved_for, "provenance": "solved"}] if sc.solved_for else [])


def diff(parent: Scenario | None, child: Scenario) -> dict:
    """Changed vs unchanged assumption rows against the parent (or the default scenario)."""
    base = parent or default_scenario(end_year=child.end_year)
    a, b = {r["field"]: r for r in describe(base)}, {r["field"]: r for r in describe(child)}
    changed = [{"field": k, "from": a.get(k, {}).get("value", "none"), "to": b[k]["value"]} for k in b if a.get(k, {}).get("value") != b[k]["value"]]
    unchanged = [b[k] for k in b if a.get(k, {}).get("value") == b[k]["value"]]
    return {"changed": changed, "unchanged": unchanged}
