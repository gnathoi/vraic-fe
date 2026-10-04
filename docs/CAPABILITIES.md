# Capability inventory (Vraic Futures Engine)

Status as of 3 October 2026. **Implemented** = in the code and deployed; **Tested** = covered by an automated check; **Incomplete** = partly built; **Proposed** = not built. Figures are measured on
a 32-core host with an NVIDIA RTX 6000 Ada 48 GB unless stated.

## Statistical model

| Item | Status | Detail |
|---|---|---|
| Model type | Implemented, tested | Age/sex cohort-component projection (10,000-draw ensemble by default, deterministic on request), single years 0-99 and 100+, annual steps 2026 to 2030-2080 |
| Event order | Implemented, tested | Ageing → deaths → births → net migration (Statistics Jersey methodology, S04) |
| Baselines | Implemented, tested | Observed end-2025 (S06, 3 suppressed cells imputed and labelled) or official projected end-2025 state (reproduction mode) |
| Mortality | Implemented, tested | Annual death probabilities derived from official projection outputs for each life-expectancy code L2-H2; optional multiplier schedule (0.6-1.4) |
| Fertility | Implemented, tested | Published 2023-25 Jersey ASFRs (S34) × yearly level calibrated to official births for each fertility code L2-H2; optional multiplier schedule (0.5-1.6) |
| Migration | Implemented, tested | Fixed annual net migration with the official age/sex profile, recovered exactly as M = M_nil + net × P; −400 to +1,500 per year (published range 0 to +800; outside it is flagged as extrapolation) |
| Uncertainty | Implemented, tested | Default: 10,000 simulations (binomial deaths, Poisson births, binomial birth sex; net migration varies around the path as an AR(1) fitted to 2001-2025; fertility and mortality levels vary, scaled to past forecast errors (1.5x and 4x the official low-high spread); homes vary with each draw's migration); changeable in conversation or the form (16-16,384; operator up to 65,536) or a single deterministic run on request; official reproduction is always deterministic. Pointwise percentiles from per-draw totals; scenario and comparator share random numbers so differences are paired |
| Not modelled | — | Gross migration flows, residential status, households, labour participation, policy or housing feedbacks, parameter and structural uncertainty |
| Data gate outcome | — | B: reduced model with documented derived parameters |

## Derived indicators

| Indicator | Status | Method |
|---|---|---|
| Population totals, 0-15, 16-64, 65+, 80+ | Implemented, tested | Sums over cells |
| Old-age and dependency ratios | Implemented, tested | Per 100 aged 16-64 |
| 65+ LTC claim-pressure index | Implemented, tested | 2024 open claims per person by band and sex (S08/S06), held constant |
| School-age populations, pupils | Implemented, tested | Year-group weights by month of birth; pupils at January-2024 participation (S36). Check: 2012 ratio predicts 2024 pupils within +3.4% (primary) and −4.2% (secondary) |
| Hospital bed days, GP appointments | Implemented, tested | PHI per-person rates by age and sex (S37). Reproduce PHI 2023 totals within 0.3% and 0.8%; bed-day growth 2025-2043 29.8% vs PHI 29.5% (2023-2043, +325 net) |
| Hospital nurse and doctor FTE | Implemented, tested | Health and Care Jersey FTE at 31 Dec 2025 (S38: 1,112 nurses/midwives, 272 doctors) held constant per age-weighted bed day. Illustrative |
| Homes needed by dwelling type | Implemented, tested | Statistics Jersey housing-needs projections decomposed exactly as H = H_nil + net × H_per_person (S39); change since end-2025, to 2040; exact for constant net migration, average net migration to date for time-varying paths; inverse target |
| School capacity, GP/teacher/carer staffing, beds, costs, housing supply | Proposed | Need capacity and service-model data; the assistant states this |

## Inverse simulation

| Item | Status | Detail |
|---|---|---|
| Solve one assumption for a target | Implemented, tested | Free parameter: net migration (−400 to +1,500/yr), fertility multiplier (0.5-1.6) or death-rate multiplier (0.6-1.4), from a chosen year; target: hold the end-2025 level, an absolute value or a % change, for any core metric and year. The whole range is sampled in one vectorised batch, monotonicity is checked, the target is bracketed and refined |
| Outcomes | Implemented, tested | Solved (value, achieved metric, sensitivity under official low/high variants); unreachable (closest achievable in range); not monotone (refused). Never a recommendation |
| Integration | Implemented, tested | Chat intent `solve_target`; message numbers computed by the server; the solved assumption becomes a normal draft (review card, confirm, run) with a `solved` provenance row; confirmed runs reach the target (acceptance test) |
| Charts | Implemented, tested | Response curve (metric vs parameter, target, solution) and frontier (net migration × fertility or death-rate multiplier, 3,465 scenarios, contour at the target); same chart-data/CSV path |
| Examples (mid-range, end-2040) | Measured | Hold ages 16-64 at 68,073: net migration +593/yr (variants +580 to +606). Hold the old-age ratio at 31.1: not reachable (best 36.4 at +1,500). Hold ages 0-15 at 15,217 at +400: fertility ×1.37 |
| Runtime | Measured | Solve incl. sensitivity ≈ 0.5 s; frontier 3,465 scenarios 0.09 s (CPU, vectorised) |

## Validation

| Check | Result |
|---|---|
| Reproduce official projections from the official end-2025 state | Mid-range max 1.6 people 2026-2060; all 125 published combinations max 22 people (emulation, not independent validation) |
| Retrospective 2017→2025 with observed births and net migration | End-2025 error +0.42% total, +0.32% aged 16-64, −0.36% aged 65+; residual mostly under-predicted deaths |
| Fertility pattern vs official births | Held-out births error ≤ 0.71/yr; implied 2034 TFR 1.026 vs published 1.01 |
| Numerical invariants | 19 unit tests: accounting, ageing, open age group, cohort timing, identity scenario, paired differences, chart tables = result values = CSV |

## Natural-language interface

| Item | Status | Detail |
|---|---|---|
| Local LLM | Implemented | Qwen3-30B-A3B-Instruct-2507-FP8 on vLLM 0.30, JSON-schema-constrained output, temperature 0 |
| Scenario drafting and follow-ups | Implemented, tested | Patch against an explicit parent; strict validation; deterministic diff; human confirmation before any run |
| Clarification, unsupported, normative, injection handling | Implemented, tested | Held-out set: 50/51 incl. inverse 6/6 and housing 3/4 (first run 39/41 on the original 41); p50 1.1 s |
| Conversational charts and tables over stored results | Implemented, tested | Trajectory, fan, difference, pyramid (≤2 years, common scale), ranking, reconciliation; table view; no re-simulation |
| Conversational downloads | Implemented, tested | Chart data (exact plotted table, CSV with provenance header), run data, assumptions, manifest, report, ZIP |
| Grounded explanations | Implemented | Numbers inserted by the server from the result; template fallback |
| Acceptance conversation (judge queries) | Tested | 30/30 checks, incl. chart = result = CSV consistency, cross-session isolation, inverse solve → confirmed run reaching the target, housing |
| System tests | Tested | Round 1: 134 prompts, 3 crowd runs (findings fixed and re-verified); round 2 |

## Exports

| Output | Status |
|---|---|
| Briefing HTML (self-contained, SVG plots, assumptions, sources, limitations, receipt) | Implemented |
| metrics.csv, age_distribution.csv, comparisons.csv, assumptions.csv, data_dictionary.csv | Implemented |
| scenario.json, result.json, manifest.json, checksums.sha256, experiment ZIP | Implemented |
| Chart data CSV (per chart) | Implemented, tested |
| Chart SVG/PNG | Implemented in the browser (ECharts) |
| Replay CLI (`python -m jfe.cli run --compare`) | Implemented |
| PDF briefing | Proposed |

## Platform and runtime (measured)

| Item | Result |
|---|---|
| Containers | Podman quadlets: jfe-db (PostgreSQL 17), jfe-llm (vLLM), jfe-api (FastAPI + static UI), jfe-worker (NumPy/CuPy), jfe-reports |
| Compute libraries | NumPy 2.5 (reference), CuPy 14.2 (CUDA 13); float64 |
| Simulation runtime (warm, scenario + comparator) | Deterministic 0.002 s (CPU). 256 draws: CPU 0.098 s, GPU 0.036 s. 4,096: 1.50 s / 0.045 s. 65,536: 26.3 s / 0.35 s |
| LLM latency | Single prompt p50 1.17 s; 10 concurrent p95 4.3 s |
| Room load | 50 browsing sessions p95 0.56 s; 20-run burst 20/20, p95 submit→complete 1.4 s |
| Audience sandbox | 13/13: presentation hold, queue cap 429, operator priority (0.21 s with a full queue), queue positions, chart requests 0.002 s during a full queue, reconnect, cancel, idempotency, one run per session, isolation |
| Access | Two codes: access code (sessions) and admin code (priority, modes, /admin) |
| Frontend | Implemented: PatternFly 6.6 + Chatbot 6.7, ECharts 6; appearance switcher (Default / Red Hat Project Felt × light / dark / system × default / glass / high contrast); per-chart Table / CSV / SVG; presentation mode. Verified on the live stack in Default light/dark, Felt light, Felt dark glass |
