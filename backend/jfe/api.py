"""HTTP API. One validated contract for chat, form and scripts; the simulation never runs inside a request."""
import asyncio
import hmac
import ipaddress
import re
import json
import os
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi.concurrency import run_in_threadpool
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import charts, compiler, db, evidence, exports, inverse
from .engine import METRICS, MODEL_VERSION, UNITS, experiment_hash, load_pack, run_experiment
from .schemas import (CODE_LABEL, FERT_MULT, FIRST_YEAR, MAX_END_YEAR, MORT_MULT, NET_LIMITS, OFFICIAL_NET_RANGE, OPERATOR_MAX_DRAWS,
                      PUBLIC_MAX_DRAWS, DEFAULT_DRAWS, Scenario, default_scenario, describe, diff, product_default)

EVENT_CODE = os.environ.get("JFE_EVENT_CODE", "")  # unset: only the admin code opens a session
OPERATOR_CODE = os.environ.get("JFE_OPERATOR_CODE", "")
SECURE_COOKIE = os.environ.get("JFE_SECURE_COOKIE", "1") == "1"
RETENTION = "48 hours"
MAX_WAITING = int(os.environ.get("JFE_MAX_WAITING", "60"))
LLM_ACTIVE, LLM_WAITING = int(os.environ.get("JFE_LLM_ACTIVE", "16")), int(os.environ.get("JFE_LLM_WAITING", "40"))
STATIC = Path(os.environ.get("JFE_STATIC", "frontend/dist"))
EXAMPLES = [  # the welcome screen shows a random few, one per area; each works as an opening question
    {"area": "Population", "text": "Use annual net migration of 250 from 2026 to 2030 and 400 from 2031 to 2040. Keep the other assumptions unchanged."},
    {"area": "Population", "text": "What if net migration were zero every year to 2040?"},
    {"area": "Population", "text": "What is the chance the total population is below today's level in 2050?"},
    {"area": "Population", "text": "What's the worst case for total population in 2050?"},
    {"area": "Ageing and care", "text": "How many working-age people per person aged 65+ now and in 2050?"},
    {"area": "Ageing and care", "text": "Show the 65+ long-term-care claim-pressure index if life expectancy follows the official high assumption."},
    {"area": "Ageing and care", "text": "Will deaths exceed births, and when?"},
    {"area": "Health", "text": "How many hospital nurses would 2040 need at today's staffing per bed day?"},
    {"area": "Health", "text": "What happens to hospital bed days by 2040?"},
    {"area": "Health", "text": "What net migration would keep hospital bed days below 80,000 in 2040?"},
    {"area": "Education", "text": "How many primary pupils should we plan for in 2035 if fertility follows the official low assumption?"},
    {"area": "Education", "text": "How many fewer primary pupils by 2035?"},
    {"area": "Education", "text": "What fertility change would keep children aged 0-15 at today's level by 2040?"},
    {"area": "Housing", "text": "How many extra homes will we need by 2040 at +600 net migration?"},
    {"area": "Housing", "text": "What net migration is consistent with 2,000 additional homes by 2040?"},
    {"area": "Workforce", "text": "What is the chance the working-age population is below today's level in 2040?"},
    {"area": "Workforce", "text": "What net migration gives an 80% chance of keeping the working-age population at today's level by 2040?"},
    {"area": "Workforce", "text": "What net migration would keep the old-age ratio at today's level by 2040?"},
    {"area": "Statistics", "text": "Reproduce Statistics Jersey's published +600 projection."},
    {"area": "Statistics", "text": "Is this a forecast? Can I trust these numbers?"},
]
MODES = ["REHEARSAL", "PRESENTATION", "OPEN_DEMO", "DRAINING", "CLOSED"]
# two codes only: the hackathon access code (JFE_EVENT_CODE) and the admin code (JFE_OPERATOR_CODE), which also opens /admin
TEMPLATES = Path(__file__).parent / "templates"
basic = HTTPBasic(auto_error=False)

state = {}
llm_sem = asyncio.Semaphore(LLM_ACTIVE)
llm_waiting = 0
hits = defaultdict(deque)


@asynccontextmanager
async def lifespan(app):
    db.init()
    pack = load_pack()
    try:  # the landing view's receipt should show the same GPU path as every other run
        run_experiment(product_default(end_year=2030), pack, "gpu")  # compiles the CUDA kernels, so the receipt shows a warm run
        state["baseline"] = run_experiment(product_default(), pack, "gpu")
    except Exception:
        state["baseline"] = run_experiment(product_default(), pack)
    state["backtest"] = evidence.backtest(write=False)
    state["http"] = httpx.AsyncClient()
    yield
    await state["http"].aclose()


app = FastAPI(title="Vraic Futures Engine", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(GZipMiddleware, minimum_size=2000)


@app.exception_handler(RequestValidationError)
async def request_validation(_request: Request, e: RequestValidationError):
    field = ".".join(str(x) for x in e.errors()[0].get("loc", ())[1:]) if e.errors() else ""
    msg = {"prompt": "Type a question of up to 4,000 characters."}.get(field, "That request is not in the expected format.")
    return JSONResponse({"detail": msg}, status_code=422)


@app.middleware("http")
async def security(request: Request, call_next):
    if request.method in ("POST", "PATCH", "DELETE"):
        if request.headers.get("x-jfe") != "1":
            return JSONResponse({"detail": "Missing request header."}, status_code=403)
        if int(request.headers.get("content-length") or 0) > 64 * 1024:
            return JSONResponse({"detail": "Request too large."}, status_code=413)
    resp = await call_next(request)
    resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; font-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["Referrer-Policy"] = "no-referrer"
    return resp


def limit(key, n, per=60.0):
    now = time.monotonic()
    dq = hits[key]
    while dq and dq[0] < now - per:
        dq.popleft()
    if len(dq) >= n:
        db.event("rate_limited", key.rsplit(":", 1)[-1])
        raise HTTPException(429, "Please wait a moment before trying again.", headers={"Retry-After": str(int(per - (now - dq[0])) + 1)})
    dq.append(now)


def session(request: Request):
    sid = request.cookies.get("jfe_session")
    row = sid and db.one("select * from sessions where id=%s and expires > now()", sid)
    if not row:
        raise HTTPException(401, "Enter the access code to start a session.")
    return row


def via_funnel(request: Request):
    # Tailscale Funnel sets this on every public-internet request and strips any copy the client sends
    return "tailscale-funnel-request" in request.headers


def admin(request: Request, creds: HTTPBasicCredentials | None = Depends(basic)):
    """Admin = a session started with the admin code, or HTTP Basic with the admin code as password (for scripts). Never via Funnel."""
    if via_funnel(request):
        raise HTTPException(404, "Not Found")
    sid = request.cookies.get("jfe_session")
    if sid and db.one("select 1 from sessions where id=%s and expires > now() and is_operator", sid):
        return "admin"
    if OPERATOR_CODE and creds and hmac.compare_digest(creds.password, OPERATOR_CODE):
        return "admin"
    raise HTTPException(401, "Admin code required.", headers={"WWW-Authenticate": 'Basic realm="vraic-fe admin"'})


def operator(request: Request, s=Depends(session)):
    if via_funnel(request):
        raise HTTPException(404, "Not Found")
    if not s["is_operator"]:
        raise HTTPException(403, "Operator access only.")
    return s


class In(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionIn(In):
    code: str = Field(max_length=64)


class HistoryItem(In):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class DraftIn(In):
    prompt: str = Field(min_length=1, max_length=4096)
    parent_run_id: str | None = None
    parent_draft_id: str | None = None
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)
    current_chart: dict | None = None


class ChartIn(In):
    spec: dict
    format: str = Field(default="json", pattern="^(json|csv)$")


class FormIn(In):
    scenario: dict
    parent_run_id: str | None = None


class PatchIn(In):
    version: int
    scenario: dict


class RunIn(In):
    draft_id: str
    version: int
    idempotency_key: str = Field(min_length=8, max_length=64)


class ExplainIn(In):
    question: str = Field(min_length=1, max_length=1000)


class SaveIn(In):
    run_id: str
    name: str = Field(min_length=1, max_length=120)
    note: str | None = Field(default=None, max_length=1000)


class ModeIn(In):
    mode: str


# ---------------------------------------------------------------- session & catalog
@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/api/session")
def create_session(body: SessionIn, request: Request, response: Response):
    # code guessing: 20 wrong codes a minute per IP (IPv6 per /64); a venue shares one IP, so valid codes are not counted
    ip = request.client.host if request.client else "?"
    if ":" in ip:
        try:
            ip = str(ipaddress.ip_network(f"{ip}/64", strict=False))
        except ValueError:
            pass
    key = f"ip:{ip}:session"
    limit(key, 20)
    is_op = bool(OPERATOR_CODE) and not via_funnel(request) and hmac.compare_digest(body.code.strip(), OPERATOR_CODE)
    if not is_op and not (EVENT_CODE and hmac.compare_digest(body.code.strip().upper(), EVENT_CODE.upper())):
        raise HTTPException(403, "That access code is not valid.")
    hits[key].pop()
    if db.mode() == "CLOSED" and not is_op:
        raise HTTPException(503, "The event demo is closed.")
    sid = db.new_id("s")
    db.q(f"insert into sessions (id, expires, is_operator) values (%s, now() + interval '{RETENTION}', %s)", sid, is_op)
    response.set_cookie("jfe_session", sid, max_age=48 * 3600, httponly=True, secure=SECURE_COOKIE, samesite="strict")
    return {"operator": is_op, "retention": RETENTION, "mode": db.mode()}


@app.delete("/api/session")
def end_session(response: Response, s=Depends(session)):
    db.q("update sessions set expires = now() where id = %s", s["id"])
    response.delete_cookie("jfe_session", httponly=True, secure=SECURE_COOKIE, samesite="strict")
    return {"ok": True}


@app.get("/api/session")
def get_session(s=Depends(session)):
    return {"operator": s["is_operator"], "retention": RETENTION, "expires": s["expires"].isoformat(), "mode": db.mode()}


@app.get("/api/catalog")
def catalog():
    pack = load_pack()
    return {
        "model_version": MODEL_VERSION, "pack_id": pack.id, "gate": pack.meta["gate"], "first_year": FIRST_YEAR, "max_end_year": MAX_END_YEAR,
        "net_limits": NET_LIMITS, "official_net_range": OFFICIAL_NET_RANGE, "fertility_multiplier": FERT_MULT, "mortality_multiplier": MORT_MULT,
        "codes": CODE_LABEL, "public_max_draws": PUBLIC_MAX_DRAWS, "operator_max_draws": OPERATOR_MAX_DRAWS,
        "metrics": {k: {"label": v, "unit": UNITS[k]} for k, v in METRICS.items()}, "examples": EXAMPLES,
        "default_scenario": product_default().model_dump(), "default_draws": DEFAULT_DRAWS, "mode": db.mode(), "llm_model": compiler.LLM_MODEL,
        "context": [
            {"text": "Statistics Jersey provisionally estimates 104,490 residents at the end of 2025.", "source": "S02, Sept 2026"},
            {"text": "Between 2020 and 2025 the population aged 65+ grew by 13% while the under-16 population fell by 7%.", "source": "S02, Sept 2026"},
        ],
    }


@app.get("/api/sources")
def sources():
    pack = load_pack()
    manifest = json.loads((Path(os.environ.get("JFE_MANIFEST", "data/manifests/sources.json"))).read_text())
    return {"pack_id": pack.id, "built": pack.meta["built"], "gate": pack.meta["gate"], "sources": pack.meta["sources"],
            "documents": manifest.get("documents", []), "quality": pack.meta["quality"]}


@app.get("/api/baseline")
def baseline():
    """Precomputed official-mid-range rebased scenario: browsing works without any compute."""
    return state["baseline"]


@app.get("/api/backtest")
def backtest():
    """Retrospective check against what actually happened (2018-2025) and reproduction of the official projection."""
    return state["backtest"]


# ---------------------------------------------------------------- charts over stored results (no re-simulation)
def _merge_chart(llm: dict | None, current: dict | None, run_id: str | None) -> charts.ChartSpec:
    llm = dict(llm or {})
    cur = dict(current or {})
    if not llm.get("metrics") and cur.get("metrics") and llm.get("kind") in ("trajectory", "difference", "fan"):
        llm["metrics"] = cur["metrics"]
    if llm.get("kind") in ("response", "frontier") and cur.get("target"):
        llm["target"] = cur["target"]
    if llm.get("basis") is None:
        llm.pop("basis", None)
    compare = llm.pop("compare_previous", False)
    if compare:
        llm["kind"] = "trajectory"  # run comparisons are trajectories
    if llm.get("kind") in ("trajectory", "difference", "fan", "reconciliation"):
        llm.pop("years", None)  # only pyramids, rankings and housing use years
    if llm.get("kind") != "ranking":
        llm.pop("basis", None)
    llm["run_id"] = run_id or "baseline"
    spec = charts.ChartSpec.model_validate(llm)
    if compare:
        spec.compare_run_ids = ["__previous__"]
    return spec


def _chart(s, spec: charts.ChartSpec):
    if spec.run_id == "baseline":
        result = state["baseline"]
    else:
        if not s:
            raise HTTPException(401, "Enter the event code to start a session.")
        result = _result_for(s, spec.run_id)
        if not result:
            raise charts.ChartError("That run has no results yet.")
    others = {}
    if spec.compare_run_ids and s:
        ids = spec.compare_run_ids
        if ids == ["__previous__"]:
            ids = [r["id"] for r in db.q("select id from runs where session_id=%s and status='complete' and id <> %s order by finished desc limit 2", s["id"], spec.run_id)]
        for rid in ids[:2]:
            res = _result_for(s, rid)
            if res:
                others[rid] = res
        spec.compare_run_ids = list(others)
    return charts.build(spec, result, others, state["backtest"])


def optional_session(request: Request):
    sid = request.cookies.get("jfe_session")
    return sid and db.one("select * from sessions where id=%s and expires > now()", sid)


@app.post("/api/chart-data")
def chart_data(body: ChartIn, request: Request, s=Depends(optional_session)):
    """The exact observations behind a chart: same table for the chart, its table view and its CSV download."""
    try:
        spec = charts.ChartSpec.model_validate(body.spec)
        if spec.kind in ("response", "frontier"):
            limit(f"chart:{s['id'] if s else request.client.host}", 30)
        c = _chart(s, spec)
    except ValidationError as e:
        raise HTTPException(422, compiler.friendly(e))
    except charts.ChartError as e:
        raise HTTPException(409, str(e))
    if body.format == "csv":
        name = f"vraic-fe-{spec.kind}-{spec.run_id[-8:]}.csv"
        return Response(charts.to_csv(c), media_type="text/csv; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "private, no-store"})
    return c


# ---------------------------------------------------------------- inverse simulation (solve one assumption for a target)
def _fmt_metric(metric, v):
    return f"{v:,.1f}" if UNITS[metric] in ("per 100", "index") else f"{v:,.0f}"


def _fmt_param(param, v):
    return f"{v:+,.0f} a year" if param == "net_migration" else f"×{v:.3g}"


PARAM_NAME = {"net_migration": "Net migration", "fertility_multiplier": "Fertility rates", "mortality_multiplier": "Death rates"}


def _lower_first(t):
    return t if len(t) > 1 and t[1].isupper() else t[:1].lower() + t[1:]  # "GP appointments" stays, "Population" -> "population"


def _solve_text(spec, res):
    m = charts.SHORT.get(spec.metric, METRICS[spec.metric])
    goal = ("its end-2025 level" if spec.target_kind == "hold_base" else "the target" if spec.target_kind == "value"
            else f"{spec.target_value:+g}% on end-2025")
    tgt = _fmt_metric(spec.metric, res["target"])
    solved_for = f"{m} = {tgt} in {spec.year} ({goal})"
    if res["status"] == "solved" and res.get("probabilistic"):
        side = "at least" if res["direction"] == "at_least" else "at most"
        msg = (f"{PARAM_NAME[spec.parameter]} {_fmt_param(spec.parameter, res['value'])} from {spec.from_year}: {res['achieved_probability']:.0%} chance that "
               f"{_lower_first(m)} is {side} {tgt} in {spec.year} (median {_fmt_metric(spec.metric, res['median_outcome'])}). "
               f"For the median alone: {_fmt_param(spec.parameter, res['median_value'])}.")
        solved_for = f"{res['probability']:.0%} chance: {m} {side} {tgt} in {spec.year}"
    elif res["status"] == "solved":
        msg = (f"{PARAM_NAME[spec.parameter]} {_fmt_param(spec.parameter, res['value'])} from {spec.from_year}: {_lower_first(m)} "
               f"{_fmt_metric(spec.metric, res['achieved'])} in {spec.year} (target {tgt}).")
        if res.get("ceiling"):  # "keep X below T": the solved value is the limit, and the baseline may already be inside it
            msg = (f"{PARAM_NAME[spec.parameter]} up to {_fmt_param(spec.parameter, res['value'])} from {spec.from_year} keeps {_lower_first(m)} "
                   f"{'at or below' if res['ceiling'] == 'at_most' else 'at or above'} {tgt} in {spec.year}"
                   + ".")
        if res.get("probability") and not res.get("probabilistic"):
            msg += " Homes follow the official method from the assumed path and have no range, so this is the single answer."
        vals = [x["value"] for x in res.get("sensitivity", []) if x["status"] == "solved"]
        which = "fertility and life-expectancy" if spec.parameter == "net_migration" else "life-expectancy" if spec.parameter == "fertility_multiplier" else "fertility"
        if vals:
            lo, hi = _fmt_param(spec.parameter, min(vals)), _fmt_param(spec.parameter, max(vals))
            span = f"{lo.removesuffix(' a year')} to {hi}" if spec.parameter == "net_migration" else f"{lo} to {hi}"
            msg += f" Same under official {which} variants." if lo == hi else f" Official {which} variants: {span}."
            if len(vals) < len(res["sensitivity"]):
                msg += " Not reachable under some variants."
        elif res.get("sensitivity"):
            msg += f" Not reachable under the official {which} variants."
    elif res["status"] == "unreachable":
        lo, hi = res["range"]
        msg = (f"Not reachable between {_fmt_param(spec.parameter, lo)} and {_fmt_param(spec.parameter, hi)}: closest {_lower_first(m)} in {spec.year} "
               f"is {_fmt_metric(spec.metric, res['closest_metric'])} (target {tgt}) at {_fmt_param(spec.parameter, res['closest_value'])}.")
    elif res["status"] == "independent":
        msg = f"{m} does not depend on {PARAM_NAME[spec.parameter].lower()} in this model."
    else:
        msg = "No unique answer: the response is not monotonic in this range."
    return msg, solved_for


async def _solve(s, r, detail, parent, parent_run_id, viewed_run_id):
    base = parent or product_default()
    pt = r.get("patch") or {}
    if any(pt.get(k) for k in ("migration_segments", "fertility", "mortality", "baseline", "draws", "uncertainty_mode", "_migration", "_rates")):
        try:  # conditions stated with the target ("under low fertility", "2,000 simulations", "fixed migration") apply before solving
            base = compiler.apply_patch(base, {k: pt.get(k) for k in ("migration_segments", "fertility", "mortality", "baseline", "title", "draws", "uncertainty_mode", "_migration", "_rates")}, parent_run_id)
        except (ValidationError, ValueError) as e:
            r["status"], r["message"] = "needs_clarification", compiler.friendly(e)
            return
    prompt = (r.get("patch") or {}).get("_prompt", "")
    if re.search(r"\b(births|deaths)\b(?![- ]rates?)|\bshare\b|\bproportion\b|per ?cent of (the )?(population|residents)|% of (the )?(population|residents)", prompt, re.I):
        # the solver targets counts and ratios only; never answer a different question silently
        r["status"], r["message"] = "unsupported", ("Goal seeking works on population counts by age, pupils, staff, GP appointments, homes and the old-age and "
                                                    "dependency ratios; births, deaths and population shares can't be targeted. The old-age ratio is the closest measure of ageing.")
        r["unsupported_reason"] = None
        return
    raw = {k: v for k, v in (r.get("target") or {}).items() if v is not None}
    stated = [float(x.replace(",", "")) for x in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,6}\b", prompt) if not 1990 <= float(x.replace(",", "")) <= 2100]
    if raw.get("target_kind") in ("hold_base", "change_pct") and stated and not re.search(r"today|current|now|2025 level|present|\d+\s*%\s*(higher|lower|more|less|rise|fall|increase|decrease)", prompt, re.I):
        raw.update(target_kind="value", target_value=stated[0])  # "keep the population at 110,000" is that number, not today's level
    raw.setdefault("year", base.end_year)
    raw.setdefault("from_year", FIRST_YEAR)
    try:
        spec = inverse.TargetSpec.model_validate(raw)
    except ValidationError as e:
        r["status"], r["message"] = "needs_clarification", compiler.friendly(e)
        return
    m_p = re.search(r"(\d{2}(?:\.\d)?)\s*(?:%|percent|per cent)\s*(?:chance|probability|likely|likelihood|confiden\w*|certain\w*|sure)|(?:probability|chance|confidence)\s+of\s+(?:at\s+least\s+)?(\d{2})\s*(?:%|percent)", prompt, re.I)
    m_in = re.search(r"\b(\d{1,2})\s+(?:in|out of)\s+(10|ten)\b", prompt, re.I)
    # IPCC calibrated language when no number is given: likely >= 66%, very likely >= 90%, extremely likely >= 95%, virtually certain >= 99%
    m_w = re.search(r"\b(virtually certain|extremely likely|very likely|likely)\b", prompt, re.I)
    prob = float(m_p.group(1) or m_p.group(2)) / 100 if m_p else (int(m_in.group(1)) / 10 if m_in else
                                                                 {"virtually certain": 0.99, "extremely likely": 0.95, "very likely": 0.9, "likely": 0.66}[m_w.group(1).lower()] if m_w else None)
    if prob is not None and spec.target_kind == "change_pct" and abs((spec.target_value or 0) - 100 * prob) < 1e-6:
        # the model read "75% chance" as "+75%": the target is the stated number, or today's level
        big = [float(x.replace(",", "")) for x in re.findall(r"\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b", prompt) if not 2000 <= float(x.replace(",", "")) <= 2100]
        spec = spec.model_copy(update={"target_kind": "value", "target_value": big[0]} if big else {"target_kind": "hold_base", "target_value": None})
    if prob is not None and (base.uncertainty.mode == "deterministic" or base.baseline_id != "observed_end2025") and not re.search(r"deterministic|single run", prompt, re.I):
        # a chance needs an ensemble, and ensembles start from the observed (whole-person) end-2025 population
        base = base.model_copy(update={"uncertainty": product_default().uncertainty, "baseline_id": "observed_end2025"})
    if prob is not None and 0.5 <= prob <= 0.99:
        dp = re.sub(r"\b(over|under)[- ]?\d+s?\b|\baged \d+\s*(and over|or over|\+)?", " ", prompt, flags=re.I)  # "over-65s" names an age group, not a side
        direction = "at_least" if re.search(r"(not|n't|never)\s+(fall|drop|go|dip|sink)\s+(below|under)|\b(at least|no fewer than|no less than|stay(s)? above|above|over)\b", dp, re.I) else \
            "at_most" if re.search(r"(not|n't|never)\s+(exceed|rise above|go above|top)|\b(at most|no more than|below|under|stay(s)? below)\b", dp, re.I) else None
        try:
            res = await run_in_threadpool(inverse.solve_probability, base, spec, prob, direction)
        except ValueError as e:
            r["status"], r["message"] = "needs_clarification", compiler.friendly(e)
            return
    else:
        res = await run_in_threadpool(inverse.solve, base, spec)
        if res["status"] == "solved" and re.search(r"\b(below|under|at most|no more than|not exceed|stay(s)? below|keep\w* .{0,40}below)\b", prompt, re.I):
            res["ceiling"] = "at_most"
        elif res["status"] == "solved" and re.search(r"\b(above|at least|no fewer than|not fall below|stay(s)? above)\b", prompt, re.I) and spec.target_kind != "hold_base":
            res["ceiling"] = "at_least"
    msg, solved_for = _solve_text(spec, res)
    detail["interpretation"] = []  # the solver's message states what was solved; model bullets here were unreliable
    r["message"] = msg
    detail["solve"] = {k: res.get(k) for k in ("status", "value", "achieved", "target", "base_value", "sensitivity", "range", "parameter_label",
                                               "closest_value", "closest_metric", "probability", "direction", "achieved_probability",
                                               "median_outcome", "median_value")} | {"spec": spec.model_dump()}
    result = _result_for(s, viewed_run_id) if viewed_run_id else state["baseline"]
    if result:
        cond = base.model_dump() if base.model_dump(exclude={"title", "parent_scenario_id", "solved_for"}) != Scenario.model_validate(result["scenario"] | {"solved_for": None}).model_dump(exclude={"title", "parent_scenario_id", "solved_for"}) else None
        # "which combinations of migration and fertility…" is a two-parameter question: show the frontier, not one curve
        two = bool(re.search(r"combination|trade-?off|frontier", prompt, re.I)) or (re.search(r"migra", prompt, re.I) and re.search(r"fertil|birth", prompt, re.I))
        kind = "frontier" if two and spec.parameter == "net_migration" else "response"
        tspec = spec.model_copy(update={"probability": res.get("probability"), "direction": res.get("direction")}) if res.get("probabilistic") else spec
        detail["chart"] = await run_in_threadpool(charts.build, charts.ChartSpec(kind=kind, target=tspec.model_dump(), run_id=viewed_run_id or "baseline", base_scenario=cond), result)
    if res["status"] == "solved":
        sc = inverse.apply(base, spec, res["value"], solved_for=solved_for)
        chance = f"{res['probability']:.0%} chance: " if res.get("probabilistic") else ""
        title = f"{PARAM_NAME[spec.parameter]} {_fmt_param(spec.parameter, res['value'])} ({chance}{_lower_first(charts.SHORT.get(spec.metric, METRICS[spec.metric]))}, {spec.year} target)"
        r["scenario"] = sc.model_copy(update={"parent_scenario_id": parent_run_id, "title": title[:120]})
        r["parent"] = base
        r["status"] = "ready_for_review"
    else:
        r["status"] = "unreachable" if res["status"] in ("unreachable", "independent") else "needs_clarification"
    db.event("solve", res["status"])


# ---------------------------------------------------------------- drafts
def _parent(s, run_id, draft_id):
    if run_id:
        r = db.one("select id, scenario from runs where id=%s and session_id=%s", run_id, s["id"])
        if not r:
            raise HTTPException(404, "Parent run not found.")
        return Scenario.model_validate(r["scenario"]), r["id"]
    if draft_id:
        d = db.one("select id, scenario from drafts where id=%s and session_id=%s and scenario is not null order by version desc limit 1", draft_id, s["id"])
        if not d:
            raise HTTPException(404, "Parent draft not found.")
        return Scenario.model_validate(d["scenario"]), None
    return None, None


def _save_draft(s, status, sc, parent, parent_run_id, message, detail, draft_id=None, version=1):
    did = draft_id or db.new_id("d")
    db.q("insert into drafts (id, session_id, version, status, scenario, parent_run_id, message, detail) values (%s,%s,%s,%s,%s,%s,%s,%s)",
         did, s["id"], version, status, db.J(sc.model_dump()) if sc else None, parent_run_id, message, db.J(detail))
    out = {"draft_id": did, "version": version, "status": status, "message": message} | detail
    if sc:
        out |= {"scenario": sc.model_dump(), "assumptions": describe(sc), "diff": diff(parent, sc), "warnings": sc.warnings(), "experiment_hash": experiment_hash(sc, load_pack())}
    return out


@app.post("/api/drafts")
async def create_draft(body: DraftIn, s=Depends(session)):
    global llm_waiting
    limit(f"s:{s['id']}:prompt", 12)
    parent, parent_run_id = _parent(s, body.parent_run_id, body.parent_draft_id)
    if parent and re.fullmatch(r"\s*(please\s+)?(undo( that| it| this| the last change)?|go back( a step| one step)?|revert( that| it)?|step back)\s*[.!]?\s*", body.prompt, re.I):
        # "undo that": the scenario this run was derived from (or the official baseline), as a draft to review
        gp = parent.parent_scenario_id and db.one("select scenario from runs where id=%s and session_id=%s", parent.parent_scenario_id, s["id"])
        back = Scenario.model_validate(gp["scenario"]) if gp else product_default(end_year=parent.end_year)
        back = back.model_copy(update={"parent_scenario_id": parent_run_id, "solved_for": None})
        return _save_draft(s, "ready_for_review", back, parent, parent_run_id, "Back to the previous assumptions.",
                           {"intent": "modify_scenario", "interpretation": [], "title": back.title})
    static = compiler.static_answer(body.prompt) or compiler.migration_chance(body.prompt, parent)
    if static:  # fixed facts about the tool itself: never left to the model
        return _save_draft(s, "explain", None, None, None, "", {"intent": "explain", "interpretation": [],
                           "explanation": {"sentences": static, "claims": [], "grounded_llm": True, "rule": True, "latency_seconds": 0.0}})
    if llm_waiting >= LLM_WAITING:
        raise HTTPException(429, "The language model queue is full. You can still use the assumptions form.", headers={"Retry-After": "15"})
    llm_waiting += 1
    acquired = False
    try:
        async with llm_sem:
            llm_waiting, acquired = llm_waiting - 1, True
            r = await compiler.draft(state["http"], body.prompt, parent, parent_run_id, [h.model_dump() for h in body.history], body.current_chart)
    except compiler.LLMUnavailable:
        db.event("draft", "llm_unavailable")
        raise HTTPException(503, "Interpretation is temporarily unavailable. Use the assumptions form instead; it runs the same validated model.")
    finally:
        if not acquired:
            llm_waiting -= 1
    db.event("draft", r["status"], r["latency_seconds"])
    detail = {k: r.get(k) for k in ("intent", "interpretation", "clarification", "unsupported_reason", "latency_seconds", "title")}
    if r["status"] == "solve_target":
        await _solve(s, r, detail, parent, parent_run_id, body.parent_run_id)
    if r["status"] == "chart":
        try:
            detail["chart"] = _chart(s, _merge_chart(r["chart"], body.current_chart, body.parent_run_id))
        except (charts.ChartError, ValidationError) as e:
            if isinstance(e, charts.ChartError) and "different units" in str(e):
                r["status"], r["message"] = "explain", ""  # several measures at once: answered as numbers below, not refused
            else:
                r["status"] = "needs_clarification"
                r["message"] = str(e) if isinstance(e, charts.ChartError) else "That chart is not supported."
                detail["clarification"] = None
    elif r["status"] == "download":
        detail["download"] = {"target": r["download"], "chart_spec": body.current_chart, "run_id": body.parent_run_id}
    elif r.get("scenario") and r.get("chart"):
        try:  # normalised to the strict ChartSpec shape; run_id is filled in by the client after the run
            after = _merge_chart(r["chart"], None, None)
            if after.kind not in ("response", "frontier"):  # target charts belong to the question, not the run
                detail["chart_after_run"] = after.model_dump(exclude={"run_id", "compare_run_ids"})
        except ValidationError:
            pass
    if r["status"] == "explain":
        # explain against the viewed run, or the public mid-range baseline when nothing has been run yet
        res = _result_for(s, body.parent_run_id) if body.parent_run_id else state["baseline"]
        if res:
            ex = compiler.fact_explanation(body.prompt, res)  # look-ups need no model slot
            if ex is None:
                async with llm_sem:
                    ex = await compiler.explain(state["http"], body.prompt, res, retry=llm_waiting <= 8)
            db.event("explain", "grounded" if ex["grounded_llm"] else "template", ex["latency_seconds"])
            detail["explanation"] = ex
    if r.get("scenario") and r["scenario"].uncertainty.draws > (OPERATOR_MAX_DRAWS if s["is_operator"] else PUBLIC_MAX_DRAWS):
        r["status"], r["scenario"] = "needs_clarification", None
        r["message"] = f"Runs use up to {PUBLIC_MAX_DRAWS:,} simulations; the default is {DEFAULT_DRAWS:,}."
    if r["status"] == "needs_clarification" and detail.get("clarification"):
        r["message"], detail["clarification"] = detail["clarification"], None  # one question, not the message and the question
    _deterministic_reply(r, detail)
    return _save_draft(s, r["status"], r.get("scenario"), r.get("parent"), parent_run_id if r.get("scenario") and r["scenario"].parent_scenario_id else None, r["message"], detail)


DOWNLOAD_LABEL = {"chart_data": "Chart data (CSV).", "chart_image": "Chart image.", "run_data": "Run data (CSV).", "assumptions": "Assumptions and manifest (JSON).",
                  "report": "Briefing report.", "experiment_zip": "Experiment package (ZIP)."}


def _deterministic_reply(r, detail):
    """Replies that describe what will actually happen are generated from the validated objects, not the model's prose."""
    if r["status"] == "ready_for_review" and not detail.get("solve"):
        d = diff(r.get("parent"), r["scenario"])
        changed = [c["field"] for c in d["changed"]]
        if not changed and r["scenario"].parent_scenario_id is None:
            r["message"] = "Same as the official mid-range assumptions."  # still runnable
        elif not changed:
            r["status"], r["scenario"] = "needs_clarification", None
            r["message"] = "Nothing changed: that request does not alter a supported assumption."
            detail["clarification"] = None
        else:
            r["message"] = ""  # the review card lists the changes
        if r.get("scenario"):  # review-card bullets may only mention numbers that are in what will run
            shown = json.dumps(r["scenario"].model_dump()) + " " + " ".join(c["to"] for c in d["changed"])
            nums = set(re.findall(r"\d+(?:\.\d+)?", shown.replace(",", "")))
            flat = len({sg.net_per_year for sg in r["scenario"].migration.segments}) == 1
            detail["interpretation"] = [b for b in detail.get("interpretation") or []
                                        if set(re.findall(r"\d+(?:\.\d+)?", b.replace(",", ""))) <= nums
                                        and not (flat and re.search(r"linear|ramp", b, re.I))]
    elif r["status"] == "chart" and detail.get("chart"):
        r["message"] = detail["chart"]["title"] + "."
    elif r["status"] == "download" and detail.get("download"):
        r["message"] = DOWNLOAD_LABEL.get(detail["download"]["target"], "Download.")
    elif r["status"] == "explain":
        r["message"] = ""


@app.post("/api/drafts/form")
def form_draft(body: FormIn, s=Depends(session)):
    parent, parent_run_id = _parent(s, body.parent_run_id, None)
    try:
        sc = Scenario.model_validate(body.scenario | {"parent_scenario_id": parent_run_id, "solved_for": None})
    except ValidationError as e:
        raise HTTPException(422, compiler.friendly(e))
    if sc.uncertainty.draws > (OPERATOR_MAX_DRAWS if s["is_operator"] else PUBLIC_MAX_DRAWS):
        raise HTTPException(422, f"Runs use 16 to {PUBLIC_MAX_DRAWS:,} simulations, or a single deterministic run.")
    db.event("draft", "form")
    return _save_draft(s, "ready_for_review", sc, parent, parent_run_id, "Drafted from the assumptions form.", {"intent": "form"})


@app.patch("/api/drafts/{draft_id}")
def patch_draft(draft_id: str, body: PatchIn, s=Depends(session)):
    cur = db.one("select * from drafts where id=%s and session_id=%s order by version desc limit 1", draft_id, s["id"])
    if not cur:
        raise HTTPException(404, "Draft not found.")
    if cur["version"] != body.version:
        raise HTTPException(409, "This draft changed since you opened it; reload it.")
    try:
        sc = Scenario.model_validate(body.scenario | {"solved_for": None})
    except ValidationError as e:
        raise HTTPException(422, compiler.friendly(e))
    parent, _ = _parent(s, cur["parent_run_id"], None)
    return _save_draft(s, "ready_for_review", sc, parent, cur["parent_run_id"], "Edited assumptions.", {"intent": "edit"}, draft_id, cur["version"] + 1)


# ---------------------------------------------------------------- runs
def _run_view(r):
    out = {k: r[k] for k in ("id", "status", "experiment_hash", "cache_status", "backend", "error", "queue_seconds", "compute_seconds", "artifacts_status", "draft_id")}
    out |= {"created": r["created"].isoformat(), "title": r["scenario"]["title"], "scenario": r["scenario"]}
    if r["status"] == "queued":
        mode = db.mode()
        # dispatcher order: operator runs first, then FIFO
        ahead = db.one("select count(*) n from runs where status in ('queued','running') and id <> %s and (is_operator > %s or (is_operator = %s and created < %s))",
                       r["id"], r["is_operator"], r["is_operator"], r["created"])["n"]
        out["queue_position"] = ahead + 1
        # measured worker occupancy per fresh run (simulation, summary and storage), not just the GPU kernel time
        recent = db.q("select extract(epoch from finished - started) s from runs where status='complete' and cache_status='Fresh calculation' and started is not null order by finished desc limit 20")
        if len(recent) >= 3:
            out["estimated_wait_seconds"] = round(out["queue_position"] * sum(float(x["s"]) for x in recent) / len(recent), 1)
        if mode in ("REHEARSAL", "PRESENTATION") and not r["is_operator"]:
            out["paused"] = "Audience compute is paused during the presentation. Your run is kept and will start when the demo opens."
    return out


@app.post("/api/runs", status_code=202)
def create_run(body: RunIn, s=Depends(session)):
    existing = db.one("select * from runs where session_id=%s and idempotency_key=%s", s["id"], body.idempotency_key)
    if existing:
        return _run_view(existing)
    mode = db.mode()
    if mode in ("DRAINING", "CLOSED"):
        raise HTTPException(503, "The demo is not accepting new runs. Completed results remain available.")
    d = db.one("select * from drafts where id=%s and version=%s and session_id=%s", body.draft_id, body.version, s["id"])
    if not d or d["status"] != "ready_for_review":
        raise HTTPException(409, "Confirm the latest reviewed version of this draft.")
    sc = Scenario.model_validate(d["scenario"])
    if sc.uncertainty.draws > (OPERATOR_MAX_DRAWS if s["is_operator"] else PUBLIC_MAX_DRAWS):
        raise HTTPException(422, f"Runs use up to {PUBLIC_MAX_DRAWS:,} simulations.")
    outstanding = db.one("select * from runs where session_id=%s and status in ('queued','running','cancel_requested','numerical_complete')", s["id"])
    if outstanding and not s["is_operator"]:
        raise HTTPException(409, f"You already have a run in progress ({outstanding['id']}). Wait for it or cancel it.")
    if db.one("select count(*) n from runs where status='queued'")["n"] >= MAX_WAITING and not s["is_operator"]:
        raise HTTPException(429, "The shared queue is full. Your draft is kept; try again shortly.", headers={"Retry-After": "30"})
    if not s["is_operator"]:
        limit(f"s:{s['id']}:run", 3)
        limit("event:run", int(os.environ.get("JFE_EVENT_RUNS_PER_MIN", "120")))
    rid = db.new_id("r")
    db.q("insert into runs (id, session_id, draft_id, draft_version, scenario, experiment_hash, status, is_operator, idempotency_key) values (%s,%s,%s,%s,%s,%s,'queued',%s,%s) on conflict (session_id, idempotency_key) do nothing",
         rid, s["id"], d["id"], d["version"], db.J(sc.model_dump()), experiment_hash(sc, load_pack()), s["is_operator"], body.idempotency_key)
    return _run_view(db.one("select * from runs where session_id=%s and idempotency_key=%s", s["id"], body.idempotency_key))


def _owned(s, run_id):
    r = db.one("select * from runs where id=%s and session_id=%s", run_id, s["id"])
    if not r:
        raise HTTPException(404, "Run not found.")
    return r


@app.get("/api/runs")
def list_runs(s=Depends(session)):
    return [_run_view(r) for r in db.q("select * from runs where session_id=%s order by created desc limit 50", s["id"])]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, s=Depends(session)):
    return _run_view(_owned(s, run_id))


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, s=Depends(session)):
    r = _owned(s, run_id)
    db.q("update runs set status = case when status='queued' then 'cancelled' else 'cancel_requested' end, finished = case when status='queued' then now() else finished end where id=%s and status in ('queued','running')", r["id"])
    return _run_view(_owned(s, run_id))


def _result_for(s, run_id):
    r = _owned(s, run_id)
    if r["status"] not in ("numerical_complete", "complete"):
        return None
    row = db.one("select result from results where experiment_hash=%s", r["experiment_hash"])
    # the cache is keyed by numbers only: overlay this run's own scenario so no other session's text is ever served
    return row and (row["result"] | {"scenario": r["scenario"]})


@app.get("/api/runs/{run_id}/result")
def get_result(run_id: str, s=Depends(session)):
    res = _result_for(s, run_id)
    if not res:
        raise HTTPException(409, "Results are not ready yet.")
    r = _owned(s, run_id)
    return res | {"run_id": run_id, "cache_status": r["cache_status"], "queue_seconds": r["queue_seconds"]}


@app.post("/api/runs/{run_id}/explain")
async def explain_run(run_id: str, body: ExplainIn, s=Depends(session)):
    limit(f"s:{s['id']}:prompt", 12)
    res = _result_for(s, run_id)
    if not res:
        raise HTTPException(409, "Results are not ready yet.")
    ex = compiler.fact_explanation(body.question, res)
    if ex is None:
        async with llm_sem:
            ex = await compiler.explain(state["http"], body.question, res, retry=llm_waiting <= 8)
    db.event("explain", "grounded" if ex["grounded_llm"] else "template", ex["latency_seconds"])
    return ex


@app.get("/api/runs/{run_id}/export/{kind}")
def export(run_id: str, kind: str, s=Depends(session)):
    r = _owned(s, run_id)
    try:
        p = exports.path_for(r["id"], kind)
    except KeyError:
        raise HTTPException(404, "Unknown export.")
    if not p.exists():
        raise HTTPException(409, "Exports are not ready for this run.")
    name = f"vraic-fe-{r['id'][-8:]}-{'experiment.zip' if kind == 'zip' else kind}"
    disp = "inline" if kind == "report.html" else "attachment"
    return FileResponse(p, media_type=exports.KINDS[kind], headers={"Content-Disposition": f'{disp}; filename="{name}"', "Cache-Control": "private, no-store"})


# ---------------------------------------------------------------- saved
@app.post("/api/saved")
def save(body: SaveIn, s=Depends(session)):
    r = _owned(s, body.run_id)
    if r["status"] != "complete":
        raise HTTPException(409, "Only completed runs can be saved.")
    sid = db.new_id("v")
    db.q("insert into saved (id, session_id, run_id, name, note) values (%s,%s,%s,%s,%s)", sid, s["id"], r["id"], body.name, body.note)
    return {"id": sid, "run_id": r["id"], "name": body.name, "note": body.note}


@app.get("/api/saved")
def list_saved(s=Depends(session)):
    rows = db.q("select v.id, v.run_id, v.name, v.note, v.created, r.scenario->>'title' title, r.experiment_hash from saved v join runs r on r.id=v.run_id where v.session_id=%s order by v.created desc", s["id"])
    return [r | {"created": r["created"].isoformat()} for r in rows]


# ---------------------------------------------------------------- operator (private: requires operator session)
@app.get("/api/operator/status")
def op_status(_=Depends(operator)):
    counts = {r["status"]: r["n"] for r in db.q("select status, count(*) n from runs group by status")}
    lat = db.q("select compute_seconds, queue_seconds from runs where status='complete' and finished > now() - interval '1 hour' order by finished desc limit 200")
    return {"mode": db.mode(), "modes": MODES, "runs": counts, "sessions_active": db.one("select count(*) n from sessions where expires > now()")["n"],
            "recent": {"n": len(lat), "max_compute_seconds": max((x["compute_seconds"] or 0 for x in lat), default=None),
                       "max_queue_seconds": max((x["queue_seconds"] or 0 for x in lat), default=None)},
            "llm": {"active_limit": LLM_ACTIVE, "waiting": llm_waiting}}


@app.post("/api/operator/mode")
def op_mode(body: ModeIn, _=Depends(operator)):
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    db.q("update settings set value=%s where key='mode'", body.mode)
    if body.mode == "CLOSED":
        db.q("update sessions set expires=now() where not is_operator")
    return {"mode": body.mode}


# ---------------------------------------------------------------- admin dashboard (HTTP Basic; never via Funnel)
@app.get("/admin")
def admin_page(_=Depends(admin)):
    return FileResponse(TEMPLATES / "admin.html", headers={"Cache-Control": "no-store"})


@app.get("/admin.js")
def admin_js(_=Depends(admin)):
    return FileResponse(TEMPLATES / "admin.js", media_type="text/javascript", headers={"Cache-Control": "no-store"})


async def vllm_metrics():
    want = {"vllm:num_requests_running": "running", "vllm:num_requests_waiting": "waiting", "vllm:kv_cache_usage_perc": "kv_cache_usage",
            "vllm:gpu_cache_usage_perc": "kv_cache_usage", "vllm:prompt_tokens_total": "prompt_tokens_total", "vllm:generation_tokens_total": "generation_tokens_total"}
    try:
        r = await state["http"].get(compiler.LLM_URL.removesuffix("/v1") + "/metrics", timeout=3)
        out = {"up": r.status_code == 200}
        for line in r.text.splitlines():
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name in want and not line.startswith("#"):
                out[want[name]] = out.get(want[name], 0) + float(line.rsplit(" ", 1)[-1])
        return out
    except httpx.HTTPError:
        return {"up": False}


PCT = "percentile_cont(0.5) within group (order by {c}) p50, percentile_cont(0.95) within group (order by {c}) p95, max({c}) max"


@app.get("/api/admin/stats")
async def admin_stats(_=Depends(admin)):
    ev = db.q("select kind, status, count(*) n, count(*) filter (where at > now() - interval '1 hour') n_1h from events group by kind, status order by kind, status")
    lat = {k: db.one(f"select count(*) n, {PCT.format(c='latency')} from events where kind=%s and latency is not null and at > now() - interval '24 hours'", k) for k in ("draft", "explain")}
    runs = db.q("select status, count(*) n, count(*) filter (where created > now() - interval '1 hour') n_1h from runs group by status")
    rl = db.one(f"select count(*) n, {PCT.format(c='queue_seconds')} from runs where status='complete' and finished > now() - interval '24 hours'")
    rc = db.one(f"select count(*) n, {PCT.format(c='compute_seconds')} from runs where status='complete' and cache_status='Fresh calculation' and finished > now() - interval '24 hours'")
    e2e = db.one(f"select count(*) n, {PCT.format(c='extract(epoch from finished-created)')} from runs where status='complete' and finished > now() - interval '24 hours'")
    per_min = db.q("""select to_char(m, 'HH24:MI') t, coalesce(d.n,0) drafts, coalesce(r.n,0) runs from generate_series(date_trunc('minute', now()) - interval '29 minutes', date_trunc('minute', now()), interval '1 minute') m
                      left join (select date_trunc('minute', at) mm, count(*) n from events where kind in ('draft','explain') group by 1) d on d.mm=m
                      left join (select date_trunc('minute', created) mm, count(*) n from runs group by 1) r on r.mm=m order by m""")
    recent = db.q("""select right(id, 6) id, status, backend, cache_status, (scenario->'uncertainty'->>'draws')::int draws, scenario->'uncertainty'->>'mode' mode,
                     round(queue_seconds::numeric, 2) queue_s, round(compute_seconds::numeric, 4) compute_s, round(extract(epoch from coalesce(finished, now())-created)::numeric, 2) total_s, is_operator, to_char(created, 'HH24:MI:SS') at
                     from runs order by created desc limit 15""")
    hb = db.one("select data, extract(epoch from now()-at) age from heartbeat where component='worker'")
    return {
        "mode": db.mode(), "modes": MODES, "now": time.strftime("%H:%M:%S"),
        "sessions": db.one("select count(*) filter (where expires > now()) active, count(*) total, count(*) filter (where created > now() - interval '1 hour') new_1h from sessions"),
        "events": ev, "latency": {"llm_draft": lat["draft"], "explain": lat["explain"], "queue_wait": rl, "compute": rc, "end_to_end": e2e},
        "runs": runs, "queued_now": db.one("select count(*) n from runs where status='queued'")["n"], "per_minute": per_min, "recent_runs": recent,
        "worker": (hb["data"] | {"heartbeat_age_s": round(float(hb["age"]), 1)}) if hb else None, "llm": await vllm_metrics() | {"model": compiler.LLM_MODEL, "active_limit": LLM_ACTIVE, "api_waiting": llm_waiting},
        "limits": {"max_waiting_runs": MAX_WAITING, "public_max_draws": PUBLIC_MAX_DRAWS, "prompts_per_session_per_min": 12, "runs_per_session_per_min": 3},
    }


@app.post("/api/admin/mode")
def admin_mode(body: ModeIn, _=Depends(admin)):
    if body.mode not in MODES:
        raise HTTPException(422, f"mode must be one of {MODES}")
    db.q("update settings set value=%s where key='mode'", body.mode)
    if body.mode == "CLOSED":
        db.q("update sessions set expires=now() where not is_operator")
    return {"mode": body.mode}


# ---------------------------------------------------------------- frontend
if not STATIC.exists():
    @app.get("/")
    def no_frontend():
        return RedirectResponse("/admin")

if STATIC.exists():
    app.mount("/assets", StaticFiles(directory=STATIC / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        f = (STATIC / path).resolve()
        if path and f.is_file() and STATIC.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
