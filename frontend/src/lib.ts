// Types, API client, formatting and shared constants. One file on purpose: everything here is small and shared.

// ------------------------------------------------------------------ contract types (mirror backend/jfe/schemas.py)
export type Code = 'L2' | 'L1' | 'MD' | 'H1' | 'H2';
export interface Segment { from_year: number; to_year: number; net_per_year: number }
export interface Point { year: number; value: number }
export interface Rate { assumption: Code; multiplier: Point[] }
export interface Scenario {
  schema_version: '1.0';
  title: string;
  parent_scenario_id: string | null;
  baseline_id: 'observed_end2025' | 'official_state_2025';
  end_year: number;
  migration: { kind: 'fixed_annual_net'; segments: Segment[] };
  fertility: Rate;
  mortality: Rate;
  uncertainty: { mode: 'deterministic' | 'process'; draws: number; migration?: 'fixed' | 'historical'; rates?: 'fixed' | 'official_range' };
  seed: number;
}

export type Summary = Record<string, number[]>; // {value} or {mean,p05,p25,p50,p75,p95}
export interface Result {
  model_version: string;
  housing?: { years: number[] };
  pack_id: string;
  experiment_hash: string;
  scenario: Scenario;
  comparator: Scenario;
  comparator_label: string;
  years: number[];
  aggregation: string;
  metrics: Record<'scenario' | 'comparator' | 'difference', Record<string, Summary>>;
  age: { scenario: number[][][]; comparator: number[][][]; aggregation: string };
  official: { label: string; matches_scenario_assumptions?: boolean; years: number[]; metrics: Record<string, number[]>; age: number[][][] };
  reproduction?: { note: string; max_abs_difference_people: Record<string, number> } | null;
  observed: { label: string; years: number[]; metrics: Record<string, (number | null)[]> };
  ltc_contributions: { bands: string[]; scenario: number[][] };
  warnings: string[];
  receipt: {
    backend: string; hardware: string; requested_draws: number; completed_draws: number;
    compute_seconds: number; summary_seconds: number; seed: number; precision: string; rng: string;
  };
  run_id?: string;
  cache_status?: string | null;
  queue_seconds?: number | null;
}

export type RunStatus = 'queued' | 'running' | 'cancel_requested' | 'numerical_complete' | 'complete' | 'failed' | 'cancelled' | 'expired';
export const TERMINAL: RunStatus[] = ['complete', 'failed', 'cancelled', 'expired'];
export interface RunView {
  id: string; status: RunStatus; experiment_hash: string; cache_status: string | null; backend: string | null;
  error: string | null; queue_seconds: number | null; compute_seconds: number | null; artifacts_status: string | null;
  draft_id: string; created: string; title: string; scenario: Scenario;
  queue_position?: number; estimated_wait_seconds?: number; paused?: string;
}

export interface AssumptionRow { field: string; value: string; provenance: string }
export interface Draft {
  draft_id: string; version: number;
  status: 'ready_for_review' | 'needs_clarification' | 'unsupported' | 'explain' | 'show_assumptions' | 'chart' | 'download' | 'unreachable';
  message: string; intent?: string; title?: string; interpretation?: string[] | null;
  clarification?: string | null; unsupported_reason?: string | null;
  explanation?: Explanation;
  scenario?: Scenario; assumptions?: AssumptionRow[];
  diff?: { changed: { field: string; from: string; to: string }[]; unchanged: AssumptionRow[] };
  warnings?: string[]; experiment_hash?: string;
  chart?: ChartData; chart_after_run?: ChartSpec;
  download?: { target: DownloadTarget; chart_spec: ChartSpec | null; run_id: string | null };
  solve?: { status: string; value?: number | null; achieved?: number | null; target?: number | null; spec: TargetSpec;
            sensitivity?: { variant: string; status: string; value: number | null }[] };
}
export interface TargetSpec { parameter: 'net_migration' | 'fertility_multiplier' | 'mortality_multiplier'; metric: string; year: number;
  target_kind: 'hold_base' | 'value' | 'change_pct'; target_value?: number | null; from_year: number }
export interface Explanation { sentences: string[]; grounded_llm: boolean }

export interface Catalog {
  model_version: string; pack_id: string; gate: string; first_year: number; max_end_year: number;
  net_limits: [number, number]; official_net_range: [number, number];
  fertility_multiplier: [number, number]; mortality_multiplier: [number, number];
  codes: Record<Code, string>; public_max_draws: number; default_draws: number; operator_max_draws: number;
  metrics: Record<string, { label: string; unit: string }>;
  examples: { area: string; text: string }[]; default_scenario: Scenario; mode: string; llm_model: string;
  context: { text: string; source: string }[];
}
export interface Session { operator: boolean; retention: string; expires?: string; mode: string }
export interface Saved { id: string; run_id: string; name: string; note: string | null; created: string; title: string }

// ------------------------------------------------------------------ API client
export class ApiError extends Error {
  constructor(message: string, public status: number, public retryAfter = 0) { super(message); }
}

const FALLBACK: Record<number, string> = {
  401: 'Enter the access code to start a session.',
  403: 'That action is not allowed.',
  404: 'That item could not be found.',
  409: 'That changed in the meantime; please try again.',
  413: 'That request is too large.',
  422: 'Some values are not supported.',
  429: 'Please wait a moment before trying again.',
  503: 'The service is temporarily unavailable.',
};

export async function api<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  let res: Response;
  try {
    res = await fetch(path, {
      method,
      credentials: 'same-origin',
      headers: method === 'GET' ? { Accept: 'application/json' } : { 'Content-Type': 'application/json', 'X-JFE': '1' },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError('The server could not be reached. Nothing you entered has been lost; please try again.', 0);
  }
  if (res.ok) return res.json() as Promise<T>;
  let detail = '';
  try {
    const j = await res.json();
    if (typeof j?.detail === 'string') detail = j.detail;
  } catch { /* non-JSON error body */ }
  let msg = detail || FALLBACK[res.status] || `Request failed (${res.status}).`;
  const retry = Number(res.headers.get('Retry-After') || 0);
  if (res.status === 429 && retry && !/try again/i.test(msg)) msg += ` Try again in about ${retry} s.`;
  // an expired session anywhere sends the app back to the login page (the initial session probe is handled by its caller)
  if (res.status === 401 && path !== '/api/session') window.dispatchEvent(new Event('jfe-signed-out'));
  throw new ApiError(msg, res.status, retry);
}

export const errText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** Fetch a file (GET, or POST with a JSON body) and hand it to the browser as a download. Errors surface as ApiError. */
export async function download(path: string, body?: unknown, fallbackName = 'vraic-fe-download') {
  let res: Response;
  try {
    res = await fetch(path, body === undefined ? { credentials: 'same-origin' } : {
      method: 'POST', credentials: 'same-origin', headers: { 'Content-Type': 'application/json', 'X-JFE': '1' }, body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError('The server could not be reached; please try again.', 0);
  }
  if (!res.ok) {
    let detail = '';
    try { detail = (await res.json())?.detail ?? ''; } catch { /* non-JSON */ }
    throw new ApiError(detail || FALLBACK[res.status] || `Download failed (${res.status}).`, res.status);
  }
  const name = /filename="?([^";]+)"?/.exec(res.headers.get('Content-Disposition') ?? '')?.[1] ?? fallbackName;
  saveBlob(await res.blob(), name);
  return name;
}

export function saveBlob(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

// ------------------------------------------------------------------ chart specs and server chart tables (POST /api/chart-data)
export type ChartKind = 'trajectory' | 'fan' | 'difference' | 'pyramid' | 'ranking' | 'reconciliation' | 'response' | 'frontier' | 'housing';
export interface ChartSpec {
  kind: ChartKind; metrics?: string[]; series?: ('scenario' | 'comparator' | 'official' | 'observed')[]; years?: number[];
  basis?: 'vs_comparator' | 'vs_start'; view?: 'chart' | 'table'; run_id?: string; compare_run_ids?: string[]; target?: TargetSpec;
}
export interface ChartData {
  spec: ChartSpec; title: string; unit: string; columns: string[]; rows: (string | number | null)[][];
  series: { column: string; metric?: string; role: string; label: string }[];
  caption: string; definitions: Record<string, string>; notes: string[]; rounding?: string;
  extra?: { x_label?: string; y_label?: string; x?: number[]; y?: number[]; target?: number;
            solution?: { status: string; value?: number | null; achieved?: number | null; target?: number | null; closest_value?: number | null; closest_metric?: number | null } };
  provenance: { run_id: string; experiment_hash: string; pack_id: string; model_version: string; backend: string; draws: number; seed: number; aggregation: string; scenario_title: string; comparator: string; official: string };
}
export type DownloadTarget = 'chart_data' | 'chart_image' | 'run_data' | 'assumptions' | 'report' | 'experiment_zip';

// ------------------------------------------------------------------ formatting
const nf0 = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 0 });
const nf1 = new Intl.NumberFormat('en-GB', { maximumFractionDigits: 1, minimumFractionDigits: 1 });
export const MINUS = '−';
export const fmt = (v: number | null | undefined, digits = 0) =>
  v == null || Number.isNaN(v) ? '–' : (digits ? nf1 : nf0).format(v).replace('-', MINUS);
export const signed = (v: number, digits = 0) => {
  const r = digits ? Math.round(v * 10) / 10 : Math.round(v);
  return r === 0 ? '0' : (r > 0 ? '+' : MINUS) + (digits ? nf1 : nf0).format(Math.abs(r));
};
export const shortHash = (h?: string) => (h ? h.slice(0, 10) : '–');

/** The central series of a summary: deterministic `value`, or the ensemble median (matches the server chart tables). */
export const central = (s: Summary) => s.value ?? s.p50 ?? s.mean;
export const isProcess = (r: Result) => !('value' in r.metrics.scenario.population_total);

// ------------------------------------------------------------------ chart & domain constants
export const COLORS = { comparator: '#6a6e73', official: '#8476d1', observed: '#151515' };
/** Scenario slots: a run keeps its colour everywhere for as long as it is on the context bar. Validated for CVD separation. */
export const SLOT_COLORS = ['#0066cc', '#009596', '#c46100'];
/** Dark-scheme shades of the same three slots (validated against the dark surface). */
export const SLOT_COLORS_DARK = ['#4394e5', '#139c86', '#cf7029'];
export const slotColors = (dark: boolean) => (dark ? SLOT_COLORS_DARK : SLOT_COLORS);
export const BASELINE_TITLE = 'Official mid-range assumptions from end-2025';
export const VINTAGE = 'End-2025 population (Statistics Jersey, Sept 2026)';
/** Short context-bar label for a scenario's starting state; the full wording comes from describe(). */
export const baselineShort = (sc: Scenario) => (sc.baseline_id === 'observed_end2025' ? VINTAGE : 'Official projected end-2025 (reproduction)');
/** Legend-length form of result.official.label; the full label stays in captions and tables. */
export const officialShort = (label: string) => label.replace(/^Official projection /, 'Official ').replace(/,? mid-range fertility and life expectancy/, '')
  .replace(/,? fertility MD/, '').replace(/,? life expectancy MD/, '').replace(/\s*\(Statistics Jersey, ([^)]+)\)/, ' ($1)');

export interface Backtest {
  pack_id?: string;
  retrospective: {
    description: string;
    variants: { variant: string; end_2025_error_model_minus_observed: Record<string, number>; end_2025_relative_error_pct: Record<string, number>; years: ({ year: number } & Record<string, number>)[] }[];
  };
  official_projection_one_year_check?: { note: string; values: Record<string, { official_projection_2025: number; observed_2025: number }> };
  reference_emulation?: { note: string; max_abs_difference_people: Record<string, number> };
}

export const BANDS: [number, number][] = [...Array.from({ length: 20 }, (_, i) => [i * 5, i * 5 + 4] as [number, number]), [100, 100]];
export const bandLabel = ([lo, hi]: [number, number]) => (lo === 100 ? '100+' : `${lo}-${hi}`);
/** grid[sex][age] -> per-band totals for one sex. */
export const byBand = (grid: number[][], sex: 0 | 1) => BANDS.map(([lo, hi]) => grid[sex].slice(lo, hi + 1).reduce((a, b) => a + b, 0));

/** Mirrors exports.CAVEATS so the in-app limitations match the briefing. */
export const LIMITATIONS = [
  "Experimental scenario from a reduced cohort model: a conditional 'what if', not a forecast or official projection.",
  'Mortality and the migration age/sex profile are derived from the official Feb 2026 projection outputs; fertility uses the published 2023-25 Jersey age pattern, scaled to official births.',
  'Migration is a fixed annual net count with a fixed age/sex profile; gross flows and residential status are not modelled.',
  'Population aged 16-64 is a demographic convention, not the number of workers.',
  'The LTC index counts open Long-Term Care benefit claims at unchanged 2024 rates by age band and sex. It is not total care need, care-home beds or staffing.',
  'No housing, labour-market or policy feedback is modelled.',
];

const CODE_LABEL: Record<Code, string> = { L2: 'lowest', L1: 'low', MD: 'mid-range', H1: 'high', H2: 'highest' };
const sgn = (n: number) => (n >= 0 ? `+${n}` : `-${Math.abs(n)}`);

/** Port of schemas.describe(): plain-language assumption rows with provenance. */
export function describe(sc: Scenario): AssumptionRow[] {
  const segs = sc.migration.segments.map((s) => `${s.from_year}-${s.to_year}: ${sgn(s.net_per_year)} people/year`).join('; ');
  const rate = (ra: Rate, what: string) => {
    const base = `official ${CODE_LABEL[ra.assumption]} (${ra.assumption}) ${what} assumption`;
    if (!ra.multiplier.length) return base;
    const pts = ra.multiplier.map((p) => `x${Number(p.value.toPrecision(3))} in ${p.year}`).join(', ');
    return `${base}, scaled by ${pts} (linear between points, held after the last)`;
  };
  return [
    { field: 'baseline', value: sc.baseline_id === 'observed_end2025' ? 'Observed end-2025 population (Statistics Jersey, Sept 2026, provisional)' : 'Reproduce official projection: official projected end-2025 state (Statistics Jersey, Feb 2026, from its end-2024 base)', provenance: 'published' },
    { field: 'horizon', value: `2026-${sc.end_year}`, provenance: 'user assumption' },
    { field: 'migration', value: `Fixed annual net migration: ${segs}`, provenance: 'user assumption' },
    { field: 'fertility', value: rate(sc.fertility, 'fertility'), provenance: sc.fertility.multiplier.length ? 'user assumption' : 'derived' },
    { field: 'mortality', value: rate(sc.mortality, 'life-expectancy'), provenance: sc.mortality.multiplier.length ? 'user assumption' : 'derived' },
    { field: 'uncertainty', value: sc.uncertainty.mode === 'deterministic' ? 'Deterministic expected values' : `${sc.uncertainty.draws.toLocaleString('en-GB')} simulations; ${sc.uncertainty.migration === 'fixed' ? 'random births and deaths, migration fixed' : 'random births, deaths and net migration'}`, provenance: 'user assumption' },
  ];
}

export const prefersReducedMotion = () => typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

export function readStore<T>(key: string, fallback: T): T {
  try {
    const v = sessionStorage.getItem(key);
    return v ? (JSON.parse(v) as T) : fallback;
  } catch {
    return fallback;
  }
}
export function writeStore(key: string, v: unknown) {
  try { sessionStorage.setItem(key, JSON.stringify(v)); } catch { /* storage unavailable: state just won't survive reload */ }
}
