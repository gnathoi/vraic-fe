import PaletteIcon from '@patternfly/react-icons/dist/esm/icons/palette-icon';
import SignOutIcon from '@patternfly/react-icons/dist/esm/icons/sign-out-alt-icon';
import { Component, useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import {
  Alert, Button, Drawer, DrawerActions, DrawerCloseButton, DrawerContent, DrawerContentBody, DrawerHead, DrawerPanelBody, DrawerPanelContent,
  Dropdown, DropdownItem, DropdownList, Form, FormGroup, FormSelect, FormSelectOption, MenuToggle, Modal, ModalBody, ModalFooter,
  ModalHeader, Popover, Spinner, Tab, TabTitleText, Tabs, TextArea, TextInput, ToggleGroup, ToggleGroupItem,
} from '@patternfly/react-core';
import { Conversation, type ChatItem, EventCode } from './Conversation';
import { Care, ChartFromData, Difference, Fan, Health, Housing, Pyramid, Schools, Trajectories, Validation, setChartTheme, type ActiveChart, type Other } from './charts';
import { setPrefs, useAppearance, type Contrast, type Resolved, type Scheme, type Theme } from './appearance';
import { SourcesPanel } from './Sources';
import {
  BASELINE_TITLE, TERMINAL, slotColors, ApiError, api, central, errText, fmt, isProcess, readStore, signed,
  download, writeStore, type Backtest, type Catalog, type ChartData, type ChartSpec, type Draft, type Result, type RunView, type Saved, type Session,
} from './lib';

type Chip = { id: string; slot: number };
const exportsSettled = (v?: RunView) => v?.artifacts_status === 'ready' || v?.artifacts_status === 'failed';
const HEADLINE: [string, string][] = [['population_total', 'Total population'], ['population_16_64', 'Aged 16-64'], ['population_65_plus', 'Aged 65+']];

// ------------------------------------------------------------------ small pieces
function MetricCards({ r, isBaseline, color }: { r: Result; isBaseline: boolean; color: string }) {
  const end = r.years[r.years.length - 1];
  const proc = isProcess(r);
  return (
    <div className="jfe-cards">
      {HEADLINE.map(([k, label]) => {
        const s = central(r.metrics.scenario[k]);
        const v = s[s.length - 1];
        const delta = isBaseline ? v - s[0] : central(r.metrics.difference[k])[s.length - 1];
        const S = r.metrics.scenario[k];
        return (
          <div className="jfe-card" key={k} style={{ borderTopColor: color }}>
            <div className="jfe-card__label">{label}<span className="jfe-card__year">end-{end}</span></div>
            <div className="jfe-card__value">{fmt(v)}</div>
            <div className="jfe-card__delta"><b>{signed(delta)}</b> {isBaseline ? 'since end-2025' : 'vs baseline'}</div>
            <div className="jfe-card__kind">
              {proc && <>90% range {fmt(S.p05[S.p05.length - 1])}–{fmt(S.p95[S.p95.length - 1])}</>}
              {proc && (() => { const pb = S.p_below_base?.[S.p_below_base.length - 1]; return pb != null && pb > 0.02 && pb < 0.98 ? <div>{Math.round(100 * pb)}% chance below end&#8209;2025</div> : null; })()}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ResultNotes({ r }: { r: Result }) {
  if (!r.warnings.length && !r.reproduction) return null;
  return (
    <section className="jfe-explain">
      {r.warnings.map((w) => <Alert key={w} variant="info" isInline isPlain title={w} />)}
      {r.reproduction && (
        <p className="jfe-repro">Official projection reproduced within {fmt(Math.max(...Object.values(r.reproduction.max_abs_difference_people)), 1)} people.</p>
      )}
    </section>
  );
}

class ViewBoundary extends Component<{ children: ReactNode; resetKey: string }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  componentDidUpdate(prev: { resetKey: string }) { if (prev.resetKey !== this.props.resetKey && this.state.failed) this.setState({ failed: false }); }
  render() {
    return this.state.failed
      ? <Alert variant="warning" isInline title="This view could not be displayed."><Button variant="link" isInline onClick={() => this.setState({ failed: false })}>Try again</Button></Alert>
      : this.props.children;
  }
}

function Receipt({ r, isBaseline }: { r: Result; isBaseline: boolean }) {
  const rc = r.receipt;
  const parts = [
    rc.completed_draws > 1 ? `${fmt(rc.completed_draws)} simulations` : 'Deterministic run',
    rc.backend.startsWith('cuda') ? 'GPU' : 'CPU', `${(rc.compute_seconds + (rc.summary_seconds ?? 0)).toFixed(2)} s`,
  ];
  return (
    <footer className="jfe-receipt" aria-label="Computation receipt" title={`${rc.hardware} · ${rc.backend} · model ${r.model_version} · pack ${r.pack_id}`}>
      {parts.map((p) => <span key={p}>{p}</span>)}
      {!isBaseline && r.cache_status === 'Reused calculation' && <span className="jfe-receipt__cache">Reused calculation</span>}
    </footer>
  );
}

function SaveModal({ runId, title, onClose }: { runId: string; title: string; onClose: () => void }) {
  const [name, setName] = useState(title.slice(0, 120));
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  return (
    <Modal isOpen variant="small" onClose={onClose} aria-labelledby="save-t" className="jfe-app-modal">
      <ModalHeader title="Save scenario" labelId="save-t" />
      <ModalBody>
        <Form id="save-form" onSubmit={async (e) => {
          e.preventDefault();
          setBusy(true);
          try { await api('/api/saved', 'POST', { run_id: runId, name: name.trim(), note: note.trim() || null }); setMsg({ ok: true, text: 'Saved.' }); } catch (err) { setMsg({ ok: false, text: errText(err) }); } finally { setBusy(false); }
        }}>
          <FormGroup label="Name" fieldId="save-name" isRequired><TextInput id="save-name" value={name} maxLength={120} onChange={(_e, v) => setName(v)} /></FormGroup>
          <FormGroup label="Note (optional)" fieldId="save-note"><TextArea id="save-note" value={note} maxLength={1000} onChange={(_e, v) => setNote(v)} resizeOrientation="vertical" /></FormGroup>
          {msg && <Alert variant={msg.ok ? 'success' : 'warning'} isInline isPlain title={msg.text} />}
        </Form>
      </ModalBody>
      <ModalFooter>
        {msg?.ok ? <Button onClick={onClose}>Done</Button> : <Button type="submit" form="save-form" isDisabled={!name.trim() || busy} isLoading={busy}>Save</Button>}
        <Button variant="link" onClick={onClose}>Cancel</Button>
      </ModalFooter>
    </Modal>
  );
}

function SavedModal({ onClose, onView }: { onClose: () => void; onView: (id: string) => void }) {
  const [rows, setRows] = useState<Saved[] | null>(null);
  const [error, setError] = useState('');
  useEffect(() => { api<Saved[]>('/api/saved').then(setRows, (e) => setError(errText(e))); }, []);
  return (
    <Modal isOpen variant="medium" onClose={onClose} aria-labelledby="saved-t" className="jfe-app-modal">
      <ModalHeader title="Saved scenarios" labelId="saved-t" />
      <ModalBody>
        {error && <Alert variant="warning" isInline isPlain title={error} />}
        {!rows && !error && <Spinner size="lg" aria-label="Loading saved scenarios" />}
        {rows && !rows.length && <p className="jfe-muted">Nothing saved yet.</p>}
        <ul className="jfe-saved">
          {rows?.map((s) => (
            <li key={s.id}>
              <div><b>{s.name}</b> <span className="jfe-muted">· {s.title} · {new Date(s.created).toLocaleString('en-GB')}</span></div>
              {s.note && <div className="jfe-muted">{s.note}</div>}
              <Button variant="link" isInline onClick={() => { onView(s.run_id); onClose(); }}>View</Button>
            </li>
          ))}
        </ul>
      </ModalBody>
    </Modal>
  );
}

function OperatorMenu() {
  const [st, setSt] = useState<{ mode: string; modes: string[]; runs: Record<string, number>; sessions_active: number } | null>(null);
  const [error, setError] = useState('');
  const load = () => api<typeof st>('/api/operator/status').then(setSt, (e) => setError(errText(e)));
  return (
    <Popover aria-label="Operator controls" headerContent="Operator" onShow={load} bodyContent={
      <div className="jfe-op">
        {error && <Alert variant="warning" isInline isPlain title={error} />}
        {st ? (
          <>
            <FormGroup label="Mode" fieldId="op-mode">
              <FormSelect id="op-mode" value={st.mode} onChange={async (_e, mode) => {
                try { await api('/api/operator/mode', 'POST', { mode }); await load(); } catch (err) { setError(errText(err)); }
              }}>{st.modes.map((m) => <FormSelectOption key={m} value={m} label={m} />)}</FormSelect>
            </FormGroup>
            <dl className="jfe-op__counts">
              {Object.entries(st.runs).map(([k, n]) => <div key={k}><dt>{k}</dt><dd>{n}</dd></div>)}
              <div><dt>active sessions</dt><dd>{st.sessions_active}</dd></div>
            </dl>
            <Button component="a" variant="link" isInline href="/admin" target="_blank" rel="noopener">Admin dashboard</Button>
          </>
        ) : !error && <Spinner size="md" aria-label="Loading" />}
      </div>
    }>
      <Button variant="link" className="jfe-hide-present">Operator</Button>
    </Popover>
  );
}

function AppearanceMenu({ ap }: { ap: Resolved }) {
  const group = <V extends string>(label: string, value: V, items: [V, string, string?][], set: (v: V) => void) => (
    <div className="jfe-ap__group" role="group" aria-labelledby={`ap-${label}`}>
      <div id={`ap-${label}`} className="jfe-ap__label">{label}</div>
      <ToggleGroup aria-labelledby={`ap-${label}`} isCompact>
        {items.map(([v, text, blocked]) => (
          <ToggleGroupItem key={v} text={text} buttonId={`ap-${label}-${v}`} isSelected={value === v} isDisabled={!!blocked} onChange={() => set(v)} />
        ))}
      </ToggleGroup>
    </div>
  );
  return (
    <Popover aria-label="Appearance" headerContent="Appearance" position="bottom-end" bodyContent={
      <div className="jfe-ap">
        {group<Theme>('Theme', ap.theme, [['default', 'Default'], ['felt', 'Red Hat']], (theme) => setPrefs({ theme }))}
        {group<Scheme>('Colour scheme', ap.scheme, [['light', 'Light'], ['dark', 'Dark'], ['system', 'System']], (scheme) => setPrefs({ scheme }))}
        {group<Contrast>('Contrast', ap.glass ? 'glass' : ap.high ? 'high' : 'default', [['default', 'Default'], ['glass', 'Glass', ap.glassBlocked], ['high', 'High contrast']], (contrast) => setPrefs({ contrast }))}
        {ap.glassBlocked && <p className="jfe-muted">Glass is unavailable: {ap.glassBlocked}</p>}
      </div>
    }>
      <Button variant="plain" aria-label="Appearance" title="Appearance" icon={<PaletteIcon />} />
    </Popover>
  );
}

// ------------------------------------------------------------------ app
export default function App() {
  const [present] = useState(() => new URLSearchParams(location.search).get('present') === '1');
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [baseline, setBaseline] = useState<Result | null>(null);
  const [session, setSession] = useState<Session | null | undefined>(undefined);
  const [fatal, setFatal] = useState('');
  const [runs, setRuns] = useState<Record<string, RunView>>({});
  const [results, setResults] = useState<Record<string, Result>>({});
  const [chips, setChips] = useState<Chip[]>(() => readStore('jfe.chips', []));
  const [viewed, setViewed] = useState<string | null>(() => readStore('jfe.viewed', null));
  const [items, setItems] = useState<ChatItem[]>(() => readStore('jfe.chat', []));
  const [tab, setTab] = useState<string | number>('pyramid');
  const [drawer, setDrawer] = useState(false);
  const [modal, setModal] = useState<'save' | 'saved' | null>(null);
  const [exportOpen, setExportOpen] = useState(false);
  const [mobile, setMobile] = useState<'results' | 'chat'>('chat');
  const [notice, setNotice] = useState('');
  const polling = useRef(new Set<string>());
  // Plot heights fill the viewport left over by the masthead, context bar and receipt (which wrap at narrow widths).
  useEffect(() => {
    const els = ['.jfe-mast', '.jfe-context', '.jfe-receipt'].map((q) => document.querySelector(q)).filter((e): e is Element => !!e);
    const set = () => document.documentElement.style.setProperty('--jfe-chrome', `${els.reduce((a, e) => a + e.getBoundingClientRect().height, 0)}px`);
    const ro = new ResizeObserver(set);
    els.forEach((e) => ro.observe(e));
    return () => ro.disconnect();
  }, [!!catalog && !!baseline]); // once the workspace is rendered
  const ap = useAppearance();
  useMemo(() => setChartTheme(ap.dark), [ap.key]); // re-read PatternFly tokens for charts on any appearance change
  const SLOT_COLORS = slotColors(ap.dark);
  // Charts measure label widths with canvas text metrics: rebuild them once the web fonts are available.
  const [fontsReady, setFontsReady] = useState(false);
  useEffect(() => { document.fonts?.ready.then(() => setFontsReady(true)); }, []);
  const tkey = `${ap.key}${fontsReady ? '' : '-fonts'}`;
  const [backtest, setBacktest] = useState<Backtest | null>(null);
  const [backtestError, setBacktestError] = useState('');
  const [analysis, setAnalysis] = useState<ChartData | null>(null);
  const active = useRef<ActiveChart | null>(null);
  const onActive = useCallback((a: ActiveChart) => { active.current = a; }, []);
  const pendingCharts = useRef<Record<string, ChartSpec>>({}); // chart_after_run, keyed by run id
  /** A fresh conversation on the baseline; the session (and saved scenarios) stay. */
  const resetAll = useCallback(() => {
    setItems([]); setChips([]); setViewed(null); setAnalysis(null); setTab('pyramid'); setMobile('chat'); setNotice('');
  }, []);
  // an analysis chart belongs to the run it was drawn for: switching runs clears it rather than showing stale numbers
  useEffect(() => {
    const rid = analysis?.spec.run_id;
    if (rid && rid !== 'baseline' && rid !== (viewed ?? 'baseline')) setAnalysis(null);
  }, [viewed]);
  const showChart = useCallback((c: ChartData, focus = true) => { setAnalysis(c); setTab('analysis'); if (focus) setMobile('results'); }, []);

  useEffect(() => writeStore('jfe.chips', chips), [chips]);
  useEffect(() => writeStore('jfe.viewed', viewed), [viewed]);
  useEffect(() => writeStore('jfe.chat', items.slice(-60)), [items]);
  useEffect(() => {
    document.documentElement.classList.toggle('jfe-present-root', present);
    const u = new URL(location.href);
    if (present) u.searchParams.set('present', '1'); else u.searchParams.delete('present');
    history.replaceState(null, '', u);
  }, [present]);

  const loadSession = useCallback(() => {
    api<Session>('/api/session').then(setSession, (e) => {
      setSession(null);
      if (!(e instanceof ApiError && e.status === 401)) setNotice(errText(e));
    });
  }, []);

  useEffect(() => {
    api<Catalog>('/api/catalog').then(setCatalog, (e) => setFatal(errText(e)));
    api<Result>('/api/baseline').then(setBaseline, (e) => setFatal(errText(e)));
    loadSession();
  }, [loadSession]);

  const addChip = useCallback((id: string) => setChips((cs) => {
    if (cs.some((c) => c.id === id)) return cs;
    const keep = cs.length >= 3 ? cs.slice(1) : cs;
    const slot = [0, 1, 2].find((s) => !keep.some((c) => c.slot === s)) ?? 0;
    return [...keep, { id, slot }];
  }), []);

  const fetchResult = useCallback(async (id: string) => {
    const r = await api<Result>(`/api/runs/${id}/result`);
    setResults((m) => ({ ...m, [id]: r }));
    return r;
  }, []);

  const viewRun = useCallback(async (id: string) => {
    try {
      if (!results[id]) await fetchResult(id);
      addChip(id);
      setViewed(id);
      setMobile('results');
    } catch (e) { setNotice(errText(e)); }
  }, [results, fetchResult, addChip]);

  /** Polls one run: every 1 s, backing off to 4 s while nothing changes. On completion the result is shown at once;
   *  polling continues until the separately rendered exports are ready (or failed). */
  const poll = useCallback((id: string, showWhenDone = true) => {
    if (polling.current.has(id)) return;
    polling.current.add(id);
    let delay = 1000;
    let last = '';
    let shown = !showWhenDone;
    const stop = () => polling.current.delete(id);
    const tick = async () => {
      try {
        const v = await api<RunView>(`/api/runs/${id}`);
        setRuns((m) => ({ ...m, [id]: v }));
        if (v.status === 'complete') {
          if (!shown) {
            shown = true;
            await fetchResult(id);
            addChip(id);
            setViewed(id);
            setMobile('results');
            const after = pendingCharts.current[id];
            if (after) {
              delete pendingCharts.current[id];
              api<ChartData>('/api/chart-data', 'POST', { spec: { ...after, run_id: id }, format: 'json' }).then(showChart, () => undefined);  // an optional follow-up chart; the result stands without it
            }
          }
          if (v.artifacts_status === 'ready' || v.artifacts_status === 'failed') { stop(); return; }
        } else if (TERMINAL.includes(v.status)) { stop(); return; }
        const key = `${v.status}/${v.artifacts_status}`;
        delay = key === last ? Math.min(delay * 1.25, 4000) : 1000;
        last = key;
      } catch (e) {
        if (e instanceof ApiError && (e.status === 401 || e.status === 404)) { stop(); if (e.status === 401) setSession(null); return; }
        delay = Math.min(delay * 2, 10000);
      }
      setTimeout(tick, delay);
    };
    setTimeout(tick, 1000);
  }, [fetchResult, addChip, showChart]);

  // On a (re)established session: restore this session's runs, resume polling, reload chip results.
  useEffect(() => {
    if (session === undefined) return;
    if (!session) { setChips([]); setViewed(null); return; }
    api<RunView[]>('/api/runs').then((list) => {
      const byId = Object.fromEntries(list.map((v) => [v.id, v]));
      setRuns(byId);
      list.filter((v) => v.status === 'complete' && !exportsSettled(v)).forEach((v) => poll(v.id, false));
      list.filter((v) => !TERMINAL.includes(v.status)).forEach((v) => {
        poll(v.id);
        setItems((xs) => (xs.some((x) => x.kind === 'run' && x.runId === v.id) ? xs : [...xs, { id: crypto.randomUUID(), ts: new Date().toISOString(), kind: 'run', runId: v.id }]));
      });
      setChips((cs) => {
        const ok = cs.filter((c) => byId[c.id]?.status === 'complete');
        ok.forEach((c) => fetchResult(c.id).catch(() => undefined));
        return ok;
      });
      setViewed((v) => (v && byId[v]?.status === 'complete' ? v : null));
    }, (e) => setNotice(errText(e)));
  }, [session, poll, fetchResult]);

  const cancel = async (id: string) => {
    try { const v = await api<RunView>(`/api/runs/${id}/cancel`, 'POST', {}); setRuns((m) => ({ ...m, [id]: v })); } catch (e) { setNotice(errText(e)); }
  };

  const result = (viewed && results[viewed]) || baseline;
  const isBaseline = !viewed || !results[viewed];
  const viewedChip = chips.find((c) => c.id === viewed);
  const color = !isBaseline && viewedChip ? SLOT_COLORS[viewedChip.slot] : SLOT_COLORS[0];
  const others: Other[] = useMemo(() => chips
    .filter((c) => c.id !== viewed && results[c.id] && results[c.id].pack_id === result?.pack_id)
    .map((c, _i, all) => {
      const t = results[c.id].scenario.title;
      const dup = all.filter((x) => x.id !== c.id && results[x.id]?.scenario.title === t).length > 0;
      return { id: c.id, title: `${t.length > 30 ? t.slice(0, 29) + '…' : t}${dup ? ` (${c.id.slice(-4)})` : ''}`, color: SLOT_COLORS[c.slot], result: results[c.id] };
    }), [chips, viewed, results, result, SLOT_COLORS]);
  const runId = isBaseline ? null : viewed;
  const title = isBaseline || !result ? BASELINE_TITLE : result.scenario.title;
  const runLabel = '';
  const exportsReady = !!runId && runs[runId]?.artifacts_status === 'ready';
  const ex = (k: string) => `/api/runs/${runId}/export/${k}`;

  // every hook above the early returns below (React needs the same hooks on every render)
  useEffect(() => {
    const out = () => { resetAll(); setSession(null); };
    window.addEventListener('jfe-signed-out', out);
    return () => window.removeEventListener('jfe-signed-out', out);
  }, [resetAll]);
  if (fatal && (!catalog || !baseline)) {
    return <div className="jfe-fatal"><h1>Vraic Futures Engine</h1><Alert variant="warning" title="The workspace could not load">{fatal} Please reload in a moment.</Alert></div>;
  }
  if (!catalog || !result) {
    return <div className="jfe-fatal"><Spinner size="xl" aria-label="Loading the workspace" /></div>;
  }

  const plot = { r: result, color, present, isBaseline, runLabel, others, tkey, metrics: catalog.metrics, runId: runId ?? 'baseline', onActive };
  const runColors = Object.fromEntries(chips.map((c) => [c.id, SLOT_COLORS[c.slot]]));
  // only the views that exist for this result: no difference on the baseline, no fan for a single run, housing only within its published years
  const hasHousing = !!result.housing?.years?.length;
  const tabKeys = ['pyramid', 'traj', ...(isBaseline ? [] : ['diff']), ...(isProcess(result) ? ['fan'] : []), 'care', 'schools', 'health',
    ...(hasHousing ? ['housing'] : []), ...(isBaseline ? ['valid'] : []), ...(analysis ? ['analysis'] : [])];
  const activeTab = tabKeys.includes(String(tab)) ? tab : 'pyramid';

  /** Carries out a download the assistant asked for. Returns a short note for the conversation. */
  const performDownload = async (d: NonNullable<Draft['download']>): Promise<string> => {
    const rid = d.run_id ?? runId;
    const run = () => { if (!rid) throw new Error('Exports need a completed run.'); return rid; };
    switch (d.target) {
      case 'chart_data':
        if (active.current) return active.current.csv();
        if (!d.chart_spec) throw new Error('No chart is shown.');
        return download('/api/chart-data', { spec: { ...d.chart_spec, view: 'chart' }, format: 'csv' });
      case 'chart_image':
        if (!active.current) throw new Error('No chart is shown.');
        return active.current.image();
      case 'run_data': return download(`/api/runs/${run()}/export/metrics.csv`);
      case 'assumptions': {
        const id = run();
        return `${await download(`/api/runs/${id}/export/scenario.json`)}, ${await download(`/api/runs/${id}/export/manifest.json`)}`;
      }
      case 'report': window.open(`/api/runs/${run()}/export/report.html`, '_blank', 'noopener'); return '';
      case 'experiment_zip': return download(`/api/runs/${run()}/export/zip`);
    }
  };

  const exit = async () => {
    try { await api('/api/session', 'DELETE'); } catch { /* already expired */ }
    resetAll(); setSession(null);
  };

  if (!session) {
    return (
      <main className="jfe-gate">
        <img className="jfe-gate__logo" src="/brand/logo-512.png" width={320} height={320} alt="" />
        <h1 className="jfe-gate__name">Vraic Futures Engine</h1>
        {session === null && <EventCode onDone={loadSession} />}
      </main>
    );
  }

  return (
    <div className={`jfe ${present ? 'jfe--present' : ''}`} data-mobile={mobile}>
      <a className="jfe-skip" href="#jfe-results">Skip to results</a>
      <header className="jfe-mast">
        <div className="jfe-mast__brand">
          <img className="jfe-mast__logo" src="/brand/logo-64.png" srcSet="/brand/logo-64.png 1x, /brand/logo-128.png 2x" width={40} height={40} alt="" />
          <h1 className="jfe-mast__title">Vraic Futures Engine</h1>
        </div>
        <nav className="jfe-mast__actions" aria-label="Workspace">
          <Button variant="link" onClick={() => setDrawer(true)}>Sources</Button>
          {session && <Button variant="link" className="jfe-hide-present" onClick={() => setModal('saved')}>Saved</Button>}
          <Button variant="secondary" size="sm" onClick={resetAll}>Reset</Button>
          <AppearanceMenu ap={ap} />
          {session?.operator && <OperatorMenu />}
          <Button variant="plain" aria-label="Exit" title="Exit" icon={<SignOutIcon />} onClick={exit} />
        </nav>
      </header>

      <div className="jfe-context" role="region" aria-label="Context">
        <div className="jfe-context__facts">
          <span title="Horizon">2026–{result.years[result.years.length - 1]}</span>
        </div>
        <ToggleGroup aria-label="Scenario shown" className="jfe-chips" isCompact>
          <ToggleGroupItem text="Official mid-range" buttonId="chip-base" isSelected={isBaseline} onChange={() => setViewed(null)}
            icon={<span className="jfe-chipdot is-ref" aria-hidden="true" />} />
          {chips.map((c) => {
            const t = results[c.id]?.scenario.title ?? runs[c.id]?.title ?? c.id;
            return <ToggleGroupItem key={c.id} buttonId={`chip-${c.id}`} isSelected={viewed === c.id} onChange={() => viewRun(c.id)}
              text={<span title={t}>{t.length > 32 ? t.slice(0, 31) + '…' : t}</span>}
              icon={<span className="jfe-chipdot" style={{ background: SLOT_COLORS[c.slot] }} aria-hidden="true" />} />;
          })}
        </ToggleGroup>
        <div className="jfe-context__actions jfe-hide-present">
          {runId && runs[runId]?.status === 'complete' && !exportsSettled(runs[runId]) && <span className="jfe-muted" aria-live="polite"><Spinner size="sm" aria-hidden="true" /> Preparing exports…</span>}
          {runId && runs[runId]?.artifacts_status === 'failed' && <span className="jfe-muted">Exports unavailable</span>}
          <Button variant="secondary" size="sm" isDisabled={!exportsReady} onClick={() => setModal('save')}>Save</Button>
          <Dropdown isOpen={exportOpen} onOpenChange={setExportOpen} onSelect={() => setExportOpen(false)} popperProps={{ position: 'right' }}
            toggle={(ref) => <MenuToggle ref={ref} size="sm" onClick={() => setExportOpen(!exportOpen)} isExpanded={exportOpen} isDisabled={!exportsReady}>Export</MenuToggle>}>
            <DropdownList>
              <DropdownItem to={ex('report.html')} target="_blank" rel="noopener" isExternalLink description="Opens in a new tab">Briefing (HTML)</DropdownItem>
              <DropdownItem to={ex('metrics.csv')} download="">Metrics (CSV)</DropdownItem>
              <DropdownItem to={ex('age_distribution.csv')} download="">Age distribution (CSV)</DropdownItem>
              <DropdownItem to={ex('scenario.json')} download="">Assumptions (JSON)</DropdownItem>
              <DropdownItem to={ex('manifest.json')} download="">Manifest (JSON)</DropdownItem>
              <DropdownItem to={ex('result.json')} download="">Result (JSON)</DropdownItem>
              <DropdownItem to={ex('zip')} download="">Experiment package (ZIP)</DropdownItem>
            </DropdownList>
          </Dropdown>
        </div>
      </div>

      <div className="jfe-mobiletabs">
        <Tabs activeKey={mobile} onSelect={(_e, k) => setMobile(k as 'results' | 'chat')} isFilled aria-label="View">
          <Tab eventKey="results" title={<TabTitleText>Results</TabTitleText>} />
          <Tab eventKey="chat" title={<TabTitleText>Conversation</TabTitleText>} />
        </Tabs>
      </div>

      <Drawer isExpanded={drawer} className="jfe-drawer">
        <DrawerContent panelContent={
          <DrawerPanelContent widths={{ default: 'width_100', lg: 'width_75', xl: 'width_66' }} focusTrap={{ enabled: true }} onKeyDown={(e) => { if (e.key === 'Escape') setDrawer(false); }}>
            <DrawerHead>
              <h2 className="jfe-drawer__title" tabIndex={-1}>Sources, evidence and assumptions</h2>
              <DrawerActions><DrawerCloseButton onClick={() => setDrawer(false)} /></DrawerActions>
            </DrawerHead>
            <DrawerPanelBody>{drawer && <SourcesPanel scenario={result.scenario} scenarioLabel={title} />}</DrawerPanelBody>
          </DrawerPanelContent>
        }>
          <DrawerContentBody className="jfe-body">
            <aside className="jfe-left" aria-label="Conversation">
              <Conversation catalog={catalog} hasSession={!!session} items={items} setItems={setItems} viewedRunId={runId}
                viewedScenario={isBaseline ? null : result.scenario} runs={runs} onRunAccepted={(v, after) => { if (after) pendingCharts.current[v.id] = after; setRuns((m) => ({ ...m, [v.id]: v })); poll(v.id); }}
                getCurrentChart={() => active.current?.spec ?? null} onChart={showChart} onDownload={performDownload}
                onCancel={cancel} onView={viewRun} onShowAssumptions={() => setDrawer(true)} onSessionStarted={loadSession} />
            </aside>
            <main className={`jfe-canvas ${isProcess(result) ? 'is-proc' : ''}`} id="jfe-results" tabIndex={-1}>
              {notice && <Alert variant="warning" isInline isPlain title={notice} actionClose={<Button variant="plain" onClick={() => setNotice('')} aria-label="Dismiss">×</Button>} className="jfe-notice-alert" />}
              <div className="jfe-canvas__head">
                <div>
                  <div className="jfe-eyebrow">{isBaseline ? 'Baseline' : 'Scenario'}</div>
                  <h2 className="jfe-canvas__title">{title}</h2>
                </div>
              </div>
              <ViewBoundary resetKey={`${runId ?? 'baseline'}-${String(activeTab)}`}>
              <MetricCards r={result} isBaseline={isBaseline} color={color} />
              <ResultNotes r={result} />
              <Tabs activeKey={activeTab} onSelect={(_e, k) => {
                setTab(k);
                if (k === 'valid' && !backtest) api<Backtest>('/api/backtest').then((b) => { setBacktest(b); setBacktestError(''); }, (e) => setBacktestError(errText(e)));
              }} aria-label="Result views" className="jfe-tabs" mountOnEnter unmountOnExit>
                <Tab eventKey="pyramid" title={<TabTitleText>Pyramid</TabTitleText>}><Pyramid {...plot} /></Tab>
                <Tab eventKey="traj" title={<TabTitleText>Trajectories</TabTitleText>}><Trajectories {...plot} /></Tab>
                {!isBaseline && <Tab eventKey="diff" title={<TabTitleText>Difference</TabTitleText>}><Difference {...plot} /></Tab>}
                {isProcess(result) && <Tab eventKey="fan" title={<TabTitleText>Fan</TabTitleText>}><Fan {...plot} /></Tab>}
                <Tab eventKey="care" title={<TabTitleText>Care</TabTitleText>}><Care {...plot} /></Tab>
                <Tab eventKey="schools" title={<TabTitleText>Schools</TabTitleText>}><Schools {...plot} /></Tab>
                <Tab eventKey="health" title={<TabTitleText>Health</TabTitleText>}><Health {...plot} runColors={runColors} /></Tab>
                {hasHousing && <Tab eventKey="housing" title={<TabTitleText>Housing</TabTitleText>}><Housing {...plot} runColors={runColors} /></Tab>}
                {isBaseline && <Tab eventKey="valid" title={<TabTitleText>Validation</TabTitleText>}><Validation present={present} data={backtest} error={backtestError} tkey={tkey} onActive={onActive} /></Tab>}
                {analysis && <Tab eventKey="analysis" title={<TabTitleText>Analysis</TabTitleText>}><ChartFromData key={JSON.stringify(analysis.spec)} data={analysis} color={color} present={present} tkey={tkey} onActive={onActive} runColors={runColors} /></Tab>}
              </Tabs>
              </ViewBoundary>
            </main>
          </DrawerContentBody>
        </DrawerContent>
      </Drawer>

      <Receipt r={result} isBaseline={isBaseline} />

      {modal === 'save' && runId && <SaveModal runId={runId} title={title} onClose={() => setModal(null)} />}
      {modal === 'saved' && <SavedModal onClose={() => setModal(null)} onView={viewRun} />}
    </div>
  );
}
