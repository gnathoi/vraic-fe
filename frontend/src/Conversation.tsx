// Left column: PatternFly ChatBot. Prompts become reviewable drafts; nothing runs until the user confirms a specific version.
import { useEffect, useMemo, useRef, useState } from 'react';
import Chatbot, { ChatbotDisplayMode } from '@patternfly/chatbot/dist/dynamic/Chatbot';
import ChatbotContent from '@patternfly/chatbot/dist/dynamic/ChatbotContent';
import ChatbotWelcomePrompt from '@patternfly/chatbot/dist/dynamic/ChatbotWelcomePrompt';
import { ChatbotFooter } from '@patternfly/chatbot/dist/dynamic/ChatbotFooter';
import MessageBox, { type MessageBoxHandle } from '@patternfly/chatbot/dist/dynamic/MessageBox';
import Message from '@patternfly/chatbot/dist/dynamic/Message';
import { MessageBar } from '@patternfly/chatbot/dist/dynamic/MessageBar';
import { Alert, Button, ExpandableSection, Form, FormGroup, Label, Spinner, TextInput } from '@patternfly/react-core';
import { ScenarioForm } from './ScenarioForm';
import { ApiError, TERMINAL, api, errText, type Catalog, type ChartData, type ChartSpec, type Draft, type RunView, type Scenario } from './lib';

type Item =
  | { id: string; ts: string; kind: 'user'; text: string }
  | { id: string; ts: string; kind: 'draft'; draft: Draft; runId?: string; idem?: string; note?: string }
  | { id: string; ts: string; kind: 'run'; runId: string }
  | { id: string; ts: string; kind: 'error'; text: string };

const now = () => new Date().toISOString();
const uid = () => crypto.randomUUID();
const hhmm = (iso: string) => new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

export interface ConversationProps {
  catalog: Catalog;
  hasSession: boolean;
  items: Item[];
  setItems: (f: (items: Item[]) => Item[]) => void;
  viewedRunId: string | null;
  viewedScenario: Scenario | null;
  runs: Record<string, RunView>;
  onRunAccepted: (v: RunView, chartAfterRun?: ChartSpec) => void;
  getCurrentChart: () => ChartSpec | null;
  onChart: (c: ChartData, focus?: boolean) => void;
  onDownload: (d: NonNullable<Draft['download']>) => Promise<string>;
  onCancel: (id: string) => void;
  onView: (id: string) => void;
  onShowAssumptions: () => void;
  onSessionStarted: () => void;
}
export type { Item as ChatItem };

// ------------------------------------------------------------------ pieces
export function EventCode({ onDone }: { onDone: () => void }) {
  const [code, setCode] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  return (
    <Form className="jfe-code" onSubmit={async (e) => {
      e.preventDefault();
      setBusy(true); setError('');
      try { await api('/api/session', 'POST', { code: code.trim() }); onDone(); } catch (err) { setError(errText(err)); } finally { setBusy(false); }
    }}>
      <FormGroup label="Access code" fieldId="event-code">
        <div className="jfe-code__row">
          <TextInput id="event-code" autoFocus value={code} onChange={(_e, v) => setCode(v)} autoComplete="off" spellCheck={false} maxLength={64} />
          <Button type="submit" isDisabled={!code.trim() || busy} isLoading={busy}>Start session</Button>
        </div>
      </FormGroup>
      {error && <Alert variant="warning" isInline isPlain title={error} />}
    </Form>
  );
}

/** "Fixed annual net migration: 2026-2070: +400" → "2026-2070: +400" when both sides share the label. */
const trimShared = (s: string, other: string) => {
  const k = s.indexOf(': ');
  return k > 0 && other.startsWith(s.slice(0, k + 2)) ? s.slice(k + 2) : s;
};

function ReviewCard({ item, latest, canRun, onConfirm, onEdit, busy }: {
  item: Extract<Item, { kind: 'draft' }>; latest: boolean; canRun: boolean; busy: boolean; onConfirm: () => void; onEdit: () => void;
}) {
  const d = item.draft;
  const [open, setOpen] = useState(false);
  const changed = d.diff?.changed ?? [];
  const unchanged = d.diff?.unchanged ?? [];
  return (
    <div className="jfe-review">
      <div className="jfe-review__eyebrow">Draft experiment · version {d.version}</div>
      <div className="jfe-review__title">{d.scenario?.title}</div>
      {!!d.interpretation?.length && (
        <ul className="jfe-review__interp">{d.interpretation.map((t, i) => <li key={i}>{t}</li>)}</ul>
      )}
      <div className="jfe-review__label">Changed assumptions</div>
      {changed.length ? (
        <dl className="jfe-review__changes">
          {changed.map((c) => (
            <div key={c.field}>
              <dt>{c.field}</dt>
              <dd><span className="jfe-from">{trimShared(c.from, c.to)}</span><span className="jfe-arrow" aria-label="changes to"> → </span><span className="jfe-to">{trimShared(c.to, c.from)}</span></dd>
            </div>
          ))}
        </dl>
      ) : <p className="jfe-muted">No assumptions differ from the parent scenario.</p>}
      <ExpandableSection toggleText={`Unchanged assumptions (${unchanged.length})`} isExpanded={open} onToggle={(_e, v) => setOpen(v)} isIndented>
        <dl className="jfe-review__changes">
          {unchanged.map((u) => <div key={u.field}><dt>{u.field} <Label isCompact variant="outline">{u.provenance}</Label></dt><dd>{u.value}</dd></div>)}
        </dl>
      </ExpandableSection>
      {d.warnings?.map((w) => <Alert key={w} variant="info" isInline isPlain title={w} className="jfe-review__warn" />)}
      {item.runId ? null : latest ? (
        <div className="jfe-review__actions">
          <Button onClick={onConfirm} isDisabled={!canRun || busy} isLoading={busy}>Confirm and run</Button>
          <Button variant="secondary" onClick={onEdit} isDisabled={busy}>Edit assumptions</Button>
        </div>
      ) : <p className="jfe-muted">Superseded by a newer version below.</p>}
    </div>
  );
}

function elapsed(iso: string, t: number) {
  const s = Math.max(0, Math.round((t - Date.parse(iso)) / 1000));
  return s < 90 ? `${s} s` : `${Math.floor(s / 60)} min ${s % 60} s`;
}

function RunStatus({ v, viewed, onCancel, onView }: { v?: RunView; viewed: boolean; onCancel: () => void; onView: () => void }) {
  const [t, setT] = useState(Date.now());
  const live = v && !TERMINAL.includes(v.status);
  useEffect(() => {
    if (!live) return;
    const h = setInterval(() => setT(Date.now()), 1000);
    return () => clearInterval(h);
  }, [live]);
  if (!v) return <div className="jfe-run"><Spinner size="sm" aria-label="Checking run" /> Checking run status…</div>;
  const lines: string[] = [];
  let head = '';
  switch (v.status) {
    case 'queued':
      head = [`Queued · position ${v.queue_position ?? '?'}`, v.estimated_wait_seconds != null ? `about ${Math.ceil(v.estimated_wait_seconds)} s` : ''].filter(Boolean).join(' · ');
      if (v.paused) lines.push(v.paused);
      break;
    case 'running': head = `Running · ${elapsed(v.created, t)}`; break;
    case 'cancel_requested': head = 'Cancelling…'; break;
    case 'numerical_complete': head = 'Complete'; break;
    case 'complete':
      head = ['Complete', v.cache_status].filter(Boolean).join(' · ');
      if (v.artifacts_status === 'failed') lines.push('Exports unavailable');
      break;
    case 'failed': head = 'Failed'; if (v.error) lines.push(v.error); break;
    case 'cancelled': head = 'Cancelled'; break;
    case 'expired': head = 'Expired'; if (v.error) lines.push(v.error); break;
  }
  return (
    <div className="jfe-run" aria-live="polite">
      <div className="jfe-run__title">{v.title}</div>
      <div className="jfe-run__head">{live && <Spinner size="sm" aria-hidden="true" />}<span>{head}</span></div>
      {lines.map((l) => <div key={l} className="jfe-run__line">{l}</div>)}
      <div className="jfe-run__actions">
        {(v.status === 'queued' || v.status === 'running') && <Button variant="secondary" size="sm" onClick={onCancel}>Cancel run</Button>}
        {v.status === 'complete' && !viewed && <Button variant="link" isInline onClick={onView}>View result</Button>}
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ main
const DRAFTING = 'Drafting';

export function Conversation(p: ConversationProps) {
  const [loadingWord, setLoadingWord] = useState(DRAFTING);
  const { items, setItems, catalog } = p;
  const [drafting, setDrafting] = useState(false);
  const [busyItem, setBusyItem] = useState<string | null>(null);
  const [form, setForm] = useState<{ mode: 'patch'; item: Extract<Item, { kind: 'draft' }> } | { mode: 'new' } | null>(null);
  const [announce, setAnnounce] = useState('');
  const pending = useRef(false);
  const box = useRef<MessageBoxHandle | null>(null);
  // Follow the conversation: new messages, drafts and run updates appear at the bottom.
  useEffect(() => { if (items.length || drafting) box.current?.scrollToBottom({ resumeSmartScroll: true }); }, [items.length, drafting]);

  const latestVersion = useMemo(() => {
    const m: Record<string, number> = {};
    items.forEach((i) => { if (i.kind === 'draft') m[i.draft.draft_id] = Math.max(m[i.draft.draft_id] ?? 0, i.draft.version); });
    return m;
  }, [items]);

  const push = (it: Item) => setItems((xs) => [...xs, it]);
  const pushDraft = (d: Draft) => {
    const id = uid();
    push({ id, ts: now(), kind: 'draft', draft: d });
    setAnnounce(`Assistant: ${d.message}`);
    if (d.status === 'show_assumptions') p.onShowAssumptions();
    // a solve keeps the review card in view on small screens; the chart is ready in the Analysis tab
    if ((d.status === 'chart' || d.status === 'unreachable' || d.solve) && d.chart) p.onChart(d.chart, !d.solve);
    if (d.status === 'download' && d.download) {
      p.onDownload(d.download).then(
        (note) => note && setItems((xs) => xs.map((x) => (x.id === id ? { ...x, note } : x))),
        (e) => pushError(e));
    }
  };
  const pushError = (e: unknown) => { push({ id: uid(), ts: now(), kind: 'error', text: errText(e) }); setAnnounce(errText(e)); };

  const send = async (raw: string | number) => {
    const text = String(raw).trim();
    if (!text || pending.current) return;
    pending.current = true;
    const history = items.flatMap((i) => (i.kind === 'user' ? [{ role: 'user', text: i.text }]
      : i.kind === 'draft' ? [{ role: 'assistant', text: [i.draft.message, i.draft.clarification].filter(Boolean).join(' ') }] : [])).slice(-6);
    const lastDraft = [...items].reverse().find((i) => i.kind === 'draft' && i.draft.scenario) as Extract<Item, { kind: 'draft' }> | undefined;
    push({ id: uid(), ts: now(), kind: 'user', text });
    setDrafting(true);
    const body = { prompt: text, parent_run_id: p.viewedRunId, parent_draft_id: lastDraft?.draft.draft_id ?? null, history, current_chart: p.getCurrentChart() };
    try {
      for (let attempt = 0; ; attempt++) {
        try {
          pushDraft(await api<Draft>('/api/drafts', 'POST', body));
          break;
        } catch (e) {
          // a busy model queue answers 429 at once: wait as asked and retry, keeping the prompt
          if (!(e instanceof ApiError) || e.status !== 429 || attempt >= 3) throw e;
          for (let n = Math.max(1, e.retryAfter || 10); n > 0; n--) {
            setLoadingWord(`Busy · retrying in ${n} s`);
            await new Promise((r) => setTimeout(r, 1000));
          }
          setLoadingWord(DRAFTING);
        }
      }
    } catch (e) {
      pushError(e);
    } finally {
      setDrafting(false);
      setLoadingWord(DRAFTING);
      pending.current = false;
    }
  };

  const confirm = async (item: Extract<Item, { kind: 'draft' }>) => {
    if (busyItem) return;
    const idem = item.idem ?? uid(); // reused on retry so a lost response can never create a second run
    setItems((xs) => xs.map((x) => (x.id === item.id ? { ...x, idem } : x)));
    setBusyItem(item.id);
    try {
      const body = { draft_id: item.draft.draft_id, version: item.draft.version, idempotency_key: idem };
      let v: RunView | null = null;
      for (let attempt = 0; !v; attempt++) {  // busy queue or run limit: wait as told and retry (same idempotency key)
        try {
          v = await api<RunView>('/api/runs', 'POST', body);
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 429 && attempt < 3)) throw e;
          await new Promise((r) => setTimeout(r, Math.min(30, Math.max(2, e.retryAfter || 5)) * 1000));
        }
      }
      setItems((xs) => [...xs.map((x) => (x.id === item.id ? { ...x, runId: v.id } : x)), { id: uid(), ts: now(), kind: 'run', runId: v.id }]);
      p.onRunAccepted(v, item.draft.chart_after_run);
      setAnnounce(`Run submitted: ${v.title}`);
    } catch (e) {
      pushError(e);
    } finally {
      setBusyItem(null);
    }
  };

  const formInitial: Scenario | null = form?.mode === 'patch' ? form.item.draft.scenario! : form ? (p.viewedScenario ?? catalog.default_scenario) : null;

  // a different few examples for each visitor: shuffled, one per area, as many as the pane holds
  const examples = useMemo(() => {
    const mobile = window.innerWidth < 1000;  // narrower cards are taller
    const n = Math.max(1, Math.min(6, Math.floor((window.innerHeight - (mobile ? 500 : 400)) / (mobile ? 150 : 135))));
    const pool = [...catalog.examples].sort(() => Math.random() - 0.5);
    const picked = pool.filter((ex, i) => pool.findIndex((o) => o.area === ex.area) === i);
    return [...picked, ...pool.filter((ex) => !picked.includes(ex))].slice(0, n);
  }, [catalog.examples]);
  const welcome = (
    <ChatbotWelcomePrompt
      title="Ask a what-if question"
      description="Examples"
      prompts={examples.map((ex) => ({ title: ex.area, message: ex.text, onClick: () => send(ex.text) }))}
    />
  );

  return (
    <div className="jfe-chat">
      <Chatbot displayMode={ChatbotDisplayMode.embedded} ariaLabel="Scenario conversation">
        <ChatbotContent>
          <MessageBox ref={box} announcement={announce} ariaLabel="Conversation" enableSmartScroll>
            {items.length === 0 && p.hasSession && welcome}
            {items.map((it) => {
              if (it.kind === 'user') return <Message key={it.id} role="user" name="You" content={it.text} timestamp={hhmm(it.ts)} isMarkdownDisabled isAvatarHidden />;
              if (it.kind === 'error') {
                return <Message key={it.id} role="bot" name="vraic-fe" timestamp={hhmm(it.ts)} isLiveRegion={false}>
                  <Alert variant="warning" isInline isPlain title={it.text} />
                </Message>;
              }
              if (it.kind === 'run') {
                return <Message key={it.id} role="bot" name="vraic-fe" timestamp={hhmm(it.ts)} isLiveRegion={false}>
                  <RunStatus v={p.runs[it.runId]} viewed={p.viewedRunId === it.runId} onCancel={() => p.onCancel(it.runId)} onView={() => p.onView(it.runId)} />
                </Message>;
              }
              const d = it.draft;
              const extra = d.status === 'ready_for_review' ? (
                <ReviewCard item={it} latest={latestVersion[d.draft_id] === d.version} canRun={p.hasSession} busy={busyItem === it.id}
                  onConfirm={() => confirm(it)} onEdit={() => setForm({ mode: 'patch', item: it })} />
              ) : d.status === 'needs_clarification' ? (
                <p className="jfe-clarify">{d.clarification}</p>
              ) : d.status === 'explain' ? (
                d.explanation ? <div className="jfe-explain-inline">
                  <p>{d.explanation.sentences.join(' ')}</p>
                </div> : null
              ) : d.status === 'chart' || d.status === 'unreachable' ? (
                d.chart ? <Button variant="link" isInline onClick={() => p.onChart(d.chart!)}>Show chart</Button> : null
              ) : d.status === 'download' ? (
                <div className="jfe-gap-top">
                  {it.note && <p className="jfe-muted">{it.note}</p>}
                  {d.download?.target === 'report' && d.download.run_id && <Button component="a" variant="link" isInline href={`/api/runs/${d.download.run_id}/export/report.html`} target="_blank" rel="noopener">Open briefing</Button>}
                </div>
              ) : (
                <Button variant="link" isInline onClick={p.onShowAssumptions}>Open sources and assumptions</Button>
              );
              return <Message key={it.id} role="bot" name="vraic-fe" timestamp={hhmm(it.ts)} content={d.message} isMarkdownDisabled extraContent={{ afterMainContent: extra }} />;
            })}
            {drafting && <Message role="bot" name="vraic-fe" isLoading loadingWord={loadingWord} />}
          </MessageBox>
        </ChatbotContent>
        <ChatbotFooter>
          {p.hasSession ? (
            <>
              <div className="jfe-notice">Do not enter personal or confidential information.</div>
              <MessageBar onSendMessage={send} placeholder="Ask a what-if question…" alwayShowSendButton isSendButtonDisabled={drafting} hasAttachButton={false} hasMicrophoneButton={false} />
            </>
          ) : null}
        </ChatbotFooter>
      </Chatbot>
      {form && formInitial && (
        <ScenarioForm
          catalog={catalog}
          initial={form.mode === 'new' ? { ...formInitial, title: p.viewedScenario ? `${formInitial.title} (edited)`.slice(0, 120) : formInitial.title } : formInitial}
          heading={form.mode === 'patch' ? 'Edit assumptions' : 'Build a scenario'}
          onClose={() => setForm(null)}
          onSubmit={async (scenario) => {
            const d = form.mode === 'patch'
              ? await api<Draft>(`/api/drafts/${form.item.draft.draft_id}`, 'PATCH', { version: form.item.draft.version, scenario })
              : await api<Draft>('/api/drafts/form', 'POST', { scenario, parent_run_id: p.viewedRunId });
            setForm(null);
            pushDraft(d);
          }}
        />
      )}
    </div>
  );
}
