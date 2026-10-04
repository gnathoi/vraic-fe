// The assumptions form. Same Scenario contract as the chat path; the server is the only validator.
import { useEffect, useRef, useState } from 'react';
import {
  Alert, Button, Form, FormGroup, FormSelect, FormSelectOption, HelperText, HelperTextItem, Modal, ModalBody, ModalFooter, ModalHeader,
  Checkbox, Radio, TextInput,
} from '@patternfly/react-core';
import PlusIcon from '@patternfly/react-icons/dist/esm/icons/plus-circle-icon';
import TrashIcon from '@patternfly/react-icons/dist/esm/icons/trash-alt-icon';
import { errText, type Catalog, type Code, type Scenario } from './lib';

type Row = Record<string, string>;
const CODES: Code[] = ['L2', 'L1', 'MD', 'H1', 'H2'];

function Rows({ label, cols, rows, setRows, addRow, min }: {
  label: string; cols: [string, string][]; rows: Row[]; setRows: (r: Row[]) => void; addRow: () => Row; min: number;
}) {
  return (
    <div className="jfe-rows" role="group" aria-label={label}>
      <div className="jfe-rows__head" aria-hidden="true">{cols.map(([, l]) => <span key={l}>{l}</span>)}<span /></div>
      {rows.map((row, i) => (
        <div className="jfe-rows__row" key={i}>
          {cols.map(([k, l]) => (
            <TextInput key={k} type="number" aria-label={`${label} row ${i + 1}: ${l}`} value={row[k]}
              onChange={(_e, v) => setRows(rows.map((r, j) => (j === i ? { ...r, [k]: v } : r)))} />
          ))}
          <Button variant="plain" aria-label={`Remove ${label} row ${i + 1}`} icon={<TrashIcon />} isDisabled={rows.length <= min}
            onClick={() => setRows(rows.filter((_, j) => j !== i))} />
        </div>
      ))}
      <Button variant="link" icon={<PlusIcon />} isInline onClick={() => setRows([...rows, addRow()])} isDisabled={rows.length >= (min ? 8 : 6)}>Add row</Button>
    </div>
  );
}

const toRows = (pts: { year: number; value: number }[]) => pts.map((p) => ({ year: String(p.year), value: String(p.value) }));

export function ScenarioForm({ initial, catalog, heading, onSubmit, onClose }: {
  initial: Scenario; catalog: Catalog; heading: string; onSubmit: (s: Scenario) => Promise<void>; onClose: () => void;
}) {
  const [title, setTitle] = useState(initial.title);
  const [end, setEnd] = useState(String(initial.end_year));
  const [segs, setSegs] = useState<Row[]>(initial.migration.segments.map((s) => ({ from: String(s.from_year), to: String(s.to_year), net: String(s.net_per_year) })));
  const [fCode, setFCode] = useState<Code>(initial.fertility.assumption);
  const [fPts, setFPts] = useState<Row[]>(toRows(initial.fertility.multiplier));
  const [mCode, setMCode] = useState<Code>(initial.mortality.assumption);
  const [mPts, setMPts] = useState<Row[]>(toRows(initial.mortality.multiplier));
  const [mode, setMode] = useState(initial.uncertainty.mode);
  const [draws, setDraws] = useState(String(initial.uncertainty.mode === 'process' ? initial.uncertainty.draws : catalog.default_draws));
  const [varies, setVaries] = useState(initial.uncertainty.migration !== 'fixed');
  const [ratesVary, setRatesVary] = useState(initial.uncertainty.rates !== 'fixed');
  const [baseline, setBaseline] = useState(initial.baseline_id);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const errRef = useRef<HTMLDivElement>(null);
  useEffect(() => { if (error) errRef.current?.scrollIntoView({ block: 'nearest' }); }, [error]);

  const changeEnd = (v: string) => {
    setEnd(v);
    const n = parseInt(v, 10);
    // keep the last migration segment aligned with the horizon; the user can still edit it
    if (n >= 2030 && n <= catalog.max_end_year && segs.length) setSegs(segs.map((s, i) => (i === segs.length - 1 ? { ...s, to: String(n) } : s)));
  };

  const submit = async () => {
    const num = (s: string) => (s.trim() === '' ? NaN : Number(s));
    const pts = (rows: Row[]) => rows.map((r) => ({ year: Math.round(num(r.year)), value: num(r.value) }));
    const sc: Scenario = {
      ...initial,
      title: title.trim() || 'Untitled scenario',
      baseline_id: baseline,
      end_year: Math.round(num(end)),
      migration: { kind: 'fixed_annual_net', segments: segs.map((s) => ({ from_year: Math.round(num(s.from)), to_year: Math.round(num(s.to)), net_per_year: Math.round(num(s.net)) })) },
      fertility: { assumption: fCode, multiplier: pts(fPts) },
      mortality: { assumption: mCode, multiplier: pts(mPts) },
      uncertainty: mode === 'process' ? { mode, draws: Math.round(num(draws)), migration: varies ? 'historical' : 'fixed', rates: ratesVary ? 'official_range' : 'fixed' } : { mode, draws: 1, migration: 'fixed', rates: 'fixed' },
    };
    const flat = [sc.end_year, sc.uncertainty.draws, ...sc.migration.segments.flatMap((s) => [s.from_year, s.to_year, s.net_per_year]), ...[...sc.fertility.multiplier, ...sc.mortality.multiplier].flatMap((p) => [p.year, p.value])];
    if (flat.some((v) => Number.isNaN(v))) { setError('Fill in every number, or remove empty rows.'); return; }
    setBusy(true); setError('');
    try { await onSubmit(sc); } catch (e) { setError(errText(e)); } finally { setBusy(false); }
  };

  const codeSelect = (id: string, value: Code, set: (c: Code) => void, kind: string) => (
    <FormSelect id={id} value={value} onChange={(_e, v) => set(v as Code)} aria-label={`${kind} assumption`}>
      {CODES.map((c) => <FormSelectOption key={c} value={c} label={`${c}: official ${catalog.codes[c]} ${kind}`} />)}
    </FormSelect>
  );
  const nextYear = (rows: Row[]) => String(Math.min(catalog.max_end_year, rows.length ? parseInt(rows[rows.length - 1].year || '2026', 10) + 4 : 2026));
  const [nlo, nhi] = catalog.net_limits, [olo, ohi] = catalog.official_net_range;

  return (
    <Modal isOpen variant="medium" onClose={onClose} aria-labelledby="sf-title" className="jfe-app-modal">
      <ModalHeader title={heading} labelId="sf-title" />
      <ModalBody>
        <div ref={errRef} aria-live="polite">{error && <Alert variant="warning" isInline title="The scenario was not accepted" className="jfe-form-error">{error}</Alert>}</div>
        <Form onSubmit={(e) => { e.preventDefault(); submit(); }} id="sf-form">
          <FormGroup label="Title" fieldId="sf-t" isRequired>
            <TextInput id="sf-t" value={title} maxLength={120} onChange={(_e, v) => setTitle(v)} />
          </FormGroup>
          <FormGroup label="Starting population" fieldId="sf-base">
            <FormSelect id="sf-base" value={baseline} onChange={(_e, v) => setBaseline(v as Scenario['baseline_id'])}>
              <FormSelectOption value="observed_end2025" label="Observed end-2025 (Statistics Jersey, Sept 2026, provisional)" />
              <FormSelectOption value="official_state_2025" label="Reproduce official projection: official projected end-2025 state (Feb 2026)" />
            </FormSelect>
            <HelperText><HelperTextItem>Official state: deterministic runs only.</HelperTextItem></HelperText>
          </FormGroup>
          <FormGroup label="End year" fieldId="sf-end" isRequired>
            <TextInput id="sf-end" type="number" value={end} onChange={(_e, v) => changeEnd(v)} className="jfe-narrow" />
            <HelperText><HelperTextItem>2030–{catalog.max_end_year}</HelperTextItem></HelperText>
          </FormGroup>
          <FormGroup label="Net migration (people per year)" fieldId="sf-mig" role="group">
            <Rows label="Migration segment" cols={[['from', 'From year'], ['to', 'To year'], ['net', 'Net per year']]} rows={segs} setRows={setSegs} min={1}
              addRow={() => {
                const last = segs[segs.length - 1];
                const from = last ? String(parseInt(last.to, 10) + 1) : String(catalog.first_year);
                return { from, to: end, net: last?.net ?? '400' };
              }} />
            <HelperText><HelperTextItem>Contiguous segments, {catalog.first_year} to end year. Range {nlo} to +{nhi}; published range {olo} to +{ohi}.</HelperTextItem></HelperText>
          </FormGroup>
          <FormGroup label="Fertility" fieldId="sf-f">
            {codeSelect('sf-f', fCode, setFCode, 'fertility')}
            <Rows label="Fertility multiplier point" cols={[['year', 'Year'], ['value', 'Multiplier']]} rows={fPts} setRows={setFPts} min={0} addRow={() => ({ year: nextYear(fPts), value: '1.0' })} />
            <HelperText><HelperTextItem>Optional rate multiplier, {catalog.fertility_multiplier[0]}–{catalog.fertility_multiplier[1]}; linear between points.</HelperTextItem></HelperText>
          </FormGroup>
          <FormGroup label="Life expectancy" fieldId="sf-m">
            {codeSelect('sf-m', mCode, setMCode, 'life expectancy')}
            <Rows label="Death-rate multiplier point" cols={[['year', 'Year'], ['value', 'Multiplier']]} rows={mPts} setRows={setMPts} min={0} addRow={() => ({ year: nextYear(mPts), value: '1.0' })} />
            <HelperText><HelperTextItem>Optional death-probability multiplier, {catalog.mortality_multiplier[0]}–{catalog.mortality_multiplier[1]}; linear between points.</HelperTextItem></HelperText>
          </FormGroup>
          <FormGroup label="Simulations" fieldId="sf-u" role="radiogroup">
            <Radio id="sf-u-p" name="sf-u" label="Ensemble" isChecked={mode === 'process'} onChange={() => setMode('process')} />
            {mode === 'process' && <TextInput id="sf-draws" aria-label="Number of simulations" type="number" min={16} max={catalog.public_max_draws} value={draws} onChange={(_e, v) => setDraws(v)} className="jfe-draws" />}
            {mode === 'process' && <Checkbox id="sf-mig-var" label="Net migration varies as in 2001–2025" isChecked={varies} onChange={(_e, v) => setVaries(v)} />}
            {mode === 'process' && <Checkbox id="sf-rate-var" label="Fertility and mortality levels vary" isChecked={ratesVary} onChange={(_e, v) => setRatesVary(v)} />}
            <Radio id="sf-u-d" name="sf-u" label="Single deterministic run" isChecked={mode === 'deterministic'} onChange={() => setMode('deterministic')} />
            <HelperText><HelperTextItem>Default {catalog.default_draws.toLocaleString('en-GB')}; up to {catalog.public_max_draws.toLocaleString('en-GB')}.</HelperTextItem></HelperText>
          </FormGroup>
        </Form>
      </ModalBody>
      <ModalFooter>
        <Button type="submit" form="sf-form" isLoading={busy} isDisabled={busy}>Review draft</Button>
        <Button variant="link" onClick={onClose}>Cancel</Button>
      </ModalFooter>
    </Modal>
  );
}
