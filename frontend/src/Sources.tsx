// Sources drawer: datasets, documents, data-quality checks, model card and the current scenario's assumptions.
import { useEffect, useState } from 'react';
import { Alert, Label, Spinner, Title } from '@patternfly/react-core';
import { Table, Tbody, Td, Th, Thead, Tr } from '@patternfly/react-table';
import CheckIcon from '@patternfly/react-icons/dist/esm/icons/check-circle-icon';
import TimesIcon from '@patternfly/react-icons/dist/esm/icons/times-circle-icon';
import { LIMITATIONS, api, describe, errText, type Backtest, type Scenario } from './lib';

interface Src { id: string; title: string; url?: string; vintage?: string; licence?: string; sha256?: string; use?: string }
interface Sources {
  pack_id: string; built: string; gate: string; sources: Src[]; documents: Src[];
  quality: {
    checks: ({ check: string; pass: boolean } & Record<string, unknown>)[];
    fertility?: { implied_tfr_2034_MD: number; published_tfr_2034_MD: number; note?: string; shape?: string };
  };
}

const PROV_COLOR: Record<string, 'blue' | 'purple' | 'grey'> = { published: 'grey', derived: 'purple', 'user assumption': 'blue' };

function SrcTable({ rows, caption }: { rows: Src[]; caption: string }) {
  return (
    <Table aria-label={caption} variant="compact" className="jfe-src-table">
      <Thead><Tr><Th>ID</Th><Th>Title</Th><Th>Vintage</Th><Th>Licence</Th><Th>SHA-256</Th><Th>Use</Th></Tr></Thead>
      <Tbody>
        {rows.map((s) => (
          <Tr key={s.id}>
            <Td dataLabel="ID">{s.id}</Td>
            <Td dataLabel="Title">{s.url ? <a href={s.url} target="_blank" rel="noreferrer noopener">{s.title}</a> : s.title}</Td>
            <Td dataLabel="Vintage">{s.vintage ?? '–'}</Td>
            <Td dataLabel="Licence">{s.licence ?? '–'}</Td>
            <Td dataLabel="SHA-256"><code title={s.sha256}>{s.sha256 ? s.sha256.slice(0, 12) : '–'}</code></Td>
            <Td dataLabel="Use">{s.use ?? ''}</Td>
          </Tr>
        ))}
      </Tbody>
    </Table>
  );
}

export function SourcesPanel({ scenario, scenarioLabel }: { scenario: Scenario; scenarioLabel: string }) {
  const [data, setData] = useState<Sources | null>(null);
  const [error, setError] = useState('');
  const [bt, setBt] = useState<Backtest | null>(null);
  useEffect(() => {
    api<Sources>('/api/sources').then(setData, (e) => setError(errText(e)));
    api<Backtest>('/api/backtest').then(setBt, () => undefined); // optional: method notes for the Validation view
  }, []);
  const f = data?.quality.fertility;
  return (
    <div className="jfe-sources">
      <section>
        <Title headingLevel="h3" size="lg">Assumptions: {scenarioLabel}</Title>
        <Table aria-label="Current scenario assumptions" variant="compact">
          <Thead><Tr><Th>Field</Th><Th>Value</Th><Th>Provenance</Th></Tr></Thead>
          <Tbody>
            {describe(scenario).map((r) => (
              <Tr key={r.field}><Td dataLabel="Field">{r.field}</Td><Td dataLabel="Value">{r.value}</Td>
                <Td dataLabel="Provenance" className="jfe-nowrap"><Label isCompact color={PROV_COLOR[r.provenance] ?? 'grey'}>{r.provenance}</Label></Td></Tr>
            ))}
          </Tbody>
        </Table>
      </section>

      <section>
        <Title headingLevel="h3" size="lg">How the model works</Title>
        <div className="jfe-prose">
          <p>A reduced age/sex cohort-component model. Each simulated year follows the official methodology: everyone ages one year, deaths are applied, births are added, then net migration is added with a fixed age and sex profile. Single years of age 0 to 99 plus an open 100+ group, for females and males.</p>
          <p>Death probabilities and the migration age/sex profile are derived from the official Statistics Jersey projection outputs (Feb 2026). The age pattern of fertility comes from published Jersey rates for 2023-25, scaled each year to the official births for the chosen fertility assumption.</p>
          <p><b>Not modelled:</b> gross immigration and emigration, residential status, household formation, the labour market, housing, prices, health outcomes, and any feedback from policy to behaviour. The 16-64 group is an age convention, not the workforce. The care index counts open benefit claims at unchanged 2024 rates; it is not care need or staffing.</p>
          <p>Each simulation draws births and deaths at random and, by default, varies annual net migration around the assumed path with the persistence and spread of Jersey's 2001-2025 record (AR(1)). Scenario and baseline share the same random paths, so their difference reflects the assumptions only. Fertility and mortality rates are not varied.</p>
        </div>
      </section>

      <section>
        <Title headingLevel="h3" size="lg">Limitations</Title>
        <ul className="jfe-bullets">{LIMITATIONS.map((l) => <li key={l}>{l}</li>)}</ul>
      </section>

      {bt && (
        <section>
          <Title headingLevel="h3" size="lg">Validation method</Title>
          <ul className="jfe-bullets">
            <li>{bt.retrospective.description}</li>
            {bt.official_projection_one_year_check && <li>{bt.official_projection_one_year_check.note}</li>}
            {bt.reference_emulation && <li>{bt.reference_emulation.note}</li>}
          </ul>
        </section>
      )}

      {error && <Alert variant="warning" isInline title={error} />}
      {!data && !error && <Spinner size="lg" aria-label="Loading sources" />}
      {data && (
        <>
          <section>
            <Title headingLevel="h3" size="lg">Data-quality checks</Title>
            <p className="jfe-muted">Pack {data.pack_id}, built {data.built}, gate {data.gate}.</p>
            <ul className="jfe-checks">
              {data.quality.checks.map((c) => (
                <li key={c.check}>{c.pass ? <CheckIcon className="jfe-pass" aria-label="pass" /> : <TimesIcon className="jfe-fail" aria-label="fail" />} {c.check}</li>
              ))}
            </ul>
            {f && (
              <div className="jfe-fert">
                <b>Fertility consistency check.</b> Implied total fertility rate in 2034 (mid-range): <span className="jfe-num">{f.implied_tfr_2034_MD.toFixed(3)}</span>, against the published <span className="jfe-num">{f.published_tfr_2034_MD.toFixed(2)}</span>. {f.note}
              </div>
            )}
          </section>
          <section>
            <Title headingLevel="h3" size="lg">Datasets</Title>
            <SrcTable rows={data.sources} caption="Datasets" />
          </section>
          {!!data.documents.length && (
            <section>
              <Title headingLevel="h3" size="lg">Documents</Title>
              <SrcTable rows={data.documents} caption="Documents" />
            </section>
          )}
        </>
      )}
    </div>
  );
}
