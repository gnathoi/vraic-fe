// Analytical plots. One ECharts hook, one frame (title, table toggle, one-line caption), one spec per view.
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, CustomChart, LineChart, ScatterChart } from 'echarts/charts';
import { GridComponent, LegendComponent, MarkLineComponent, MarkPointComponent, TitleComponent, TooltipComponent, VisualMapContinuousComponent } from 'echarts/components';
import { SVGRenderer } from 'echarts/renderers';
import { Alert, Button, FormSelect, FormSelectOption, Slider, ToggleGroup, ToggleGroupItem } from '@patternfly/react-core';
import { Table, Tbody, Td, Th, Thead, Tr } from '@patternfly/react-table';
import PlayIcon from '@patternfly/react-icons/dist/esm/icons/play-icon';
import PauseIcon from '@patternfly/react-icons/dist/esm/icons/pause-icon';
import TableIcon from '@patternfly/react-icons/dist/esm/icons/table-icon';
import DownloadIcon from '@patternfly/react-icons/dist/esm/icons/download-icon';
import {
  BANDS, COLORS, api, bandLabel, byBand, central, download, errText, fmt, isProcess, officialShort, prefersReducedMotion, saveBlob, signed,
  type Backtest, type ChartData, type ChartSpec, type Result,
} from './lib';

echarts.use([BarChart, CustomChart, LineChart, ScatterChart, GridComponent, TooltipComponent, LegendComponent, MarkLineComponent, MarkPointComponent, VisualMapContinuousComponent, TitleComponent, SVGRenderer]);


/** Chart colours for the current appearance. Text, grid and surfaces come from PatternFly tokens; data colours are fixed
 *  per light/dark so their meaning (scenario / comparator / official / observed) never changes with the theme.
 *  ponytail: module-level, refreshed by App via setChartTheme before plots render; plots take `tkey` so options rebuild. */
export const T = {
  ink: '#151515', ink2: '#4d5258', grid: '#e0e0e0', axis: '#8a8d90', faint: '#b8bbbe', surface: '#ffffff', border: '#d2d2d2',
  comparator: COLORS.comparator, official: COLORS.official, observed: COLORS.observed, model: '#0066cc',
  // a concrete family (never var()): ECharts measures label widths on a canvas, which cannot resolve CSS variables
  font: '"Red Hat Text", RedHatText, "Helvetica Neue", Arial, sans-serif',
};
export function setChartTheme(dark: boolean) {
  const cs = getComputedStyle(document.documentElement);
  const tok = (name: string, fallback: string) => cs.getPropertyValue(`--pf-t--global--${name}`).trim() || fallback;
  Object.assign(T, {
    ink: tok('text--color--regular', dark ? '#ffffff' : '#151515'),
    ink2: tok('text--color--subtle', dark ? '#c7c7c7' : '#4d5258'),
    grid: tok('border--color--subtle', dark ? '#383838' : '#e0e0e0'),
    axis: tok('border--color--default', dark ? '#707070' : '#8a8d90'),
    faint: tok('border--color--default', dark ? '#707070' : '#b8bbbe'),
    surface: tok('background--color--floating--default', dark ? '#292929' : '#ffffff'),
    border: tok('border--color--default', dark ? '#707070' : '#d2d2d2'),
    comparator: dark ? '#a3a3a3' : COLORS.comparator,
    official: dark ? '#b6a6e9' : COLORS.official,
    observed: dark ? '#f2f2f2' : COLORS.observed,
    model: dark ? '#4394e5' : '#0066cc',
    font: tok('font--family--body', T.font),
  });
}

export interface Other { id: string; title: string; color: string; result: Result }
export type MetricDefs = Record<string, { label: string; unit: string }>;
export interface PlotProps {
  r: Result; color: string; present: boolean; isBaseline: boolean; others?: Other[]; runLabel: string; tkey: string; metrics: MetricDefs;
  /** 'baseline' or the viewed run id, for server chart specs. */
  runId: string; onActive?: (a: ActiveChart) => void;
}

function useEChart(option: echarts.EChartsCoreOption) {
  const ref = useRef<HTMLDivElement>(null);
  const chart = useRef<echarts.ECharts | null>(null);
  const latest = useRef(option);
  const shown = useRef(false);
  const size = useRef('');  // the size the chart was last laid out at
  latest.current = option;
  const apply = (c: echarts.ECharts) => {
    try {
      c.setOption(latest.current, { replaceMerge: ['series', 'xAxis', 'yAxis', 'grid', 'title'] });
    } catch (e) {
      console.error('chart option rejected', e); // keep the workspace alive; the table view still shows the numbers
    }
  };
  useEffect(() => {
    const el = ref.current!;
    const c = echarts.init(el, undefined, { renderer: 'svg' });
    chart.current = c;
    // a freshly opened tab can lay out at zero size: draw once it has a size. resize() only on a real size change,
    // because it cuts short the entry animation (the observer always fires once on observe)
    const ro = new ResizeObserver(() => {
      const now = `${el.clientWidth}x${el.clientHeight}`;
      if (!el.clientWidth || now === size.current) return;
      const first = !shown.current;
      size.current = now;
      c.resize();
      if (first) { shown.current = true; apply(c); }
    });
    ro.observe(el);
    return () => { ro.disconnect(); c.dispose(); chart.current = null; shown.current = false; };
  }, []);
  useEffect(() => {
    const c = chart.current, el = ref.current;
    if (c && el?.clientWidth) {
      if (!shown.current) { c.resize(); size.current = `${el.clientWidth}x${el.clientHeight}`; }  // real size before the first draw
      shown.current = true;
      apply(c);
    }
  }, [option]);
  return { ref, chart };
}

const base = (present: boolean) => {
  const fs = present ? 15 : 12;
  return {
    fs,
    opt: {
      animation: !prefersReducedMotion(),
      animationDuration: 400,
      animationDurationUpdate: 300,
      textStyle: { fontFamily: T.font, fontSize: fs, color: T.ink2 },
      legend: { top: 0, left: 0, itemWidth: 24, itemHeight: 10, itemGap: 16, itemStyle: { opacity: 0 }, // line legends show the stroke (incl. dashes) without a marker
        textStyle: { fontSize: fs, color: T.ink2 } },
      tooltip: {
        confine: true, backgroundColor: T.surface, borderColor: T.border, borderWidth: 1, padding: [8, 10],
        textStyle: { color: T.ink, fontSize: fs, fontFamily: T.font }, extraCssText: 'box-shadow:var(--pf-t--global--box-shadow--md);font-variant-numeric:tabular-nums;',
      },
    },
  };
};
const valueAxis = (fs: number, extra: object = {}) => ({
  type: 'value', axisLine: { show: false }, axisTick: { show: false },
  splitLine: { lineStyle: { color: T.grid } }, axisLabel: { fontSize: fs, color: T.ink2 }, nameTextStyle: { fontSize: fs, color: T.ink2 }, ...extra,
});
const catAxis = (fs: number, data: (string | number)[], extra: object = {}) => ({
  type: 'category', data, boundaryGap: false, axisLine: { lineStyle: { color: T.axis } }, axisTick: { show: false },
  axisLabel: { fontSize: fs, color: T.ink2 }, ...extra,
});
const compact = (v: number) => (Math.abs(v) >= 1000 ? `${fmt(v / 1000, v % 1000 ? 1 : 0)}k` : fmt(v));
/** Symmetric, round axis limit and tick step for zero-centred charts. */
const niceSym = (m: number) => {
  const step = [5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 2500, 5000, 10000, 20000, 25000, 50000].find((st) => st * 4 >= m) ?? 100000;
  return { max: Math.max(step, Math.ceil(m / step) * step), step };
};

// ------------------------------------------------------------------ frame
function DataTable({ caption, head, rows }: { caption: string; head: string[]; rows: (string | number)[][] }) {
  return (
    <div className="jfe-table-wrap" tabIndex={0} role="region" aria-label={caption}>
      <Table aria-label={caption} variant="compact" isStickyHeader>
        <Thead><Tr>{head.map((h, i) => <Th key={i} modifier="wrap">{h}</Th>)}</Tr></Thead>
        <Tbody>
          {rows.map((row, i) => (
            <Tr key={i}>{row.map((c, j) => <Td key={j} dataLabel={head[j]} className={j ? 'jfe-num' : undefined}>{c}</Td>)}</Tr>
          ))}
        </Tbody>
      </Table>
    </div>
  );
}

/** What the chatbot and the download actions need from the chart currently on screen. */
export interface ActiveChart { spec: ChartSpec | null; title: string; isTable: () => boolean; csv: () => Promise<string>; image: () => string }

const csvCell = (v: unknown) => {
  let t = v == null ? '' : String(v);
  if (/^[=+\-@\t\r]/.test(t) && Number.isNaN(Number(t))) t = `'${t}`; // spreadsheet formula injection guard
  return /[",\n]/.test(t) ? `"${t.replace(/"/g, '""')}"` : t;
};
const fileStem = (title: string) => `vraic-fe-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 60)}`;

function Frame(props: {
  id: string; title: string; controls?: ReactNode; option: echarts.EChartsCoreOption;
  table: { head: string[]; rows: (string | number)[][] }; caption: ReactNode; footer?: ReactNode; empty?: string; overlay?: ReactNode;
  /** Server-equivalent spec: CSV downloads come from POST /api/chart-data so they match the server table exactly. */
  spec?: ChartSpec; csvFromServer?: boolean; initialTable?: boolean; footnote?: ReactNode; onActive?: (a: ActiveChart) => void; compactHeight?: boolean;
}) {
  const [showTable, setShowTable] = useState(!!props.initialTable);
  const [error, setError] = useState('');
  const { ref, chart } = useEChart(props.option);
  const latest = useRef({ props, showTable });
  latest.current = { props, showTable };

  const csv = async () => {
    const { props: p } = latest.current;
    if (p.spec && p.csvFromServer !== false) return download('/api/chart-data', { spec: { ...p.spec, view: 'chart' }, format: 'csv' }, `${fileStem(p.title)}.csv`);
    const name = `${fileStem(p.title)}.csv`;
    saveBlob(new Blob([[p.table.head, ...p.table.rows].map((r) => r.map(csvCell).join(',')).join('\n') + '\n'], { type: 'text/csv;charset=utf-8' }), name);
    return name;
  };
  const image = () => {
    const c = chart.current;
    if (!c || latest.current.showTable) throw new Error('Switch to the chart view to download the image.');
    const name = `${fileStem(latest.current.props.title)}.svg`;
    saveBlob(new Blob([c.renderToSVGString()], { type: 'image/svg+xml' }), name);
    return name;
  };
  useEffect(() => {
    props.onActive?.({ spec: props.spec ?? null, title: props.title, isTable: () => latest.current.showTable, csv, image });
  }, [JSON.stringify(props.spec), props.title]);

  const act = async (f: () => Promise<string> | string) => {
    setError('');
    try { await f(); } catch (e) { setError(errText(e)); }
  };
  return (
    <figure className={`jfe-plot jfe-plot--${props.id} ${props.compactHeight ? 'is-compact' : ''}`} aria-labelledby={`${props.id}-title`}>
      <div className="jfe-plot__head">
        <div className="jfe-plot__titles">
          <h3 id={`${props.id}-title`} className="jfe-plot__title">{props.title}</h3>
        </div>
        <div className="jfe-plot__controls">
          {props.controls}
          <div className="jfe-toolbar" role="toolbar" aria-label={`${props.title}: view and downloads`}>
            <Button variant="plain" size="sm" icon={<TableIcon />} aria-pressed={showTable} className={showTable ? 'is-on' : ''} onClick={() => setShowTable(!showTable)} isDisabled={!!props.empty}>Table</Button>
            <Button variant="plain" size="sm" icon={<DownloadIcon />} onClick={() => act(csv)} isDisabled={!!props.empty}>CSV</Button>
            <Button variant="plain" size="sm" icon={<DownloadIcon />} onClick={() => act(image)} isDisabled={!!props.empty || showTable}>SVG</Button>
          </div>
        </div>
      </div>
      {error && <Alert variant="warning" isInline isPlain title={error} className="jfe-plot__error" />}
      <div className="jfe-plot__body">
        {props.empty && <div className="jfe-plot__empty">{props.empty}</div>}
        <div ref={ref} className="jfe-chart" role="img" aria-label={props.title} hidden={showTable || !!props.empty} />
        {!showTable && !props.empty && props.overlay}
        {showTable && !props.empty && <DataTable caption={props.title} head={props.table.head} rows={props.table.rows} />}
      </div>
      {props.footer}
      <figcaption className="jfe-plot__caption">{props.caption}{props.footnote && <span className="jfe-plot__note"> · {props.footnote}</span>}</figcaption>
    </figure>
  );
}

/** The single caption line under a plot: aggregation, any required caveat, run and data pack. */
export const SOURCE = 'Source: Statistics Jersey population estimate (end-2025) and projections 2025-2080; vraic-fe cohort model';
const Caption = ({ items }: { r: Result; items: (string | false | undefined)[]; runLabel: string }) => (
  <>{[...items, SOURCE].filter(Boolean).join(' · ')}</>
);

// ------------------------------------------------------------------ P01 pyramid
export function Pyramid({ r, color, present, isBaseline, runLabel, tkey, runId, onActive }: PlotProps) {
  const years = r.years;
  const [yi, setYi] = useState(years.length - 1);
  const [playing, setPlaying] = useState(false);
  useEffect(() => { setYi(r.years.length - 1); setPlaying(false); }, [r]);
  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setYi((i) => {
      if (i >= years.length - 1) { setPlaying(false); return i; }
      return i + 1;
    }), 650);
    return () => clearInterval(t);
  }, [playing, years.length]);

  // Baseline view: the scenario is the official mid-range itself, so the outline shows the observed end-2025 structure instead.
  const refLabel = isBaseline ? 'End-2025 observed' : 'Baseline';
  const refGrid = (t: number) => (isBaseline ? r.age.scenario[0] : r.age.comparator[t]);
  const xmax = useMemo(() => {
    let m = 0;
    r.years.forEach((_, t) => [r.age.scenario[t], isBaseline ? r.age.scenario[0] : r.age.comparator[t]].forEach((g) => {
      m = Math.max(m, ...byBand(g, 0), ...byBand(g, 1));
    }));
    const step = [250, 500, 1000, 2000, 2500, 5000].find((st) => st * 5 >= m) ?? 10000;
    return { max: Math.ceil((m * 1.02) / step) * step, step };
  }, [r, isBaseline]);

  const t = Math.min(yi, years.length - 1);  // the state can briefly hold an index from a longer previous result
  const year = years[t];
  const sF = byBand(r.age.scenario[t], 0), sM = byBand(r.age.scenario[t], 1);
  const cF = byBand(refGrid(t), 0), cM = byBand(refGrid(t), 1);
  const labels = BANDS.map(bandLabel);
  const scenName = isBaseline ? 'Baseline' : 'Scenario';

  const option = useMemo(() => {
    const { fs, opt } = base(present);
    const bar = { type: 'bar', barCategoryGap: '14%', emphasis: { disabled: true } };
    const outline = { ...bar, barGap: '-100%', z: 3, itemStyle: { color: 'rgba(0,0,0,0)', borderColor: T.ink, borderWidth: 1 } };
    return {
      ...opt,
      legend: { ...opt.legend, data: [{ name: scenName, itemStyle: { color, opacity: 1 } }, { name: refLabel, itemStyle: { color: 'rgba(0,0,0,0)', borderColor: T.ink, borderWidth: 1, opacity: 1 } }], icon: 'rect', itemHeight: 10, left: 'center' },
      grid: { left: 52, right: 16, top: 30, bottom: 42 },
      tooltip: {
        ...opt.tooltip, trigger: 'item',
        formatter: (p: { seriesId: string; dataIndex: number }) => {
          const female = p.seriesId.endsWith('F');
          const s = (female ? sF : sM)[p.dataIndex], c = (female ? cF : cM)[p.dataIndex];
          return `<b>${labels[p.dataIndex]}</b> · ${female ? 'Female' : 'Male'} · end-${year}<br/>${scenName}: <b>${fmt(s)}</b><br/>${refLabel}: ${fmt(c)}<br/>Difference: ${signed(s - c)} people`;
        },
      },
      xAxis: valueAxis(fs, { name: 'People', nameLocation: 'middle', nameGap: 24, min: -xmax.max, max: xmax.max, interval: xmax.step, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => compact(Math.abs(v)) } }),
      yAxis: { type: 'category', data: labels, axisTick: { show: false }, axisLine: { lineStyle: { color: T.axis } }, axisLabel: { fontSize: present ? 13 : 11, color: T.ink2, interval: 0 } },
      series: [
        { ...bar, id: 'sF', name: scenName, stack: 's', data: sF.map((v) => -v), itemStyle: { color } },
        { ...bar, id: 'sM', name: scenName, stack: 's', data: sM, itemStyle: { color } },
        { ...outline, id: 'cF', name: refLabel, stack: 'c', data: cF.map((v) => -v) },
        { ...outline, id: 'cM', name: refLabel, stack: 'c', data: cM },
      ],
    };
  }, [present, tkey, color, xmax, year, isBaseline, r]);

  const controls = (
    <div className="jfe-year">
      <Button variant="plain" size="sm" aria-label={playing ? 'Pause animation' : 'Play years in sequence'} icon={playing ? <PauseIcon /> : <PlayIcon />}
        onClick={() => { if (!playing && yi >= years.length - 1) setYi(0); setPlaying(!playing); }} />
      <Slider className="jfe-year__slider" value={year} min={years[0]} max={years[years.length - 1]} step={1}
        isInputVisible inputValue={year} inputLabel="" inputAriaLabel="Year" thumbAriaLabel="Year" showBoundaries={false}
        onChange={(_e, v, input) => {
          const y = Math.round(input ?? v);
          if (y >= years[0] && y <= years[years.length - 1]) { setPlaying(false); setYi(years.indexOf(y)); }
        }} />
    </div>
  );
  return (
    <Frame id="p01" title={`Population by age and sex, end-${year}`} onActive={onActive}
      spec={{ kind: 'pyramid', years: isBaseline ? [years[0], year] : [year], run_id: runId }}
      controls={controls} option={option}
      table={{
        head: ['Age band', `Female ${scenName.toLowerCase()}`, `Female ${refLabel.toLowerCase()}`, `Male ${scenName.toLowerCase()}`, `Male ${refLabel.toLowerCase()}`],
        rows: labels.map((l, i) => [l, fmt(sF[i]), fmt(cF[i]), fmt(sM[i]), fmt(cM[i])]).reverse(),
      }}
      overlay={<div className="jfe-sexes" aria-hidden="true"><span>Female</span><span>Male</span></div>}
      caption={<Caption r={r} runLabel={runLabel} items={[]} />}
    />
  );
}

// ------------------------------------------------------------------ P02 trajectories
const FLOWS = ['births', 'deaths', 'net_migration'];
const unitDigits = (unit: string) => (unit === 'people' ? 0 : 1);

const along = (axis: number[], years: number[], vals: number[] | undefined) =>
  axis.map((y) => { const i = years.indexOf(y); return vals && i >= 0 ? vals[i] : null; });

export function Trajectories({ r, color, present, isBaseline, others = [], runLabel, tkey, metrics, runId, onActive }: PlotProps) {
  const choices = Object.entries(metrics).filter(([k]) => !FLOWS.includes(k) && r.metrics.scenario[k]);
  const [chosen, setMetric] = useState('population_total');
  const metric = r.metrics.scenario[chosen] ? chosen : 'population_total';
  const { label, unit } = metrics[metric] ?? { label: 'Total population', unit: 'people' };
  const digits = unitDigits(unit);
  const proc = isProcess(r);
  const end = Math.max(r.years[r.years.length - 1], ...others.map((o) => o.result.years[o.result.years.length - 1]));
  const axis = useMemo(() => Array.from({ length: end - r.observed.years[0] + 1 }, (_, i) => r.observed.years[0] + i), [r, end]);
  const S = r.metrics.scenario[metric], C = r.metrics.comparator[metric];
  const scenName = isBaseline ? 'Baseline' : 'Scenario';
  const changeYears = r.scenario.migration.segments.slice(1).map((s) => s.from_year);

  const option = useMemo(() => {
    const { fs, opt } = base(present);
    // Value x-axis with [year, value] pairs: no null padding, so labels and animations never see missing points.
    const pairs = (years: number[], vals: (number | null)[] | undefined) => (vals ? years.flatMap((y, i) => (vals[i] == null ? [] : [[y, vals[i]]])) : []);
    const line = { type: 'line', showSymbol: false, symbol: 'none', emphasis: { disabled: true } };
    const series: object[] = [];
    const legend: (string | object)[] = [];
    if (proc) {
      for (const [lo, hi, name, op] of [['p05', 'p95', '90% range', 0.13], ['p25', 'p75', '50% range', 0.22]] as const) {
        series.push({ ...line, id: `${lo}-base`, stack: lo, data: pairs(r.years, S[lo]), lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true });
        legend.push({ name, icon: 'rect', itemStyle: { color, opacity: op * 2, borderWidth: 0 } });
        series.push({ ...line, id: `${lo}-band`, name, stack: lo, data: r.years.map((y, i) => [y, S[hi][i] - S[lo][i]]), lineStyle: { opacity: 0 }, areaStyle: { color, opacity: op }, itemStyle: { color }, silent: true });
      }
    }
    if (r.observed.metrics[metric]) {
      series.push({ type: 'scatter', id: 'obs', name: 'Observed (to 2025)', data: pairs(r.observed.years, r.observed.metrics[metric]), symbolSize: present ? 8 : 7, itemStyle: { color: T.observed }, z: 5 });
    }
    series.push({ ...line, id: 'off', name: officialShort(r.official.label), data: pairs(r.official.years, r.official.metrics[metric]), lineStyle: { color: T.official, width: 1.5 }, itemStyle: { color: T.official } });
    if (!isBaseline) {
      series.push({ ...line, id: 'comp', name: 'Baseline', data: pairs(r.years, central(C)), lineStyle: { color: T.comparator, width: 1.75, type: [6, 4] }, itemStyle: { color: T.comparator } });
    }
    for (const o of others) {
      series.push({ ...line, id: `other-${o.id}`, name: o.title, data: pairs(o.result.years, central(o.result.metrics.scenario[metric])), lineStyle: { color: o.color, width: 1.5 }, itemStyle: { color: o.color } });
    }
    series.push({
      ...line, id: 'scen', name: scenName, data: pairs(r.years, central(S)), lineStyle: { color, width: 2.75 }, itemStyle: { color }, z: 6,
      endLabel: { show: true, formatter: (p: { value: number[] }) => fmt(p.value[1], digits), color: T.ink, fontWeight: 600, fontSize: fs },
      markLine: {
        silent: true, symbol: 'none', animation: false,
        label: { fontSize: fs - 1, color: T.ink2, formatter: '{b}' },
        data: [{ xAxis: 2025, name: 'Base 2025', lineStyle: { color: T.axis, type: 'solid', width: 1 } },
          ...changeYears.map((y) => ({ xAxis: y, name: '', label: { show: false }, lineStyle: { color: T.faint, type: [3, 3], width: 1 } }))],
      },
    });
    const bandText = (id: string, year: number) => {
      const i = r.years.indexOf(year);
      const [lo, hi] = id.startsWith('p05') ? ['p05', 'p95'] : ['p25', 'p75'];
      return i < 0 ? '–' : `${fmt(S[lo][i], digits)}–${fmt(S[hi][i], digits)}`;
    };
    type P = { seriesId: string; seriesName: string; marker: string; value: number[] };
    return {
      ...opt,
      legend: { ...opt.legend, data: [...series.map((x) => (x as { name?: string }).name).filter((n): n is string => !!n && !legend.some((l) => (l as { name: string }).name === n)).map((n) => (n.startsWith('Observed') ? { name: n, itemStyle: { opacity: 1 } } : n)), ...legend] },
      grid: { left: 8, right: 72, top: (present ? 60 : 52) + (proc || others.length ? 22 : 0), bottom: 24, containLabel: true },
      tooltip: {
        ...opt.tooltip, trigger: 'axis', axisPointer: { type: 'line', lineStyle: { color: T.axis } },
        formatter: (ps: P[]) => {
          const year = ps[0]?.value[0];
          const lines = ps.filter((p) => !p.seriesId.endsWith('-base')).map((p) =>
            `${p.marker}${p.seriesName}: <b>${p.seriesId.endsWith('-band') ? bandText(p.seriesId, year) : fmt(p.value[1], digits)}</b>`);
          return `<b>End-${year}</b><br/>${lines.join('<br/>')}`;
        },
      },
      xAxis: valueAxis(fs, { min: axis[0], max: axis[axis.length - 1], interval: 5, splitLine: { show: false }, axisLine: { show: true, lineStyle: { color: T.axis } }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => String(v) } }),
      yAxis: valueAxis(fs, { scale: true, name: unit, nameLocation: 'end', nameGap: 12, nameTextStyle: { align: 'left', fontSize: fs, color: T.ink2 }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => (digits ? fmt(v) : compact(v)) } }),
      series,
    };
  }, [present, tkey, color, metric, r, isBaseline, others, axis]);

  const head = ['Year', 'Observed', officialShort(r.official.label), ...(isBaseline ? [] : ['Baseline']), scenName, ...(proc ? ['5th pct', '95th pct'] : [])];
  const rows = axis.map((y) => {
    const ri = r.years.indexOf(y), oi = r.official.years.indexOf(y), bi = r.observed.years.indexOf(y);
    const obs = r.observed.metrics[metric];
    return [y, bi >= 0 && obs ? fmt(obs[bi], digits) : '', oi >= 0 ? fmt(r.official.metrics[metric][oi], digits) : '',
      ...(isBaseline ? [] : [ri >= 0 ? fmt(central(C)[ri], digits) : '']), ri >= 0 ? fmt(central(S)[ri], digits) : '',
      ...(proc ? [ri >= 0 ? fmt(S.p05[ri], digits) : '', ri >= 0 ? fmt(S.p95[ri], digits) : ''] : [])];
  });

  const controls = (
    <FormSelect aria-label="Metric" value={metric} onChange={(_e, v) => setMetric(v)} className="jfe-metric-select">
      {choices.map(([k, d]) => <FormSelectOption key={k} value={k} label={d.label} />)}
    </FormSelect>
  );
  return (
    <Frame id="p02" title={unit === 'people' ? `${label} (people)` : label} onActive={onActive}
      spec={{ kind: 'trajectory', metrics: [metric], series: isBaseline ? ['scenario', 'official', 'observed'] : ['scenario', 'comparator', 'official', 'observed'], run_id: runId, compare_run_ids: others.map((o) => o.id) }}
      controls={controls} option={option} table={{ head, rows }}
      caption={<Caption r={r} runLabel={runLabel} items={[proc ? `Pointwise 50%/90% ranges, ${fmt(r.receipt.completed_draws)} simulations; ${[r.scenario.uncertainty.migration === 'historical' ? 'net migration as in 2001-2025' : 'migration fixed', r.scenario.uncertainty.rates === 'official_range' ? 'fertility and mortality levels' : ''].filter(Boolean).join('; ')}` : 'Deterministic expected values']} />}
    />
  );
}

// ------------------------------------------------------------------ P03 difference
const DIFF_METRICS: [string, string, number[] | 'solid'][] = [
  ['population_total', 'Total', 'solid'], ['population_16_64', 'Aged 16-64', [8, 4]], ['population_65_plus', 'Aged 65+', [2, 3]],
];
export function Difference({ r, color, present, isBaseline, runLabel, tkey, runId, onActive }: PlotProps) {
  const D = r.metrics.difference;
  const option = useMemo(() => {
    const { fs, opt } = base(present);
    const m = niceSym(Math.max(10, ...DIFF_METRICS.flatMap(([k]) => central(D[k]).map(Math.abs))) * 1.05);
    return {
      ...opt,
      grid: { left: 8, right: 96, top: present ? 56 : 48, bottom: 24, containLabel: true },
      tooltip: { ...opt.tooltip, trigger: 'axis', valueFormatter: (v: number) => `${signed(v)} people` },
      xAxis: catAxis(fs, r.years.map(String)),
      yAxis: valueAxis(fs, { min: -m.max, max: m.max, interval: m.step, name: 'people', nameLocation: 'end', nameGap: 12, nameTextStyle: { align: 'left', fontSize: fs, color: T.ink2 }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => signed(v) } }),
      series: DIFF_METRICS.map(([k, name, type], i) => ({
        type: 'line', id: k, name, data: central(D[k]), showSymbol: false, symbol: 'none', emphasis: { disabled: true },
        lineStyle: { color, width: i ? 2 : 2.75, type }, itemStyle: { color },
        areaStyle: i ? undefined : { color, opacity: 0.1 },
        endLabel: { show: true, formatter: (p: { value: number }) => signed(p.value), color: T.ink, fontSize: fs },
        labelLayout: { moveOverlap: 'shiftY' },
        markLine: i ? undefined : { silent: true, symbol: 'none', label: { show: false }, data: [{ yAxis: 0 }], lineStyle: { color: T.ink, width: 1, type: 'solid' } },
      })),
    };
  }, [present, tkey, color, r]);
  return (
    <Frame id="p03" title="Scenario minus comparator (people)" option={option} onActive={onActive}
      spec={{ kind: 'difference', metrics: DIFF_METRICS.map(([k]) => k), run_id: runId }}
      empty={isBaseline ? 'No difference: this view is the comparator.' : undefined}
      table={{ head: ['Year', 'Total', 'Aged 16-64', 'Aged 65+'], rows: r.years.map((y, i) => [y, ...DIFF_METRICS.map(([k]) => signed(central(D[k])[i]))]) }}
      caption={<Caption r={r} runLabel={runLabel} items={['Scenario minus baseline']} />}
    />
  );
}

// ------------------------------------------------------------------ P04 care
const BAND_OPACITY = [1, 0.7, 0.45, 0.25];
export function Care({ r, color, present, isBaseline, runLabel, tkey, runId, onActive }: PlotProps) {
  const K = 'ltc_claims_index_65plus';
  const S = central(r.metrics.scenario[K]), C = central(r.metrics.comparator[K]);
  const contrib = r.ltc_contributions;
  const option = useMemo(() => {
    const { fs, opt } = base(present);
    const years = r.years.map(String);
    return {
      ...opt,
      title: [
        { text: 'Index (2025 = 100)', left: 0, top: 0, textStyle: { fontSize: fs, fontWeight: 600, color: T.ink } },
        { text: 'Open claims by age band (claims)', left: '54%', top: 0, textStyle: { fontSize: fs, fontWeight: 600, color: T.ink } },
      ],
      legend: [
        { ...opt.legend, top: 24, left: 0, data: [isBaseline ? 'Baseline' : 'Scenario', ...(isBaseline ? [] : ['Baseline']), officialShort(r.official.label)] },
        { ...opt.legend, top: 24, left: '54%', icon: 'rect', itemHeight: 10, itemWidth: 10, itemStyle: { opacity: 1 }, data: contrib.bands },
      ],
      grid: [
        { left: 8, width: '42%', top: present ? 70 : 60, bottom: 24, containLabel: true },
        { left: '54%', right: 8, top: present ? 70 : 60, bottom: 24, containLabel: true },
      ],
      tooltip: { ...opt.tooltip, trigger: 'axis' },
      xAxis: [catAxis(fs, years, { gridIndex: 0 }), catAxis(fs, years, { gridIndex: 1, boundaryGap: true })],
      yAxis: [
        valueAxis(fs, { gridIndex: 0, scale: true }),
        valueAxis(fs, { gridIndex: 1, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => compact(v) } }),
      ],
      series: [
        { type: 'line', id: 'scen', name: isBaseline ? 'Baseline' : 'Scenario', data: S, showSymbol: false, symbol: 'none', lineStyle: { color, width: 2.75 }, itemStyle: { color }, z: 5,
          endLabel: { show: true, formatter: (p: { value: number }) => fmt(p.value, 1), color: T.ink, fontWeight: 600, fontSize: fs }, tooltip: { valueFormatter: (v: number) => fmt(v, 1) } },
        ...(isBaseline ? [] : [{ type: 'line', id: 'comp', name: 'Baseline', data: C, showSymbol: false, symbol: 'none', lineStyle: { color: T.comparator, width: 1.75, type: [6, 4] }, itemStyle: { color: T.comparator }, tooltip: { valueFormatter: (v: number) => fmt(v, 1) } }]),
        { type: 'line', id: 'off', name: officialShort(r.official.label), data: along(r.years, r.official.years, r.official.metrics[K]), showSymbol: false, symbol: 'none', lineStyle: { color: T.official, width: 1.5 }, itemStyle: { color: T.official }, tooltip: { valueFormatter: (v: number | null) => fmt(v, 1) } },
        ...contrib.bands.map((b, i) => ({
          type: 'bar', id: `band-${b}`, name: b, stack: 'ltc', xAxisIndex: 1, yAxisIndex: 1, barCategoryGap: '25%',
          data: contrib.scenario.map((row) => row[i]), itemStyle: { color, opacity: BAND_OPACITY[i], borderColor: T.surface, borderWidth: 1 },
          emphasis: { disabled: true }, tooltip: { valueFormatter: (v: number) => `${fmt(v)} claims` },
        })),
      ],
    };
  }, [present, tkey, color, r, isBaseline]);
  const head = ['Year', 'Index (scenario)', ...(isBaseline ? [] : ['Index (comparator)']), ...contrib.bands.map((b) => `Claims ${b}`)];
  const rows = r.years.map((y, t) => [y, fmt(S[t], 1), ...(isBaseline ? [] : [fmt(C[t], 1)]), ...contrib.scenario[t].map((v) => fmt(v))]);
  return (
    <Frame id="p04" title="65+ long-term-care claim index (2025 = 100)" option={option} table={{ head, rows }} onActive={onActive}
      spec={{ kind: 'trajectory', metrics: ['ltc_claims_index_65plus'], run_id: runId }}
      caption={<Caption r={r} runLabel={runLabel} items={['Illustrative open-claim pressure at unchanged 2024 claim rates, not total care need or staffing demand']} />}
    />
  );
}

// ------------------------------------------------------------------ validation (retrospective backtest)
const VAL_METRICS: [string, string][] = [
  ['population_total', 'Total'], ['population_0_15', '0-15'], ['population_16_64', '16-64'], ['population_65_plus', '65+'],
  ['population_80_plus', '80+'], ['deaths', 'Deaths'],
];
const VAL_LABEL = Object.fromEntries(VAL_METRICS);

/** Round an axis end outward to a tidy step (a fifth of the order of magnitude of the span). */
const nice = (v: number, span: number, round: (x: number) => number) => {
  const step = Math.pow(10, Math.floor(Math.log10(span || 1))) / 5;
  return round(v / step) * step;
};

/** Retrospective variants by what they knew: one is fed the births and migration that happened, one assumes migration. */
const vName = (v: string) => (/observed/.test(v) ? 'Actual births and migration (hindsight)' : /400/.test(v) ? '+400 a year assumed (no hindsight)' : v);

export function Validation({ present, data, error, tkey, onActive }: { present: boolean; data: Backtest | null; error: string; tkey: string; onActive?: (a: ActiveChart) => void }) {
  const [metric, setMetric] = useState('population_total');
  const variants = data?.retrospective.variants ?? [];
  const years = variants[0]?.years.map((y) => y.year) ?? [];
  const available = VAL_METRICS.filter(([k]) => variants[0]?.years[0]?.[`model_${k}`] !== undefined);
  const unit = metric === 'deaths' ? 'deaths' : 'people';

  const option = useMemo(() => {
    const { fs, opt } = base(present);
    if (!variants.length) return opt;
    const obs = variants[0].years.map((y) => y[`observed_${metric}`] ?? null);
    return {
      ...opt,
      legend: { ...opt.legend, data: [...variants.map((v) => `Model: ${vName(v.variant)}`), { name: 'Observed (Statistics Jersey)', itemStyle: { opacity: 1 } }] },
      grid: { left: 8, right: 24, top: present ? 56 : 48, bottom: 24, containLabel: true },
      tooltip: { ...opt.tooltip, trigger: 'axis', valueFormatter: (v: number | null) => (v == null ? '–' : fmt(v)) },
      xAxis: catAxis(fs, years.map(String), { boundaryGap: true }),
      yAxis: valueAxis(fs, { scale: true, name: unit, nameLocation: 'end', nameGap: 12, nameTextStyle: { align: 'left', fontSize: fs, color: T.ink2 }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => compact(v) } }),
      series: [
        ...variants.map((v, i) => ({
          type: 'line', id: `v${i}`, name: `Model: ${vName(v.variant)}`, data: v.years.map((y) => y[`model_${metric}`] ?? null), showSymbol: false, symbol: 'none', emphasis: { disabled: true },
          lineStyle: i ? { color: T.comparator, width: 2, type: [6, 4] } : { color: T.model, width: 2.75 }, itemStyle: { color: i ? T.comparator : T.model },
        })),
        { type: 'scatter', id: 'obs', name: 'Observed (Statistics Jersey)', data: obs, symbolSize: present ? 9 : 8, itemStyle: { color: T.observed }, z: 5 },
      ],
    };
  }, [present, tkey, metric, data]);

  const rows = variants.length ? variants[0].years.map((y, i) => [y.year, fmt(y[`observed_${metric}`]), ...variants.map((v) => fmt(v.years[i][`model_${metric}`]))]) : [];
  const errKeys = variants.length ? Object.keys(variants[0].end_2025_error_model_minus_observed) : [];
  const oc = data?.official_projection_one_year_check;
  const em = data?.reference_emulation;
  const footer = data && (
    <div className="jfe-val">
      <div className="jfe-val__block">
        <h4>End-2025 error, model minus observed</h4>
        <table className="jfe-mini">
          <thead><tr><th scope="col">Metric</th>{variants.map((v) => <th scope="col" key={v.variant}>{vName(v.variant)}</th>)}</tr></thead>
          <tbody>{errKeys.map((k) => (
            <tr key={k}><th scope="row">{VAL_LABEL[k] ?? k}</th>{variants.map((v) => (
              <td key={v.variant} className="jfe-num">{signed(v.end_2025_error_model_minus_observed[k])} <span className="jfe-muted">({signed(v.end_2025_relative_error_pct[k], 1)}%)</span></td>
            ))}</tr>
          ))}</tbody>
        </table>
      </div>
      {oc && (
        <div className="jfe-val__block">
          <h4>Official projection (Feb 2026, +400 assumed; actual 2025 net migration +182) vs observed, end-2025</h4>
          <table className="jfe-mini">
            <thead><tr><th scope="col">Metric</th><th scope="col">Projected 2025</th><th scope="col">Observed 2025</th></tr></thead>
            <tbody>{Object.entries(oc.values).map(([k, v]) => (
              <tr key={k}><th scope="row">{VAL_LABEL[k] ?? k}</th><td className="jfe-num">{fmt(v.official_projection_2025)}</td><td className="jfe-num">{fmt(v.observed_2025)}</td></tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {em && (
        <div className="jfe-val__block">
          <h4>Reproduction of official projections, max |engine − official| (people)</h4>
          <p className="jfe-num">{Object.entries(em.max_abs_difference_people).map(([k, v]) => `${VAL_LABEL[k] ?? k} ${fmt(v, 1)}`).join(' · ')}</p>
        </div>
      )}
    </div>
  );
  const controls = available.length ? (
    <ToggleGroup isCompact aria-label="Validation metric">
      {available.map(([k, l]) => <ToggleGroupItem key={k} text={l} buttonId={`v-${k}`} isSelected={metric === k} onChange={() => setMetric(k)} />)}
    </ToggleGroup>
  ) : undefined;
  return (
    <Frame id="p08" onActive={onActive} spec={{ kind: 'reconciliation', run_id: 'baseline' }} csvFromServer={false} title={`Model vs observed, ${years[0] ?? 2018}-${years[years.length - 1] ?? 2025} (${metric === 'deaths' ? 'deaths' : 'people'})`}
      controls={controls} option={option} empty={error || (!data ? 'Loading the retrospective check…' : undefined)}
      table={{ head: ['Year', 'Observed', ...variants.map((v) => `Model: ${vName(v.variant)}`)], rows }}
      footer={footer}
      caption={<>Retrospective run from observed end-2017 · Source: Statistics Jersey population estimates (Sept 2026); vraic-fe cohort model</>}
    />
  );
}

// ------------------------------------------------------------------ schools: pupil proxies (illustrative, not capacity)
const SCHOOL: [string, string][] = [['pupils_primary_proxy', 'Primary'], ['pupils_secondary_proxy', 'Secondary incl. sixth form']];

export function Schools({ r, color, present, isBaseline, runLabel, tkey, runId, onActive }: PlotProps) {
  const available = SCHOOL.every(([k]) => r.metrics.scenario[k]);
  const scenName = isBaseline ? 'Baseline' : 'Scenario';
  const census = 'School census (January)';
  const offName = officialShort(r.official.label);
  const first = Math.min(r.years[0], ...SCHOOL.flatMap(([k]) => r.observed.years.filter((_, i) => r.observed.metrics[k]?.[i] != null)));
  const option = useMemo(() => {
    const { fs, opt } = base(present);
    if (!available) return opt;
    const pairs = (years: number[], vals: (number | null)[] | undefined) => (vals ? years.flatMap((y, i) => (vals[i] == null ? [] : [[y, vals[i]]])) : []);
    const line = { type: 'line', showSymbol: false, symbol: 'none', emphasis: { disabled: true } };
    const xAxis = (gridIndex: number) => valueAxis(fs, { gridIndex, min: first, max: r.years[r.years.length - 1], minInterval: 1, splitLine: { show: false }, axisLine: { show: true, lineStyle: { color: T.axis } }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => String(v) } });
    return {
      ...opt,
      title: SCHOOL.map(([, t], i) => ({ text: t, left: i ? '54%' : 0, top: present ? 30 : 26, textStyle: { fontSize: fs, fontWeight: 600, color: T.ink } })),
      legend: { ...opt.legend, data: [{ name: census, itemStyle: { opacity: 1 } }, offName, ...(isBaseline ? [] : ['Baseline']), scenName] },
      grid: [0, 1].map((i) => ({ left: i ? '54%' : 8, width: '42%', top: present ? 80 : 70, bottom: 24, containLabel: true })),
      tooltip: { ...opt.tooltip, trigger: 'axis', formatter: (ps: { seriesName: string; marker: string; value: number[] }[]) => `<b>School year from Sept ${ps[0]?.value[0]}</b><br/>${ps.map((p) => `${p.marker}${p.seriesName}: <b>${fmt(p.value[1])}</b>`).join('<br/>')}` },
      xAxis: [xAxis(0), xAxis(1)],
      yAxis: [0, 1].map((gridIndex) => valueAxis(fs, { gridIndex, scale: true, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => compact(v) } })),
      series: SCHOOL.flatMap(([k], i) => [
        { type: 'scatter', id: `obs-${k}`, name: census, xAxisIndex: i, yAxisIndex: i, data: pairs(r.observed.years, r.observed.metrics[k]), symbolSize: present ? 8 : 7, itemStyle: { color: T.observed }, z: 5 },
        { ...line, id: `off-${k}`, name: offName, xAxisIndex: i, yAxisIndex: i, data: pairs(r.official.years, r.official.metrics[k]), lineStyle: { color: T.official, width: 1.5 }, itemStyle: { color: T.official } },
        ...(isBaseline ? [] : [{ ...line, id: `comp-${k}`, name: 'Baseline', xAxisIndex: i, yAxisIndex: i, data: pairs(r.years, central(r.metrics.comparator[k])), lineStyle: { color: T.comparator, width: 1.75, type: [6, 4] }, itemStyle: { color: T.comparator } }]),
        { ...line, id: `scen-${k}`, name: scenName, xAxisIndex: i, yAxisIndex: i, data: pairs(r.years, central(r.metrics.scenario[k])), lineStyle: { color, width: 2.75 }, itemStyle: { color }, z: 6,
          endLabel: { show: true, formatter: (p: { value: number[] }) => fmt(p.value[1]), color: T.ink, fontWeight: 600, fontSize: fs } },
      ]),
    };
  }, [present, tkey, color, r, isBaseline, available, first]);

  const years = Array.from({ length: r.years[r.years.length - 1] - first + 1 }, (_, i) => first + i);
  const val = (src: { years: number[]; vals?: (number | null)[] }, y: number) => { const i = src.years.indexOf(y); return i >= 0 && src.vals ? fmt(src.vals[i]) : '–'; };
  const head = ['School year (from Sept)', ...SCHOOL.flatMap(([, t]) => [`${t}: census`, ...(isBaseline ? [] : [`${t}: comparator`]), `${t}: ${scenName.toLowerCase()}`])];
  const rows = available ? years.map((y) => [y, ...SCHOOL.flatMap(([k]) => [
    val({ years: r.observed.years, vals: r.observed.metrics[k] }, y),
    ...(isBaseline ? [] : [val({ years: r.years, vals: central(r.metrics.comparator[k]) }, y)]),
    val({ years: r.years, vals: central(r.metrics.scenario[k]) }, y),
  ])]) : [];
  return (
    <Frame id="p05s" title="Projected pupils (people)" option={option} table={{ head, rows }} onActive={onActive}
      spec={{ kind: 'trajectory', metrics: SCHOOL.map(([k]) => k), run_id: runId }}
      empty={available ? undefined : 'Pupil projections are not available for this result.'}
      caption={<Caption r={r} runLabel={runLabel} items={['Unchanged January-2024 participation; not school capacity', 'Year = school year starting September', 'Dots: January school census']} />}
    />
  );
}

// ------------------------------------------------------------------ generic renderer for server chart tables (chatbot analysis, Fan, Health)
const pctCol = (c: string) => c.endsWith('_pct');
const unitDigitsOf = (unit: string) => (['index', 'per 100'].includes(unit) ? 1 : 0);

export function ChartFromData({ data, color, present, tkey, onActive, runColors = {}, id = 'pa', controls, compactHeight }: {
  data: ChartData; color: string; present: boolean; tkey: string; onActive?: (a: ActiveChart) => void; runColors?: Record<string, string>;
  id?: string; controls?: ReactNode; compactHeight?: boolean;
}) {
  const { spec, columns, rows } = data;
  // on the baseline the comparator is the same run: drop it and call the main series Baseline
  const onBaseline = !spec.run_id || spec.run_id === 'baseline';
  const shown = onBaseline ? data.series.filter((s) => s.role !== 'comparator') : data.series;
  const metrics = spec.metrics ?? [];
  const short = data.title.split(' / ');
  const metricShort = (m?: string) => (m && short.length === metrics.length ? short[metrics.indexOf(m)].replace(/\s*\([^)]*\)$/, '') : m ?? '');
  const roleName = (sr: ChartData['series'][number]) => ({
    scenario: onBaseline ? 'Baseline' : 'Scenario', comparator: 'Baseline', official: officialShort(sr.label), observed: 'Observed', difference: metricShort(sr.metric) || sr.label,
  } as Record<string, string>)[sr.role] ?? (sr.label.length > 32 ? sr.label.slice(0, 31) + '…' : sr.label);
  // Several metrics on one chart: the legend lists roles once; metrics are told apart by markers and end labels.
  const seriesName = (sr: ChartData['series'][number]) => (metrics.length > 1 && sr.role === 'scenario' ? metricShort(sr.metric) : roleName(sr));
  const groupLabel = (g: string) => (g === 'population_total' ? 'Total' : g.replace(/^population_/, '').replace(/_plus$/, '+').replace(/_/g, '-'));
  const digits = unitDigitsOf(data.unit);
  const col = (name: string) => columns.indexOf(name);
  const num = (v: unknown) => (v == null || v === '' ? null : Number(v));

  const option = useMemo(() => {
    const { fs, opt } = base(present);
    const line = { type: 'line', showSymbol: false, symbol: 'none', emphasis: { disabled: true } };
    const yName = { name: data.unit, nameLocation: 'end', nameGap: 12, nameTextStyle: { align: 'left', fontSize: fs, color: T.ink2 } };
    const fmtV = (v: number | null) => (v == null ? '–' : fmt(v, digits));
    const yearAxis = (ys: number[], extra: object = {}) => valueAxis(fs, { min: Math.min(...ys), max: Math.max(...ys), minInterval: 1, splitLine: { show: false }, axisLine: { show: true, lineStyle: { color: T.axis } }, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => String(v) }, ...extra });
    const yi = col('year');
    const years = yi >= 0 ? [...new Set(rows.map((r) => Number(r[yi])))] : [];
    const pts = (c: number) => rows.flatMap((r) => (num(r[c]) == null ? [] : [[Number(r[yi]), num(r[c]) as number]]));
    const axisTooltip = { ...opt.tooltip, trigger: 'axis', valueFormatter: (v: number | number[] | null) => fmtV(Array.isArray(v) ? v[1] : v) };
    const band = (id: string, lo: number, hi: number, op: number, name?: string) => [
      { ...line, id: `${id}-b`, stack: id, data: pts(lo), lineStyle: { opacity: 0 }, tooltip: { show: false }, silent: true },
      { ...line, id: `${id}-h`, name, stack: id, data: rows.flatMap((r) => (num(r[lo]) == null || num(r[hi]) == null ? [] : [[Number(r[yi]), (num(r[hi]) as number) - (num(r[lo]) as number)]])), lineStyle: { opacity: 0 }, areaStyle: { color, opacity: op }, itemStyle: { color }, silent: true, tooltip: { show: false } },
    ];
    const grid = { left: 8, right: 72, top: present ? 60 : 52, bottom: 24, containLabel: true };

    if (spec.kind === 'trajectory') {
      const series = shown.map((sr) => {
        const c = col(sr.column);
        const mi = metrics.indexOf(sr.metric ?? '');
        const name = seriesName(sr);
        if (sr.role === 'observed') return { type: 'scatter', id: sr.column, name, data: pts(c), symbol: mi > 0 ? 'triangle' : 'circle', symbolSize: present ? 8 : 7, itemStyle: { color: T.observed }, z: 5 };
        const st = sr.role === 'comparator' ? { color: T.comparator, width: 1.75, type: [6, 4] } : sr.role === 'official' ? { color: T.official, width: 1.5 }
          : sr.role === 'other_run' ? { color: runColors[sr.column.split('run:')[1]] ?? T.comparator, width: 1.5 } : { color, width: 2.75 };
        return {
          ...line, ...(mi > 0 ? { showSymbol: true, symbol: 'triangle', symbolSize: 6 } : {}), id: sr.column, name, data: pts(c), lineStyle: st, itemStyle: { color: st.color }, z: sr.role === 'scenario' ? 6 : 3,
          endLabel: sr.role === 'scenario' ? { show: true, formatter: (p: { value: number[] }) => fmt(p.value[1], digits), color: T.ink, fontWeight: 600, fontSize: fs } : undefined,
        };
      });
      const names = [...new Set(series.map((x) => x.name))];
      return {
        ...opt, grid: { ...grid, top: grid.top + (names.length > 4 && !compactHeight ? 22 : 0) },
        legend: { ...opt.legend, ...(compactHeight ? { type: 'scroll', right: 0 } : {}), data: names.map((n) => (n === 'Observed' || series.find((x) => x.name === n && 'showSymbol' in x && x.showSymbol) ? { name: n, itemStyle: { opacity: 1 } } : n)) },
        tooltip: axisTooltip, xAxis: yearAxis(years), yAxis: valueAxis(fs, { scale: true, ...yName, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => (digits ? fmt(v) : compact(v)) } }), series,
      };
    }
    if (spec.kind === 'fan') {
      return {
        ...opt, grid,
        legend: { ...opt.legend, data: [{ name: '90% range', icon: 'rect', itemStyle: { color, opacity: 0.26 } }, { name: '50% range', icon: 'rect', itemStyle: { color, opacity: 0.44 } }, 'Median', 'Baseline median'] },
        tooltip: { ...opt.tooltip, trigger: 'axis', formatter: (ps: { value: number[] }[]) => {
          const y = ps[0]?.value[0]; const r = rows.find((x) => Number(x[yi]) === y); if (!r) return '';
          const v = (c: string) => fmtV(num(r[col(c)]));
          return `<b>End-${y}</b><br/>Median: <b>${v('p50')}</b><br/>50%: ${v('p25')}–${v('p75')}<br/>90%: ${v('p05')}–${v('p95')}<br/>Baseline median: ${v('comparator_p50')}`;
        } },
        xAxis: yearAxis(years), yAxis: valueAxis(fs, { scale: true, ...yName, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => (digits ? fmt(v) : compact(v)) } }),
        series: [
          ...band('b90', col('p05'), col('p95'), 0.13, '90% range'), ...band('b50', col('p25'), col('p75'), 0.22, '50% range'),
          { ...line, id: 'cmp', name: 'Baseline median', data: pts(col('comparator_p50')), lineStyle: { color: T.comparator, width: 1.75, type: [6, 4] }, itemStyle: { color: T.comparator } },
          { ...line, id: 'p50', name: 'Median', data: pts(col('p50')), lineStyle: { color, width: 2.75 }, itemStyle: { color }, z: 6, endLabel: { show: true, formatter: (p: { value: number[] }) => fmt(p.value[1], digits), color: T.ink, fontWeight: 600, fontSize: fs } },
        ],
      };
    }
    if (spec.kind === 'difference') {
      const main = shown.filter((sr) => sr.role === 'difference');
      const all = main.flatMap((sr) => [sr.column, `${sr.column}_p05`, `${sr.column}_p95`]).map(col).filter((c) => c >= 0);
      const m = niceSym(Math.max(10, ...rows.flatMap((r) => all.map((c) => Math.abs(num(r[c]) ?? 0)))) * 1.05);
      const dash: (number[] | 'solid')[] = ['solid', [8, 4], [2, 3]];
      return {
        ...opt, grid: { ...grid, right: 110 },
        legend: main.length > 1 ? { ...opt.legend, data: main.map((sr) => seriesName(sr)) } : { show: false },
        tooltip: { ...axisTooltip, valueFormatter: (v: number | number[]) => signed(Array.isArray(v) ? v[1] : v, digits) },
        xAxis: yearAxis(years), yAxis: valueAxis(fs, { min: -m.max, max: m.max, interval: m.step, ...yName, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => signed(v) } }),
        series: main.flatMap((sr, i) => [
          ...(col(`${sr.column}_p05`) >= 0 ? band(`d${i}`, col(`${sr.column}_p05`), col(`${sr.column}_p95`), 0.12) : []),
          { ...line, id: sr.column, name: seriesName(sr), data: pts(col(sr.column)), lineStyle: { color, width: i ? 2 : 2.75, type: dash[i % 3] }, itemStyle: { color },
            endLabel: { show: true, formatter: (p: { value: number[] }) => signed(p.value[1], digits), color: T.ink, fontSize: fs }, labelLayout: { moveOverlap: 'shiftY' },
            markLine: i ? undefined : { silent: true, symbol: 'none', label: { show: false }, data: [{ yAxis: 0 }], lineStyle: { color: T.ink, width: 1, type: 'solid' } } },
        ]),
      };
    }
    if (spec.kind === 'pyramid') {
      const [ai, si, yc, sc, cc] = ['age_band', 'sex', 'year', 'scenario', 'comparator'].map(col);
      const pys = [...new Set(rows.map((r) => Number(r[yc])))];
      const bands = [...new Set(rows.map((r) => String(r[ai])))];
      const val = (y: number, sex: string, c: number) => bands.map((b) => num(rows.find((r) => Number(r[yc]) === y && r[si] === sex && r[ai] === b)?.[c]) ?? 0);
      const mx = Math.max(...rows.flatMap((r) => [num(r[sc]) ?? 0, num(r[cc]) ?? 0]));
      const step = [250, 500, 1000, 2000, 2500, 5000].find((st) => st * (pys.length > 1 ? 3 : 5) >= mx) ?? 10000;
      const xmax = Math.ceil((mx * 1.02) / step) * step;
      const w = pys.length > 1 ? 44 : 92;
      const bar = { type: 'bar', barCategoryGap: '14%', emphasis: { disabled: true } };
      const outline = { ...bar, barGap: '-100%', z: 3, itemStyle: { color: 'rgba(0,0,0,0)', borderColor: T.ink, borderWidth: 1 } };
      return {
        ...opt,
        title: pys.map((y, i) => ({ text: `End-${y}`, left: `${4 + i * 50}%`, top: 22, textStyle: { fontSize: fs, fontWeight: 600, color: T.ink } })),
        legend: { ...opt.legend, left: 'center', icon: 'rect', data: [{ name: 'Scenario', itemStyle: { color, opacity: 1 } }, { name: 'Baseline', itemStyle: { color: 'rgba(0,0,0,0)', borderColor: T.ink, borderWidth: 1, opacity: 1 } }] },
        grid: pys.map((_, i) => ({ left: `${4 + i * 50}%`, width: `${w}%`, top: 48, bottom: 36, containLabel: true })),
        tooltip: { ...opt.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => fmt(Math.abs(v)) },
        xAxis: pys.map((_, i) => valueAxis(fs, { gridIndex: i, min: -xmax, max: xmax, interval: step, name: 'Female  |  Male (people)', nameLocation: 'middle', nameGap: 22, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => compact(Math.abs(v)) } })),
        yAxis: pys.map((_, i) => ({ type: 'category', gridIndex: i, data: bands, axisTick: { show: false }, axisLine: { lineStyle: { color: T.axis } }, axisLabel: { fontSize: present ? 12 : 10, color: T.ink2, interval: 0 } })),
        series: pys.flatMap((y, i) => [
          { ...bar, name: 'Scenario', stack: `s${i}`, xAxisIndex: i, yAxisIndex: i, data: val(y, 'F', sc).map((v) => -v), itemStyle: { color } },
          { ...bar, name: 'Scenario', stack: `s${i}`, xAxisIndex: i, yAxisIndex: i, data: val(y, 'M', sc), itemStyle: { color } },
          { ...outline, name: 'Baseline', stack: `c${i}`, xAxisIndex: i, yAxisIndex: i, data: val(y, 'F', cc).map((v) => -v) },
          { ...outline, name: 'Baseline', stack: `c${i}`, xAxisIndex: i, yAxisIndex: i, data: val(y, 'M', cc) },
        ]),
      };
    }
    if (spec.kind === 'ranking') {
      const key = spec.basis === 'vs_start' ? columns.find((c) => c === 'change_vs_start') ?? 'change_vs_start' : 'difference_vs_comparator';
      const kc = col(key), ac = col('age_band');
      const vals = rows.map((r) => num(r[kc]) ?? 0);
      const m = niceSym(Math.max(10, ...vals.map(Math.abs)) * 1.15);
      return {
        ...opt, grid: { left: 8, right: 24, top: 30, bottom: 24, containLabel: true }, legend: { show: false },
        tooltip: { ...opt.tooltip, trigger: 'item', formatter: (p: { name: string; value: number }) => `<b>${p.name}</b><br/>${signed(p.value)} people` },
        xAxis: valueAxis(fs, { min: -m.max, max: m.max, interval: m.step, name: data.unit, nameLocation: 'middle', nameGap: 22, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => signed(v) } }),
        yAxis: { type: 'category', inverse: true, data: rows.map((r) => String(r[ac])), axisTick: { show: false }, axisLine: { lineStyle: { color: T.axis } }, axisLabel: { fontSize: present ? 12 : 11, color: T.ink2, interval: 0 } },
        series: [{ type: 'bar', itemStyle: { color }, barCategoryGap: '25%', emphasis: { disabled: true },
          data: vals.map((v) => ({ value: v, label: { position: v < 0 ? 'left' : 'right' } })),
          label: { show: true, formatter: (p: { value: number }) => signed(p.value), color: T.ink, fontSize: fs - 1 } }],
      };
    }
    if (spec.kind === 'housing') {
      const tc = col('dwelling_type'), sc = col('scenario'), cc = col('comparator');
      const types = rows.map((r) => String(r[tc]));
      return {
        ...opt, grid: { left: 8, right: 56, top: 36, bottom: 28, containLabel: true },
        legend: { ...opt.legend, icon: 'rect', itemStyle: { opacity: 1 }, data: [{ name: 'Scenario', itemStyle: { color } }, { name: 'Baseline', itemStyle: { color: T.comparator } }] },
        tooltip: { ...opt.tooltip, trigger: 'axis', axisPointer: { type: 'shadow' }, valueFormatter: (v: number) => `${fmt(v)} homes` },
        xAxis: valueAxis(fs, { name: data.unit, nameLocation: 'middle', nameGap: 24, axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => signed(v) } }),
        yAxis: { type: 'category', inverse: true, data: types, axisTick: { show: false }, axisLine: { lineStyle: { color: T.axis } }, axisLabel: { fontSize: fs, color: T.ink2, interval: 0 } },
        series: [
          { type: 'bar', name: 'Scenario', data: rows.map((r) => num(r[sc])), itemStyle: { color }, barGap: '10%', barCategoryGap: '30%', emphasis: { disabled: true },
            label: { show: true, position: 'right', formatter: (p: { value: number }) => signed(p.value), color: T.ink, fontSize: fs - 1 } },
          { type: 'bar', name: 'Baseline', data: rows.map((r) => num(r[cc])), itemStyle: { color: T.comparator }, emphasis: { disabled: true } },
        ],
      };
    }
    if (spec.kind === 'response' || spec.kind === 'frontier') {
      const ex = data.extra ?? {};
      const isNet = spec.target?.parameter === 'net_migration';
      const pfmt = (v: number) => (isNet ? signed(v) : `×${v.toFixed(2)}`);
      if (spec.kind === 'response') {
        const xc = col('parameter_value'), qc = col('metric_quantile'), mc = qc >= 0 ? qc : col('metric_value'), tc = col('target');
        const curve = rows.flatMap((r) => (num(r[mc]) == null ? [] : [[Number(r[xc]), num(r[mc]) as number]]));
        // probabilistic goal seek: the chosen-probability curve is the main line, the median is dashed
        const median = qc >= 0 ? rows.flatMap((r) => (num(r[col('metric_value')]) == null ? [] : [[Number(r[xc]), num(r[col('metric_value')]) as number]])) : [];
        const mainName = (qc >= 0 ? shown.find((s) => s.column === 'metric_quantile')?.label : shown[0]?.label) ?? '';
        const tgt = num(rows[0]?.[tc]);
        const sol = ex.solution;
        const solved = sol?.status === 'solved' && sol.value != null && sol.achieved != null;
        const xsAll = rows.map((r) => Number(r[xc]));
        return {
          ...opt, grid: { ...grid, right: 32, bottom: 40 }, legend: median.length ? { ...opt.legend, data: [mainName, 'Median'] } : { show: false },
          tooltip: { ...opt.tooltip, trigger: 'axis', formatter: (ps: { value: number[] }[]) => `${pfmt(ps[0].value[0])}<br/><b>${fmtV(ps[0].value[1])}</b>` },
          xAxis: valueAxis(fs, { min: Math.min(...xsAll), max: Math.max(...xsAll), name: ex.x_label, nameLocation: 'middle', nameGap: 28, splitLine: { show: false },
            axisLine: { show: true, onZero: false, lineStyle: { color: T.axis } }, axisLabel: { fontSize: fs, color: T.ink2, formatter: pfmt, showMinLabel: false } }),
          // the target stays in view even when it is out of reach (that gap is the answer)
          yAxis: valueAxis(fs, { ...yName, scale: true, axisLine: { show: false, onZero: false },
            ...(tgt == null ? {} : { min: (v: { min: number; max: number }) => nice(Math.min(v.min, tgt), Math.max(v.max, tgt) - Math.min(v.min, tgt), Math.floor),
                                     max: (v: { min: number; max: number }) => nice(Math.max(v.max, tgt), Math.max(v.max, tgt) - Math.min(v.min, tgt), Math.ceil) }) }),
          series: [...(median.length ? [{ ...line, id: 'median', name: 'Median', data: median, lineStyle: { color: T.comparator, width: 1.5, type: [6, 4] }, itemStyle: { color: T.comparator } }] : []),
            { ...line, id: 'response', name: mainName, data: curve, lineStyle: { color, width: 2.75 }, itemStyle: { color },
            markLine: { silent: true, symbol: 'none', data: [
              ...(tgt == null ? [] : [{ yAxis: tgt, lineStyle: { color: T.ink, type: [6, 4], width: 1 }, label: { show: true, formatter: `Target ${fmtV(tgt)}`, position: 'insideStartTop', color: T.ink2, fontSize: fs } }]),
              ...(solved ? [{ xAxis: sol!.value as number, lineStyle: { color, type: 'solid', width: 1 }, label: { show: true, formatter: pfmt(sol!.value as number), position: 'end', distance: 6, color: T.ink, fontSize: fs, fontWeight: 600 } }] : []),
            ] },
            markPoint: solved ? { symbol: 'circle', symbolSize: 11, data: [{ coord: [sol!.value, sol!.achieved] }], itemStyle: { color, borderColor: T.surface, borderWidth: 2 }, label: { show: false } } : undefined }],
        };
      }
      const xs = ex.x ?? [], ys = ex.y ?? [];
      const zc = col('metric_value');
      const z = rows.map((r) => num(r[zc]) as number);
      const tgt = ex.target ?? 0;
      const dx = xs.length > 1 ? xs[1] - xs[0] : 1, dy = ys.length > 1 ? ys[1] - ys[0] : 1;
      const contour: number[][] = [];
      ys.forEach((yv, j) => {
        for (let i = 0; i < xs.length - 1; i++) {
          const a = z[j * xs.length + i] - tgt, b = z[j * xs.length + i + 1] - tgt;
          if (a === 0 || a * b < 0) { contour.push([xs[i] + (a === 0 ? 0 : (a / (a - b)) * dx), yv]); break; }
        }
      });
      const zmin = Math.min(...z), zmax = Math.max(...z);
      return {
        ...opt, grid: { ...grid, right: 96, bottom: 40 }, legend: { show: false },
        tooltip: { ...opt.tooltip, trigger: 'item', formatter: (p: { seriesType: string; value: number[] }) => (p.seriesType === 'custom'
          ? `${ex.x_label}: ${signed(p.value[0])}<br/>${ex.y_label}: ×${p.value[1].toFixed(2)}<br/><b>${fmtV(p.value[2])}</b>` : `Target ${fmtV(tgt)}`) },
        visualMap: { type: 'continuous', min: zmin, max: zmax, dimension: 2, seriesIndex: 0, right: 0, top: 'middle', itemHeight: 180, calculable: false,
          inRange: { color: ['#f7fbff', '#c6dbef', '#6baed6', '#2171b5', '#08306b'] }, text: [fmt(zmax, digits), fmt(zmin, digits)], textStyle: { color: T.ink2, fontSize: fs } },
        xAxis: valueAxis(fs, { min: xs[0] - dx / 2, max: xs[xs.length - 1] + dx / 2, name: ex.x_label, nameLocation: 'middle', nameGap: 28, splitLine: { show: false }, axisLine: { onZero: false },
          axisLabel: { fontSize: fs, color: T.ink2, showMinLabel: false, showMaxLabel: false, formatter: (v: number) => signed(v) } }),
        yAxis: valueAxis(fs, { min: ys[0] - dy / 2, max: ys[ys.length - 1] + dy / 2, name: ex.y_label, nameLocation: 'end', nameGap: 12, splitLine: { show: false }, axisLine: { onZero: false },
          nameTextStyle: { align: 'left', fontSize: fs, color: T.ink2 }, axisLabel: { fontSize: fs, color: T.ink2, showMinLabel: false, showMaxLabel: false, formatter: (v: number) => `×${v.toFixed(1)}` } }),
        series: [
          { type: 'custom', id: 'cells', data: rows.map((_r, k) => [xs[k % xs.length], ys[Math.floor(k / xs.length)], z[k]]), encode: { x: 0, y: 1 }, emphasis: { disabled: true },
            renderItem: (_p: unknown, api: { value: (i: number) => number; coord: (v: number[]) => number[]; visual: (k: string) => string }) => {
              const a = api.coord([api.value(0) - dx / 2, api.value(1) - dy / 2]), b = api.coord([api.value(0) + dx / 2, api.value(1) + dy / 2]);
              return { type: 'rect', shape: { x: Math.min(a[0], b[0]), y: Math.min(a[1], b[1]), width: Math.abs(b[0] - a[0]) + 0.6, height: Math.abs(b[1] - a[1]) + 0.6 }, style: { fill: api.visual('color') } };
            } },
          { ...line, id: 'contour', name: 'Target', data: contour, lineStyle: { color: '#f0ab00', width: 3 }, z: 5 },
        ],
      };
    }
    // reconciliation: engine minus official, by age group
    const gc = col('age_group'), dc = col('engine_minus_official');
    const groups = [...new Set(rows.map((r) => String(r[gc])))];
    const palette = [T.model, T.official, T.comparator, T.observed];
    const dash: (number[] | 'solid')[] = ['solid', [8, 4], [2, 3], [10, 3, 2, 3]];
    return {
      ...opt, grid, legend: { ...opt.legend, data: groups.map(groupLabel) },
      tooltip: { ...axisTooltip, valueFormatter: (v: number | number[]) => `${signed(Array.isArray(v) ? v[1] : v, 1)} people` },
      xAxis: yearAxis(years), yAxis: valueAxis(fs, { ...yName, name: 'engine minus official (people)', axisLabel: { fontSize: fs, color: T.ink2, formatter: (v: number) => signed(v, 1) } }),
      series: groups.map((g, i) => ({ ...line, id: g, name: groupLabel(g), data: rows.filter((r) => String(r[gc]) === g).map((r) => [Number(r[yi]), num(r[dc])]), lineStyle: { color: palette[i % 4], width: 2, type: dash[i % 4] }, itemStyle: { color: palette[i % 4] } })),
    };
  }, [data, present, tkey, color, JSON.stringify(runColors), compactHeight]);

  const head = columns.map((c) => {
    const sr = shown.find((x) => x.column === c);
    return sr ? seriesName(sr) : c.replace(/_/g, ' ').replace(/^./, (ch) => ch.toUpperCase());
  });
  const keep = new Set(['year', 'age_band', 'sex', 'age_group']);
  const body = rows.map((r) => r.map((v, j) => (v == null ? '–' : typeof v === 'number' && !keep.has(columns[j]) ? (pctCol(columns[j]) ? fmt(v, 1) : fmt(v, digits)) : String(v))));
  return (
    <Frame id={id} title={data.title} option={option} table={{ head, rows: body }} spec={spec} initialTable={spec.view === 'table'} onActive={onActive}
      controls={controls} compactHeight={compactHeight} caption={`${data.caption.replace(/\.$/, '')} · ${SOURCE}`} footnote={data.notes.length ? data.notes.join(' ') : undefined} />
  );
}

/** Fetches the server chart table for a spec and renders it. */
export function ServerChart({ spec, ...rest }: { spec: ChartSpec } & Omit<Parameters<typeof ChartFromData>[0], 'data'>) {
  const [state, setState] = useState<{ key: string; data?: ChartData; error?: string }>({ key: '' });
  const key = JSON.stringify(spec);
  useEffect(() => {
    let live = true;
    api<ChartData>('/api/chart-data', 'POST', { spec, format: 'json' }).then(
      (data) => live && setState({ key, data }), (e) => live && setState({ key, error: errText(e) }));
    return () => { live = false; };
  }, [key]);
  if (state.key === key && state.data) return <ChartFromData data={state.data} {...rest} />;
  return (
    <figure className={`jfe-plot ${rest.compactHeight ? 'is-compact' : ''}`}>
      <div className="jfe-plot__head"><div className="jfe-plot__controls">{rest.controls}</div></div>
      <div className="jfe-plot__empty">{state.key === key && state.error ? state.error : 'Loading…'}</div>
    </figure>
  );
}

export function Fan({ metrics, runId, ...rest }: Omit<Parameters<typeof ChartFromData>[0], 'data'> & { metrics: MetricDefs; runId: string }) {
  const [metric, setMetric] = useState('population_total');
  const controls = (
    <FormSelect aria-label="Metric" value={metric} onChange={(_e, v) => setMetric(v)} className="jfe-metric-select">
      {Object.entries(metrics).filter(([k]) => !FLOWS.includes(k)).map(([k, d]) => <FormSelectOption key={k} value={k} label={d.label} />)}
    </FormSelect>
  );
  return <ServerChart id="pfan" spec={{ kind: 'fan', metrics: [metric], run_id: runId }} controls={controls} {...rest} />;
}

export function Housing({ runId, isBaseline, onActive, ...rest }: Omit<Parameters<typeof ChartFromData>[0], 'data'> & { runId: string; isBaseline: boolean }) {
  const series: ChartSpec['series'] = isBaseline ? ['scenario', 'official'] : ['scenario', 'comparator', 'official'];
  return (
    <div className="jfe-pair">
      <ServerChart id="pho1" spec={{ kind: 'trajectory', metrics: ['homes_needed_additional'], series, run_id: runId }} onActive={onActive} compactHeight {...rest} />
      <ServerChart id="pho2" spec={{ kind: 'housing', run_id: runId }} compactHeight {...rest} />
    </div>
  );
}

export function Health({ runId, isBaseline, onActive, ...rest }: Omit<Parameters<typeof ChartFromData>[0], 'data'> & { runId: string; isBaseline: boolean }) {
  const series: ChartSpec['series'] = isBaseline ? ['scenario', 'official', 'observed'] : ['scenario', 'comparator', 'official', 'observed'];
  return (
    <div className="jfe-pair">
      <ServerChart id="ph1" spec={{ kind: 'trajectory', metrics: ['nurses_fte_proxy', 'doctors_fte_proxy'], series, run_id: runId }} onActive={onActive} compactHeight {...rest} />
      <ServerChart id="ph2" spec={{ kind: 'trajectory', metrics: ['hospital_bed_days'], series, run_id: runId }} compactHeight {...rest} />
    </div>
  );
}
