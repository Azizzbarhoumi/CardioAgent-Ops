import { useMemo, useState } from 'react';
import type { BenchmarkModel, ConfusionMatrix } from '../types';
import { modelName, useI18n, type Key } from '../i18n';

export const fmt = (m?: number | null, s?: number | null) => (m == null ? '–' : `${m.toFixed(3)} ± ${(s ?? 0).toFixed(3)}`);
const TABLE = ['recall', 'specificity', 'balanced_accuracy', 'f1', 'accuracy', 'roc_auc'] as const;
const GROUPED = ['recall', 'balanced_accuracy', 'f1'] as const;

function GroupedBars({ models }: { models: BenchmarkModel[] }) {
  const { t } = useI18n();
  const w = 900, h = 270, left = 40, base = h - 40, sy = (v: number) => base - (Math.max(0.5, v) - 0.5) * 2 * (base - 16);
  const gw = (w - left) / Math.max(models.length, 1), bw = gw / 4.2;
  return <figure className="chart-fig"><svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={t('bench.grouped')}>
    {[0.5, 0.6, 0.7, 0.8, 0.9, 1].map(v => <g key={v}><line className="grid" x1={left} x2={w} y1={sy(v)} y2={sy(v)} /><text x={4} y={sy(v) + 4}>{v.toFixed(1)}</text></g>)}
    {models.map((m, i) => { const gx = left + 6 + i * gw + gw * 0.1; return <g key={m.model_key} data-bars-key={m.model_key}>
      {GROUPED.map((k, j) => { const mu = m.mean.test[k] ?? 0, sd = m.std.test[k] ?? 0, x = gx + j * bw;
        return <g key={k}><rect className={`b${j}`} x={x} y={sy(mu)} width={bw - 4} height={base - sy(mu)}><title>{`${modelName(t, m.model_key, m.display_name, true)} · ${t(`metric.${k}`)}: ${fmt(mu, sd)}`}</title></rect>
          <line className="err" x1={x + bw / 2 - 2} x2={x + bw / 2 - 2} y1={sy(mu + sd)} y2={sy(mu - sd)} /><line className="err" x1={x + bw / 2 - 6} x2={x + bw / 2 + 2} y1={sy(mu + sd)} y2={sy(mu + sd)} /><line className="err" x1={x + bw / 2 - 6} x2={x + bw / 2 + 2} y1={sy(mu - sd)} y2={sy(mu - sd)} /></g>; })}
      <text x={gx + bw * 1.5} y={base + 18} textAnchor="middle">{modelName(t, m.model_key, m.display_name, true)}</text></g>; })}
  </svg>
  <figcaption className="legend">{GROUPED.map((k, j) => <span key={k}><i className={`b${j}`} />{t(`metric.${k}`)}</span>)}<span>{t('bench.groupedNote')}</span></figcaption></figure>;
}

function TrainTest({ models }: { models: BenchmarkModel[] }) {
  const { t } = useI18n();
  const w = 900, h = 250, left = 40, base = h - 46, sy = (v: number) => base - (Math.max(0.5, v) - 0.5) * 2 * (base - 20);
  const gw = (w - left) / Math.max(models.length, 1), bw = gw / 3.2;
  return <figure className="chart-fig"><svg className="chart" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={t('bench.tvt')}>
    {[0.5, 0.75, 1].map(v => <g key={v}><line className="grid" x1={left} x2={w} y1={sy(v)} y2={sy(v)} /><text x={4} y={sy(v) + 4}>{v}</text></g>)}
    {models.map((m, i) => { const gx = left + 6 + i * gw + gw * 0.17, tr = m.mean.train.balanced_accuracy ?? 0, te = m.mean.test.balanced_accuracy ?? 0;
      return <g key={m.model_key}>{([['tr', tr, m.std.train.balanced_accuracy], ['te', te, m.std.test.balanced_accuracy]] as const).map(([c, v, sd], j) =>
        <g key={c}><rect className={c} x={gx + j * bw} y={sy(v)} width={bw - 4} height={base - sy(v)}><title>{`${t(c === 'tr' ? 'split.train' : 'split.test')}: ${fmt(v, sd)}`}</title></rect>
          <text className="val" x={gx + j * bw + bw / 2 - 2} y={sy(v) - 5} textAnchor="middle">{v.toFixed(3)}</text></g>)}
        <text x={gx + bw} y={base + 17} textAnchor="middle">{modelName(t, m.model_key, m.display_name, true)}</text>
        <text x={gx + bw} y={base + 33} textAnchor="middle" className="muted-t">{t('bench.gap', { v: (tr - te).toFixed(3) })}</text></g>; })}
  </svg>
  <figcaption className="legend"><span><i className="tr" />{t('split.train')}</span><span><i className="te" />{t('split.test')}</span><span>{t('bench.tvtNote')}</span></figcaption></figure>;
}

function Heatmap({ models }: { models: BenchmarkModel[] }) {
  const { t } = useI18n();
  const folds = [...new Set(models.flatMap(m => m.folds.map(f => f.fold)))].sort();
  const vals = models.flatMap(m => m.folds.map(f => f.test.recall ?? 0));
  const lo = Math.min(...vals, 1), hi = Math.max(...vals, 0);
  return <div className="ts"><table className="heat" data-testid="recall-heatmap"><thead><tr><th>{t('rel.recall')}</th>{folds.map(f => <th key={f} className="r">{t('bench.cmFold')} {f}</th>)}</tr></thead>
    <tbody>{models.map(m => <tr key={m.model_key}><th>{modelName(t, m.model_key, m.display_name, true)}</th>{folds.map(f => { const v = m.folds.find(x => x.fold === f)?.test.recall;
      return <td key={f} className="n" style={{ ['--v' as string]: v == null ? 0 : ((v - lo) / Math.max(hi - lo, 1e-9)).toFixed(3) }}>{v == null ? '–' : v.toFixed(3)}</td>; })}</tr>)}</tbody></table></div>;
}

function Confusion({ models }: { models: BenchmarkModel[] }) {
  const { t } = useI18n();
  const [key, setKey] = useState(models[0]?.model_key ?? '');
  const [fold, setFold] = useState<'sum' | number>('sum');
  const [split, setSplit] = useState<'test' | 'train'>('test');
  const model = models.find(m => m.model_key === key) ?? models[0];
  const cm: ConfusionMatrix | null = useMemo(() => {
    if (!model) return null;
    const list = model.folds.filter(f => fold === 'sum' || f.fold === fold).map(f => f.confusion_matrices?.[split]).filter(Boolean) as ConfusionMatrix[];
    if (!list.length) return null;
    // summing counts over folds is a display of the logged counts, not a new metric
    const s = list.reduce((a, m) => ({ tn: a.tn + m.tn, fp: a.fp + m.fp, fn: a.fn + m.fn, tp: a.tp + m.tp }), { tn: 0, fp: 0, fn: 0, tp: 0 });
    return { labels: ['normal', 'abnormal'], matrix: [[s.tn, s.fp], [s.fn, s.tp]], ...s, total: s.tn + s.fp + s.fn + s.tp };
  }, [model, fold, split]);
  if (!model) return null;
  return <div className="cm">
    <div className="cm-sel">
      <label>{t('bench.cmModel')} <select value={model.model_key} onChange={e => setKey(e.target.value)}>{models.map(m => <option key={m.model_key} value={m.model_key}>{modelName(t, m.model_key, m.display_name, true)}</option>)}</select></label>
      <label>{t('bench.cmFold')} <select value={String(fold)} onChange={e => setFold(e.target.value === 'sum' ? 'sum' : Number(e.target.value))}><option value="sum">{t('bench.cmSum')}</option>{model.folds.map(f => <option key={f.fold} value={f.fold}>{t('bench.cmFold')} {f.fold}</option>)}</select></label>
      <label>{t('bench.cmSplit')} <select value={split} onChange={e => setSplit(e.target.value as 'test' | 'train')}><option value="test">{t('split.test')}</option><option value="train">{t('split.train')}</option></select></label>
    </div>
    {cm ? <table className="cm-table" data-testid="confusion-matrix"><thead><tr><td /><th scope="col">{t('bench.cmPredNormal')}</th><th scope="col">{t('bench.cmPredAbnormal')}</th></tr></thead>
      <tbody><tr><th scope="row">{t('bench.cmTrueNormal')}</th><td className="n ok">{cm.tn}</td><td className="n">{cm.fp}</td></tr>
        <tr><th scope="row">{t('bench.cmTrueAbnormal')}</th><td className="n">{cm.fn}</td><td className="n ok">{cm.tp}</td></tr></tbody></table>
      : <p className="muted">{model.confusion_matrix_warning ?? t('bench.cmMissing')}</p>}
    <p className="caption">{t('bench.cmNote')}</p>
  </div>;
}

export function BenchmarkView({ models, warning, loading }: { models: BenchmarkModel[]; warning?: string; loading?: boolean }) {
  const { t } = useI18n();
  return <div className="page">
    <section className="pnl"><h2 className="h"><span>{t('bench.eyebrow')}</span></h2><h1 className="page-title">{t('bench.title')}</h1><p className="desc">{t('bench.desc')}</p></section>
    {warning && <div className="alert warning" role="alert">{warning}</div>}
    {loading && !models.length && <div className="loading" role="status">{t('bench.loading')}</div>}
    {models.length > 0 && <>
      <section className="pnl"><h2 className="h"><span>{t('bench.table')}</span></h2><div className="ts"><table data-testid="benchmark-table">
        <thead><tr><th>{t('col.model')}</th>{TABLE.map(k => <th key={k} className="r">{t('metric.testPrefix', { m: t(`metric.${k}` as Key) })}</th>)}<th className="r">{t('metric.trainPrefix', { m: t('metric.balanced_accuracy') })}</th></tr></thead>
        <tbody>{models.map(m => <tr key={m.model_key} data-model-key={m.model_key}><td>{modelName(t, m.model_key, m.display_name)}</td>{TABLE.map(k => <td key={k} className="n">{fmt(m.mean.test[k], m.std.test[k])}</td>)}<td className="n">{fmt(m.mean.train.balanced_accuracy, m.std.train.balanced_accuracy)}</td></tr>)}</tbody></table></div></section>
      <section className="pnl"><h2 className="h"><span>{t('bench.grouped')}</span></h2><GroupedBars models={models} /></section>
      <div className="two">
        <section className="pnl"><h2 className="h"><span>{t('bench.tvt')}</span></h2><TrainTest models={models} /></section>
        <section className="pnl"><h2 className="h"><span>{t('bench.heatmap')}</span></h2><Heatmap models={models} /><p className="caption">{t('bench.heatmapNote')}</p></section>
      </div>
      <section className="pnl"><h2 className="h"><span>{t('bench.cm')}</span></h2><Confusion models={models} /></section>
      <section className="pnl"><h2 className="h"><span>{t('bench.links')}</span></h2><div className="link-grid">{models.map(m => <div className="link-card" key={m.model_key} data-chart-key={m.model_key}>
        <b>{modelName(t, m.model_key, m.display_name)}</b>
        <a className="btn sm cv-button" href={m.benchmark_run_url} target="_blank" rel="noreferrer">{t('bench.cvButton')}</a>
        <a href={m.train_vs_test_url} target="_blank" rel="noreferrer">{t('bench.tvtFigure')}</a></div>)}</div></section>
    </>}
  </div>;
}
