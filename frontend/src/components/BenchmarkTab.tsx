import type { BenchmarkModel } from '../types';

const COLUMNS = [['recall', 'Recall (abnormal)'], ['accuracy', 'Accuracy'], ['balanced_accuracy', 'Balanced acc.'], ['f1', 'F1'], ['roc_auc', 'ROC AUC'], ['specificity', 'Specificity']] as const;
const CHART = [['accuracy', 'Acc.'], ['balanced_accuracy', 'Bal. acc.'], ['recall', 'Recall'], ['f1', 'F1']] as const;
export const fmt = (m?: number | null, s?: number | null) => m == null ? '–' : `${m.toFixed(3)} ± ${(s ?? 0).toFixed(3)}`;

function TrainVsTest({ model }: { model: BenchmarkModel }) {
  const w = 320, h = 150, pad = 26, group = (w - pad) / CHART.length, bar = group / 3.2, span = h - 10;
  return <svg className="tvt-chart" viewBox={`0 0 ${w} ${h + 22}`} role="img" aria-label={`${model.display_name} train vs test`}>
    {[0, 0.5, 1].map(v => <g key={v}><line x1={pad} x2={w} y1={h - v * span} y2={h - v * span} className="grid" /><text x={2} y={h - v * span + 3}>{v}</text></g>)}
    {CHART.map(([key, label], i) => {
      const x = pad + i * group + group / 2;
      return <g key={key}>{(['train', 'test'] as const).map((split, j) => {
        const v = model.mean[split][key] ?? 0, sd = model.std[split][key] ?? 0, bx = x - bar + j * bar, top = h - v * span;
        return <g key={split}><rect className={split} x={bx} y={top} width={bar - 2} height={h - top}><title>{`${split} ${label}: ${fmt(v, sd)}`}</title></rect>
          <line className="err" x1={bx + bar / 2 - 1} x2={bx + bar / 2 - 1} y1={top - sd * span} y2={top + sd * span} /></g>;
      })}<text x={x} y={h + 14} textAnchor="middle">{label}</text></g>;
    })}
  </svg>;
}

export function BenchmarkTab({ models, warning }: { models: BenchmarkModel[]; warning?: string }) {
  return <div className="page-content">
    <div className="section-heading"><div><div className="eyebrow">PATIENT-GROUPED 3-FOLD CROSS-VALIDATION</div><h2>Benchmark</h2><p>Same 3 folds for every model. Threshold and early stopping are chosen on a validation split inside each training part. Mean ± std over the 3 test folds. Each model is reported separately.</p></div></div>
    {warning && <div className="alert warning">{warning}</div>}
    <div className="table-scroll panel"><table data-testid="benchmark-table"><thead><tr><th>Model</th>{COLUMNS.map(([, l]) => <th key={l}>Test {l}</th>)}<th>Train recall</th><th>Train bal. acc.</th><th>MLflow</th></tr></thead>
      <tbody>{models.map(m => <tr key={m.model_key} data-model-key={m.model_key}><td>{m.display_name}</td>{COLUMNS.map(([k]) => <td key={k}>{fmt(m.mean.test[k], m.std.test[k])}</td>)}<td>{fmt(m.mean.train.recall, m.std.train.recall)}</td><td>{fmt(m.mean.train.balanced_accuracy, m.std.train.balanced_accuracy)}</td><td><a className="small-link" href={m.benchmark_run_url} target="_blank" rel="noreferrer">3-fold results ↗</a></td></tr>)}</tbody></table></div>
    <div className="info-grid">{models.map(m => <article className="panel info-card" key={m.model_key} data-chart-key={m.model_key}><div className="chart-heading"><h3>{m.display_name}: train vs test</h3><span><i className="swatch train" />train <i className="swatch test" />test</span></div><TrainVsTest model={m} />
      <div className="card-links"><a className="button secondary cv-button" href={m.benchmark_run_url} target="_blank" rel="noreferrer">3-fold results in MLflow ↗</a><a className="small-link" href={m.train_vs_test_url} target="_blank" rel="noreferrer">train vs test figure</a></div></article>)}</div>
  </div>;
}
