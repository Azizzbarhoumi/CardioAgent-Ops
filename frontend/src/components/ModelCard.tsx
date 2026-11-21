import type { ModelResult } from '../types';
import { ScoreBar } from './ScoreBar';

const shortVersion = (version: string) => version.length > 34 ? `${version.slice(0, 31)}…` : version;
const pct = (value: number) => (value * 100).toFixed(1);
export function scoreName(type: string) {
  return type === 'decision_score' ? 'Decision score' : type === 'probability' ? 'Probability of abnormal' : 'Class confidence';
}
export function ModelCard({ result, analysisUrl }: { result: ModelResult; analysisUrl?: string | null }) {
  const abnormal = result.prediction === 1;
  const summary = result.benchmark_summary;
  const rule = result.label_rule ?? (result.score_type === 'decision_score' ? 'decision_score > 0' : 'P(abnormal) >= threshold');
  return <article className="model-card" data-model-key={result.model_key}><div className="model-card-top"><span>{result.model_name}</span><span className={`prediction ${abnormal ? 'abnormal' : 'normal'}`}>{result.prediction_label}</span></div>
    <div className="score-line"><span>{scoreName(result.score_type)}</span><strong data-testid="score">{result.score.toFixed(4)}</strong></div>
    <ScoreBar score={result.score} type={result.score_type} threshold={result.threshold} />
    <div className="metric-grid"><div><span>Threshold</span><strong data-testid="threshold">{result.threshold.toFixed(4)}</strong></div><div><span>Inference</span><strong>{result.latency_ms.toFixed(2)} ms</strong></div></div>
    {summary && <div className="cv-line" data-testid="cv-summary">3-fold test recall {pct(summary.mean_test_recall)} ± {pct(summary.std_test_recall)}% · accuracy {pct(summary.mean_test_accuracy)} ± {pct(summary.std_test_accuracy)}%</div>}
    <div className="card-links">
      {result.benchmark_run_url
        ? <a className="button secondary cv-button" href={result.benchmark_run_url} target="_blank" rel="noreferrer">3-fold results in MLflow ↗</a>
        : <span className="muted">{result.benchmark_warning ?? '3-fold results unavailable'}</span>}
      {result.benchmark_train_vs_test_url && <a className="small-link" href={result.benchmark_train_vs_test_url} target="_blank" rel="noreferrer">train vs test</a>}
      {analysisUrl && <a className="small-link" href={analysisUrl} target="_blank" rel="noreferrer">analysis in MLflow</a>}
    </div>
    <div className="version-line" title={result.model_version}>Version: {shortVersion(result.model_version)}</div>
    <details className="model-details"><summary>Model details</summary><dl><dt>Full source</dt><dd>{result.model_version}</dd><dt>Label rule</dt><dd>{rule}</dd><dt>Label mapping</dt><dd>0 = Normal · 1 = Abnormal</dd>{result.model_run_id && <><dt>Training run</dt><dd>{result.model_run_id}</dd></>}{(result.benchmark_folds_urls ?? []).length > 0 && <><dt>Fold runs</dt><dd>{(result.benchmark_folds_urls ?? []).map((url, i) => <a key={url} className="small-link" href={url} target="_blank" rel="noreferrer">fold {i + 1} </a>)}</dd></>}</dl></details>
  </article>;
}
