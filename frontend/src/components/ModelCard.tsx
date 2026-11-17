import type { ModelResult } from '../types';
import { ScoreBar } from './ScoreBar';

const shortVersion = (version: string) => version.length > 34 ? `${version.slice(0, 31)}…` : version;
export function ModelCard({ result }: { result: ModelResult }) {
  const abnormal = result.prediction === 1;
  const scoreName = result.score_type === 'decision_score' ? 'Decision score' : result.score_type === 'probability' ? 'Probability of abnormal' : 'Class confidence';
  return <article className="model-card" data-model-key={result.model_key}><div className="model-card-top"><span>{result.model_name}</span><span className={`prediction ${abnormal ? 'abnormal' : 'normal'}`}>{result.prediction_label}</span></div>
    <div className="score-line"><span>{scoreName}</span><strong>{result.score.toFixed(4)}</strong></div>
    <ScoreBar score={result.score} type={result.score_type} threshold={result.threshold} />
    <div className="metric-grid"><div><span>Threshold</span><strong>{result.threshold.toFixed(2)}</strong></div><div><span>Inference</span><strong>{result.latency_ms.toFixed(2)} ms</strong></div></div>
    <div className="version-line" title={result.model_version}>Version: {shortVersion(result.model_version)}</div>
    <details className="model-details"><summary>Model details</summary><dl><dt>Full source</dt><dd>{result.model_version}</dd><dt>Label mapping</dt><dd>0 = Normal · 1 = Abnormal</dd>{result.model_run_id && <><dt>Training run</dt><dd>{result.model_run_id}</dd></>}</dl></details>
  </article>;
}
