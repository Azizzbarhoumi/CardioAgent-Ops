import type { ModelInfo } from '../types';

const METRICS = [
  ['Accuracy', ['mean_accuracy', 'accuracy']],
  ['Recall on abnormal', ['mean_recall', 'recall']],
  ['Balanced accuracy', ['mean_balanced_accuracy', 'balanced_accuracy']],
  ['F1', ['mean_f1', 'f1']],
] as const;

export function ModelsInfoPanel({ models }: { models: ModelInfo[] }) {
  return <div className="info-grid">{models.map(model => <article className="panel info-card" key={model.key}><div className="info-card-heading"><span className="model-symbol">{model.key.startsWith('svm') ? 'S' : model.key === 'cnn_custom' ? 'C' : 'T'}</span><div><h3>{model.name}{model.kernel ? ` · ${model.kernel}` : ''}</h3><span>{model.type}</span></div><i className={`dot ${model.loaded ? 'on' : 'off'}`} /></div><dl><dt>Input</dt><dd>{model.input_features ? `${model.input_features.length} features: ${model.input_features.join(', ')}` : `${model.input_shape?.join(' × ')} image`}</dd><dt>Output</dt><dd>{model.score_type === 'decision_score' ? 'Decision score (positive means abnormal)' : 'Probability of abnormal'} · threshold {model.threshold.toFixed(2)}</dd><dt>Labels</dt><dd>0 = Normal · 1 = Abnormal</dd></dl>{Object.keys(model.metrics).length > 0 && <div className="model-metrics">{METRICS.map(([label, keys]) => { const key = keys.find(name => model.metrics[name] != null); return key ? <div key={key}><span>{label}</span><strong>{Number(model.metrics[key]).toFixed(3)}</strong></div> : null; })}</div>}<div className="source-line">Source: {model.source}</div></article>)}</div>;
}
