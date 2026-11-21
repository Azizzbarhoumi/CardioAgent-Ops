import type { ModelInfo } from '../types';

const METRICS = [['Recall on abnormal', 'mean_test_recall'], ['Accuracy', 'mean_test_accuracy'], ['Balanced accuracy', 'mean_test_balanced_accuracy'], ['F1', 'mean_test_f1']] as const;
const symbol = (key: string) => key.startsWith('svm') ? 'S' : key === 'cnn_custom' ? 'C' : key === 'vgg16_transfer' ? 'V' : 'T';

export function ModelsInfoPanel({ models }: { models: ModelInfo[] }) {
  return <div className="info-grid">{models.map(model => <article className="panel info-card" key={model.key}><div className="info-card-heading"><span className="model-symbol">{symbol(model.key)}</span><div><h3>{model.name}</h3><span>{model.framework}</span></div><i className={`dot ${model.loaded ? 'on' : 'off'}`} /></div><dl><dt>Input</dt><dd>{model.input}</dd><dt>Output</dt><dd>{model.score_type === 'decision_score' ? 'Decision score (positive means abnormal)' : 'Probability of abnormal'} · threshold {model.threshold.toFixed(4)}</dd><dt>Label rule</dt><dd>{model.label_rule}</dd><dt>Labels</dt><dd>0 = Normal · 1 = Abnormal</dd><dt>App model</dt><dd>fold {model.app_fold} (best validation balanced accuracy)</dd></dl>{Object.keys(model.metrics ?? {}).length > 0 && <div className="model-metrics">{METRICS.map(([label, key]) => model.metrics[key] != null ? <div key={key}><span>{label} (3-fold test)</span><strong>{Number(model.metrics[key]).toFixed(3)}</strong></div> : null)}</div>}</article>)}</div>;
}
