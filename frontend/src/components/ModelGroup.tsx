import type { ModelGroupData } from '../types';
import { ModelCard } from './ModelCard';
export function ModelGroup({ group, analysisUrl }: { group: ModelGroupData; analysisUrl?: string | null }) {
  return <section className="model-group"><div className="group-title"><div><h3>{group.name}</h3><span>{group.models.length === 2 ? 'One algorithm · two kernel choices' : 'Independent model result'}</span></div><span className="group-count">{group.models.length} {group.models.length === 1 ? 'result' : 'results'}</span></div><div className={`model-grid ${group.models.length > 1 ? 'two-up' : ''}`}>{group.models.map(model => <div className="model-slot" key={model.model_key}>{model.model_key.startsWith('svm_') && <div className="kernel-label">{model.model_key === 'svm_linear' ? 'LINEAR KERNEL' : 'RBF KERNEL'}</div>}<ModelCard result={model} analysisUrl={analysisUrl} /></div>)}</div></section>;
}
