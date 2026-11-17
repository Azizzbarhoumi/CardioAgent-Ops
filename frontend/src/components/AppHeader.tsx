import type { Health } from '../types';
import { StatusPill } from './StatusPill';

export function AppHeader({ health }: { health: Health | null }) {
  return <header className="app-header"><div><div className="eyebrow">CARDIOAGENT · MODEL LAB</div><h1>Heart Sound Model Lab</h1><p>Explore independent analyses of a phonocardiogram recording.</p></div><div className="status-stack"><StatusPill label={health?.models_loaded ? 'API and models ready' : 'API unavailable'} healthy={Boolean(health?.models_loaded)} /><StatusPill label={health?.mlflow_reachable ? 'MLflow connected' : 'MLflow unavailable'} healthy={Boolean(health?.mlflow_reachable)} /></div></header>;
}
