export function StatusPill({ label, healthy }: { label: string; healthy: boolean }) {
  return <span className={`status-pill ${healthy ? 'healthy' : 'unhealthy'}`}><i />{label}</span>;
}
