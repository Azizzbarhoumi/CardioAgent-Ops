export function ScoreBar({ score, type, threshold }: { score: number; type: string; threshold: number }) {
  const decision = type === 'decision_score';
  const extent = decision
    ? Math.min(48, Math.abs((Math.atan(score) / Math.PI) * 100))
    : Math.max(0, Math.min(100, score * 100));
  const left = decision ? (score < 0 ? 50 - extent : 50) : 0;
  const width = decision ? extent : Math.max(0, Math.min(100, score * 100));
  const marker = decision ? 50 : Math.max(0, Math.min(100, threshold * 100));
  const typeLabel = decision ? 'decision score' : 'probability';
  return <div className={`score-track ${decision ? 'decision' : ''}`} role="img" aria-label={`${typeLabel} ${score.toFixed(4)}, threshold ${threshold}`}><span className="score-fill" style={{ left: `${left}%`, width: `${width}%` }} /><span className="score-marker" style={{ left: `${marker}%` }} /></div>;
}
