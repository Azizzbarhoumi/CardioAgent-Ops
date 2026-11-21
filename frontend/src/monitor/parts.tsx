// Small presentational parts of the monitor UI. They only display API values.
import type { ModelResult } from '../types';
import { labelText, useI18n } from '../i18n';
import { axisPosition, axisTicks, distanceToThreshold, fmtPct, fmtScore, fmtThreshold, scoreKind } from '../lib/display';

export function PredictionBadge({ result, className = 'prediction' }: { result: Pick<ModelResult, 'prediction' | 'prediction_label'>; className?: string }) {
  const { t, lang } = useI18n();
  const abnormal = result.prediction === 1;
  return <span className={`${className} badge ${abnormal ? 'abn' : 'nor'}`} data-label={abnormal ? 'abnormal' : 'normal'}>
    {abnormal
      ? <svg viewBox="0 0 16 16" aria-hidden="true" data-icon="warning"><path d="M8 1.5 15 14H1z" fill="none" stroke="currentColor" strokeWidth="1.6" /><path d="M8 6v4M8 11.6v.4" stroke="currentColor" strokeWidth="1.6" /></svg>
      : <svg viewBox="0 0 16 16" aria-hidden="true" data-icon="check"><circle cx="8" cy="8" r="6.6" fill="none" stroke="currentColor" strokeWidth="1.6" /><path d="m5 8.2 2.1 2.1L11.2 6" fill="none" stroke="currentColor" strokeWidth="1.7" /></svg>}
    {labelText(t, result.prediction, result.prediction_label, lang)}
  </span>;
}

/** Bar with the score marker and a fixed threshold line on the model's own axis. */
export function ScoreScale({ score, threshold, type }: { score: number; threshold: number; type?: string | null }) {
  const { t } = useI18n();
  const kind = scoreKind(type);
  const s = axisPosition(score, kind), th = axisPosition(threshold, kind);
  return <div className="scale">
    <div className="scale-track" role="img" aria-label={t('axis.aria', { type: t(`axis.${kind}`), score: fmtScore(score), thr: fmtThreshold(threshold) })}>
      <span className="scale-normal" style={{ width: `${th.pct}%` }} />
      <span className="scale-abnormal" style={{ left: `${th.pct}%` }} />
      <span className="scale-threshold" data-testid="threshold-line" style={{ left: `${th.pct}%` }} />
      <span className={`scale-marker ${s.clamped ? 'clamped' : ''}`} data-testid="score-marker" data-pos={s.pct.toFixed(3)} style={{ left: `${s.pct}%` }} />
    </div>
    <div className="scale-ticks" aria-hidden="true">{axisTicks(kind).map(v => <span key={v} style={{ left: `${axisPosition(v, kind).pct}%` }}>{kind === 'decision_score' && v > 0 ? `+${v}` : v}</span>)}</div>
    <div className="scale-sides"><span>{t('axis.normalSide')}</span>{s.clamped && <span className="offscale">{t('axis.offscale')}</span>}<span>{t('axis.abnormalSide')}</span></div>
  </div>;
}

export function DistanceTag({ score, threshold, type }: { score: number; threshold: number; type?: string | null }) {
  const { t } = useI18n();
  const d = distanceToThreshold(score, threshold, type);
  const rule = `${t(`dist.rule.${d.kind}`, { cut: d.cutoff, d: d.distance.toFixed(4) })} ${t('dist.disclaimer')}`;
  return <span className={`dist ${d.far ? 'far' : 'close'}`} title={rule} data-testid="distance-tag" tabIndex={0} aria-label={`${t(d.far ? 'dist.far' : 'dist.close')}. ${rule}`}>
    {t(d.far ? 'dist.far' : 'dist.close')}
  </span>;
}

function RelRow({ label, mean, std }: { label: string; mean?: number | null; std?: number | null }) {
  const { t } = useI18n();
  if (mean == null) return <div className="rel-row"><span>{label}</span><span className="rel-bar" /><b>{t('rel.unavailable')}</b></div>;
  const sd = std ?? 0;
  return <div className="rel-row"><span>{label}</span>
    <span className="rel-bar" aria-hidden="true"><i className="rel-sd" style={{ left: `${Math.max(0, mean - sd) * 100}%`, width: `${Math.min(1, sd * 2) * 100}%` }} /><i className="rel-pt" style={{ left: `${mean * 100}%` }} /></span>
    <b>{fmtPct(mean)} ± {fmtPct(sd)}%</b></div>;
}

export function Reliability({ result }: { result: ModelResult }) {
  const { t } = useI18n();
  const s = result.benchmark_summary;
  return <div className="rel" data-testid="reliability">
    <div className="rel-title">{t('rel.title')} <span>({t('rel.note')})</span></div>
    <RelRow label={t('rel.recall')} mean={s?.mean_test_recall} std={s?.std_test_recall} />
    <RelRow label={t('rel.specificity')} mean={s?.mean_test_specificity} std={s?.std_test_specificity} />
    <RelRow label={t('rel.balanced')} mean={s?.mean_test_balanced_accuracy} std={s?.std_test_balanced_accuracy} />
  </div>;
}
