// Display helpers. Nothing here computes a prediction or a metric: values come from the API and are only
// positioned on a scale or formatted.
import type { ModelResult } from '../types';

/** Distance-to-threshold rule (item 14). ONE config constant, in the score's own units.
 *  decision_score (SVM): far when |score - threshold| >= 1.0  (threshold is 0, so this is the absolute margin)
 *  probability:          far when |score - threshold| >= 0.15
 *  This describes how far the score sits from the threshold. It is NOT a probability of being correct
 *  and NOT a confidence level. */
export const DISTANCE_CUTOFF = { decision_score: 1.0, probability: 0.15 } as const;

export type ScoreKind = 'decision_score' | 'probability';
export const scoreKind = (type?: string | null): ScoreKind => (type === 'decision_score' ? 'decision_score' : 'probability');

export function distanceToThreshold(score: number, threshold: number, type?: string | null) {
  const kind = scoreKind(type);
  const distance = Math.abs(score - threshold);
  // 1e-9 tolerance so a score exactly at the cut-off (e.g. 0.46 - 0.15 in binary floating point) counts as far
  return { kind, distance, cutoff: DISTANCE_CUTOFF[kind], far: distance >= DISTANCE_CUTOFF[kind] - 1e-9 };
}

/** Decision-score axis: fixed, centred on 0, from -SVM_AXIS to +SVM_AXIS. Probability axis: 0 to 1. */
export const SVM_AXIS = 3;
export function axisDomain(type?: string | null): [number, number] {
  return scoreKind(type) === 'decision_score' ? [-SVM_AXIS, SVM_AXIS] : [0, 1];
}
/** Position (0-100 %) of a value on the model's axis, clamped; `clamped` tells the UI to show an overflow mark. */
export function axisPosition(value: number, type?: string | null) {
  const [lo, hi] = axisDomain(type);
  const raw = ((value - lo) / (hi - lo)) * 100;
  return { pct: Math.min(100, Math.max(0, raw)), clamped: raw < 0 ? 'low' : raw > 100 ? 'high' : null };
}
export function axisTicks(type?: string | null): number[] {
  return scoreKind(type) === 'decision_score' ? [-3, -2, -1, 0, 1, 2, 3] : [0, 0.25, 0.5, 0.75, 1];
}

export const fmtScore = (v: number) => v.toFixed(4);
export const fmtThreshold = (v: number) => v.toFixed(4);
export const fmtMs = (v: number) => `${v.toFixed(2)} ms`;
export const fmtPct = (v?: number | null) => (v == null ? '–' : (v * 100).toFixed(1));

/** "v2-fold3" -> 3 ; anything else -> null (raw value stays in Model details). */
export function foldFromVersion(r: Pick<ModelResult, 'model_version' | 'benchmark_summary'>): number | null {
  const app = r.benchmark_summary?.app_fold;
  if (app != null && !Number.isNaN(Number(app))) return Number(app);
  const m = /fold\s*(\d+)/i.exec(r.model_version ?? '');
  return m ? Number(m[1]) : null;
}

export const MODEL_ORDER = ['svm_linear', 'svm_rbf', 'decision_tree', 'cnn_custom', 'vgg16_transfer'];
const orderOf = (key: string) => { const i = MODEL_ORDER.indexOf(key); return i < 0 ? 99 : i; };
export const sortResults = (rs: ModelResult[]) => [...rs].sort((a, b) => orderOf(a.model_key) - orderOf(b.model_key));
