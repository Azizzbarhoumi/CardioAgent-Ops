import { describe, expect, it } from 'vitest';
import { axisPosition, DISTANCE_CUTOFF, distanceToThreshold, foldFromVersion } from './display';

describe('distance-to-threshold rule (boundaries)', () => {
  const svm = DISTANCE_CUTOFF.decision_score, prob = DISTANCE_CUTOFF.probability;
  it('SVM uses the absolute margin, inclusive at the cut-off', () => {
    expect(distanceToThreshold(-svm, 0, 'decision_score').far).toBe(true);
    expect(distanceToThreshold(svm, 0, 'decision_score').far).toBe(true);
    expect(distanceToThreshold(-(svm - 1e-6), 0, 'decision_score').far).toBe(false);
    expect(distanceToThreshold(svm - 1e-6, 0, 'decision_score').far).toBe(false);
    expect(distanceToThreshold(-2.0044, 0, 'decision_score').far).toBe(true);
    expect(distanceToThreshold(-0.6187, 0, 'decision_score').far).toBe(false);
  });
  it('probabilities use |score - threshold|, inclusive at the cut-off', () => {
    expect(distanceToThreshold(0.46 - prob, 0.46, 'probability').far).toBe(true);
    expect(distanceToThreshold(0.46 + prob, 0.46, 'probability').far).toBe(true);
    expect(distanceToThreshold(0.46 - prob + 1e-6, 0.46, 'probability').far).toBe(false);
    expect(distanceToThreshold(0.46, 0.46, 'probability').far).toBe(false);
    expect(distanceToThreshold(0.3954, 0.81, 'probability').far).toBe(true);
  });
  it('returns the cut-off from the single config constant', () => {
    expect(distanceToThreshold(0.5, 0.5, 'probability').cutoff).toBe(DISTANCE_CUTOFF.probability);
    expect(distanceToThreshold(0, 0, 'decision_score').cutoff).toBe(DISTANCE_CUTOFF.decision_score);
  });
});

describe('score axis', () => {
  it('SVM axis is centred on 0 and linear from -3 to +3', () => {
    expect(axisPosition(0, 'decision_score').pct).toBe(50);
    expect(axisPosition(-3, 'decision_score').pct).toBe(0);
    expect(axisPosition(1.5, 'decision_score').pct).toBe(75);
    expect(axisPosition(-2.0044, 'decision_score').pct).toBeCloseTo((-2.0044 + 3) / 6 * 100, 9);
    expect(axisPosition(5, 'decision_score')).toEqual({ pct: 100, clamped: 'high' });
  });
  it('probability axis goes from 0 to 1', () => {
    expect(axisPosition(0.3954, 'probability').pct).toBeCloseTo(39.54, 9);
    expect(axisPosition(0.81, 'probability').pct).toBeCloseTo(81, 9);
    expect(axisPosition(0, 'probability').clamped).toBeNull();
  });
});

it('readable model label comes from the app fold, raw version kept elsewhere', () => {
  expect(foldFromVersion({ model_version: 'v2-fold3', benchmark_summary: null })).toBe(3);
  expect(foldFromVersion({ model_version: 'weird', benchmark_summary: null })).toBeNull();
});
