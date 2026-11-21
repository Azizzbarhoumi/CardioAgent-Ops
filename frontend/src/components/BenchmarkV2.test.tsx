import { describe, expect, it } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { BenchmarkTab } from './BenchmarkTab';
import type { BenchmarkModel, ModelResult } from '../types';

const base = 'http://localhost:5000/#/experiments/1/runs';
const result = (key: string, name: string, prediction: 0 | 1, score: number, scoreType: string, threshold: number): ModelResult => ({
  model_name: name, model_key: key, model_version: 'v2-fold3', prediction, prediction_label: prediction ? 'Abnormal' : 'Normal',
  confidence_or_score: score, confidence_type: scoreType, score, score_type: scoreType, threshold, latency_ms: 3,
  label_rule: scoreType === 'decision_score' ? 'decision_score > 0' : 'P(abnormal) >= threshold (chosen on validation)',
  benchmark_run_url: `${base}/parent-${key}`, benchmark_train_vs_test_url: `${base}/parent-${key}/artifacts/figures/train_vs_test.png`,
  benchmark_folds_urls: [1, 2, 3].map(i => `${base}/child-${key}-${i}`),
  benchmark_summary: { n_folds: 3, mean_test_recall: 0.778, std_test_recall: 0.038, mean_test_accuracy: 0.804, std_test_accuracy: 0.019, parent_run_id: `parent-${key}` },
});
const cards = [
  result('svm_rbf', 'SVM (RBF)', 0, -0.6012, 'decision_score', 0),
  result('decision_tree', 'Decision Tree', 1, 1.0, 'probability', 0.8404),
  result('cnn_custom', 'CNN', 0, 0.3972, 'probability', 0.8056),
  result('vgg16_transfer', 'Transfer Learning (VGG16)', 0, 0.1116, 'probability', 0.4564),
];

describe('v2 model cards', () => {
  it.each(cards)('$model_key shows label, score type, threshold and the MLflow buttons', r => {
    const { container } = render(<ModelCard result={r} analysisUrl="http://localhost:5000/#/experiments/2/runs/analysis" />);
    const card = within(container.querySelector(`[data-model-key="${r.model_key}"]`) as HTMLElement);
    expect(card.getByText(r.prediction_label)).toBeInTheDocument();
    expect(card.getByText(r.score_type === 'decision_score' ? 'Decision score' : 'Probability of abnormal')).toBeInTheDocument();
    expect(card.getByTestId('score')).toHaveTextContent(r.score.toFixed(4));
    expect(card.getByTestId('threshold')).toHaveTextContent(r.threshold.toFixed(4));
    expect(card.getByRole('link', { name: /3-fold results in MLflow/ })).toHaveAttribute('href', r.benchmark_run_url!);
    expect(card.getByRole('link', { name: 'train vs test' })).toHaveAttribute('href', r.benchmark_train_vs_test_url!);
    expect(card.getByRole('link', { name: 'analysis in MLflow' })).toHaveAttribute('href', 'http://localhost:5000/#/experiments/2/runs/analysis');
    expect(card.getByTestId('cv-summary')).toHaveTextContent('3-fold test recall 77.8 ± 3.8%');
  });

  it('says why the 3-fold link is missing when MLflow is unavailable', () => {
    render(<ModelCard result={{ ...cards[0], benchmark_run_url: null, benchmark_warning: 'MLflow benchmark unavailable: timeout' }} />);
    expect(screen.getByText('MLflow benchmark unavailable: timeout')).toBeInTheDocument();
  });
});

const metrics = (v: number) => ({ accuracy: v, balanced_accuracy: v, precision: v, recall: v, f1: v, macro_f1: v, roc_auc: v, specificity: v });
const bench: BenchmarkModel[] = ['svm_linear', 'vgg16_transfer'].map((key, i) => ({
  model_key: key, display_name: key === 'svm_linear' ? 'SVM (linear)' : 'Transfer Learning (VGG16)', parent_run_id: `p${i}`,
  benchmark_run_url: `${base}/p${i}`, train_vs_test_url: `${base}/p${i}/artifacts/figures/train_vs_test.png`, benchmark_folds_urls: [],
  mean: { train: metrics(0.8), validation: metrics(0.79), test: metrics(0.7 + i / 10) }, std: { train: metrics(0.01), validation: metrics(0.01), test: metrics(0.02) },
  folds: [],
}));

describe('Benchmark tab', () => {
  it('renders the models x mean±std table, one train-vs-test chart per model and the links', () => {
    const { container } = render(<BenchmarkTab models={bench} />);
    const table = screen.getByTestId('benchmark-table');
    expect(within(table).getByText('SVM (linear)')).toBeInTheDocument();
    expect(within(table).getAllByText('0.800 ± 0.020').length).toBeGreaterThan(0);
    expect(container.querySelectorAll('svg.tvt-chart')).toHaveLength(2);
    expect(container.querySelector('[data-chart-key="vgg16_transfer"] a.cv-button')).toHaveAttribute('href', `${base}/p1`);
    expect(within(table).getAllByRole('link').map(a => a.getAttribute('href'))).toEqual([`${base}/p0`, `${base}/p1`]);
  });

  it('never shows an aggregated verdict', () => {
    const { container } = render(<><BenchmarkTab models={bench} />{cards.map(c => <ModelCard key={c.model_key} result={c} />)}</>);
    expect(container.textContent?.toLowerCase()).not.toMatch(/combined|overall verdict|majority|ensemble|final diagnosis/);
  });
});
