import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { ModelGroup } from './ModelGroup';
import type { ModelResult } from '../types';

const results: ModelResult[] = [
  {
    model_name: 'SVM (linear)',
    model_key: 'svm_linear',
    model_version: 'model-id-123456789012345678901234567890123',
    prediction: 0,
    prediction_label: 'Normal',
    confidence_or_score: -0.42,
    confidence_type: 'decision_score',
    score: -0.42,
    score_type: 'decision_score',
    threshold: 0,
    latency_ms: 4.25,
    kernel: 'linear',
  },
  {
    model_name: 'SVM (RBF)',
    model_key: 'svm_rbf',
    model_version: 'rbf-model-version',
    prediction: 1,
    prediction_label: 'Abnormal',
    confidence_or_score: 0.42,
    confidence_type: 'decision_score',
    score: 0.42,
    score_type: 'decision_score',
    threshold: 0,
    latency_ms: 2.5,
    kernel: 'rbf',
  },
  {
    model_name: 'Decision Tree',
    model_key: 'decision_tree',
    model_version: 'tree-model-version',
    prediction: 0,
    prediction_label: 'Normal',
    confidence_or_score: 0.8224,
    confidence_type: 'probability',
    score: 0.1776,
    score_type: 'probability',
    threshold: 0.5,
    latency_ms: 1.75,
  },
  {
    model_name: 'CNN',
    model_key: 'cnn_custom',
    model_version: 'cnn-model-version',
    prediction: 1,
    prediction_label: 'Abnormal',
    confidence_or_score: 0.6779,
    confidence_type: 'probability',
    score: 0.6779,
    score_type: 'probability',
    threshold: 0.5,
    latency_ms: 6.1,
  },
];

describe('model result components', () => {
  it.each(results)('$model_key displays its prediction and score contract', result => {
    const { container } = render(<ModelCard result={result} />);
    const card = container.querySelector(`[data-model-key="${result.model_key}"]`);

    expect(card?.querySelector('.prediction')).toHaveTextContent(result.prediction_label);
    expect(card?.querySelector('.score-line span')).toHaveTextContent(
      result.score_type === 'decision_score' ? 'Decision score' : 'Probability of abnormal',
    );
    expect(card?.querySelector('.score-line strong')).toHaveTextContent(result.score.toFixed(4));
    expect(card?.querySelector('.metric-grid')).toHaveTextContent(result.threshold.toFixed(2));
    expect(card?.querySelector('.metric-grid')).toHaveTextContent(`${result.latency_ms.toFixed(2)} ms`);
    expect(container.querySelector('details')?.textContent).toContain(result.model_version);
  });

  it('renders the two SVM kernel results in one group', () => {
    render(
      <ModelGroup
        group={{ key: 'svm', name: 'SVM (2 kernels)', models: results.slice(0, 2) }}
      />,
    );

    expect(screen.getByText('SVM (2 kernels)')).toBeInTheDocument();
    expect(screen.getByText('LINEAR KERNEL')).toBeInTheDocument();
    expect(screen.getByText('RBF KERNEL')).toBeInTheDocument();
  });
});
