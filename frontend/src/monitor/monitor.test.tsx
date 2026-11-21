import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import App from '../App';
import { en, fr, translate } from '../i18n';
import { csvRows, toCsv, toJson, CSV_COLUMNS } from '../lib/export';
import { fmtMs, fmtPct, fmtScore, fmtThreshold } from '../lib/display';
import type { Analysis, ModelResult } from '../types';

// The realistic case from the screenshot: 5 results, all Normal.
const RAW: [string, string, 'decision_score' | 'probability', number, number, number][] = [
  ['svm_linear', 'SVM (linear)', 'decision_score', -2.0044, 0, 1.8],
  ['svm_rbf', 'SVM (RBF)', 'decision_score', -1.0741, 0, 3.1],
  ['decision_tree', 'Decision Tree', 'probability', 0.0323, 0.84, 0.9],
  ['cnn_custom', 'CNN', 'probability', 0.3954, 0.81, 42.6],
  ['vgg16_transfer', 'Transfer Learning (VGG16)', 'probability', 0.1641, 0.46, 118.4],
];
const results: ModelResult[] = RAW.map(([key, name, type, score, thr, ms], i) => ({
  model_name: name, model_key: key, model_version: 'v2-fold3', model_run_id: `run${i}`, prediction: 0, prediction_label: 'Normal',
  confidence_or_score: score, confidence_type: type, score, score_type: type, threshold: thr, latency_ms: ms,
  benchmark_run_url: `http://localhost:5000/#/experiments/1/runs/parent_${key}`,
  benchmark_train_vs_test_url: `http://localhost:5000/#/experiments/1/runs/parent_${key}/artifacts/figures/train_vs_test.png`,
  benchmark_folds_urls: [1, 2, 3].map(f => `http://localhost:5000/#/experiments/1/runs/${key}_f${f}`),
  benchmark_summary: { n_folds: 3, mean_test_recall: 0.766 + i / 100, std_test_recall: 0.027, mean_test_accuracy: 0.78, std_test_accuracy: 0.01, parent_run_id: `parent_${key}`,
    mean_test_specificity: 0.82 + i / 100, std_test_specificity: 0.023, mean_test_balanced_accuracy: 0.793 + i / 100, std_test_balanced_accuracy: 0.003, app_fold: '3' },
}));
const analysis: Analysis = {
  filename: 'pascal_normal_246.wav', duration_s: 7.42, source_sample_rate: 44100, sample_rate_after_resampling: 2000, results,
  model_groups: [], waveform_points: [[0, 0], [3.7, 0.5], [7.42, -0.2]], mel_spectrogram_png_base64: 'aGVsbG8=', waveform_png_base64: '',
  mel_spectrogram_axes: { left: 0.1, right: 0.8, top: 0.1, bottom: 0.85, time_start_s: 0, time_end_s: 7.42 },
  total_processing_ms: 812, mlflow_run_id: 'abc', mlflow_run_url: 'http://localhost:5000/#/experiments/2/runs/abc', warning: null,
};

const ok = (body: unknown) => ({ ok: true, status: 200, json: async () => body, blob: async () => new Blob(['RIFF']) }) as Response;
function fakeFetch() {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/health')) return ok({ status: 'ok', models_loaded: true, mlflow_reachable: true, loaded_models: [] });
    if (url.endsWith('/models')) return ok({ models: [] });
    if (url.endsWith('/samples')) return ok({ samples: [{ id: 'normal', label: 'normal', file: 'n.wav' }, { id: 'abnormal', label: 'abnormal', file: 'a.wav' }] });
    if (init?.method === 'POST') return ok(analysis);
    return ok({ analyses: [] });
  });
}
async function renderAnalyzed() {
  vi.stubGlobal('fetch', fakeFetch());
  const view = render(<App />);
  fireEvent.change(screen.getByLabelText('Upload WAV recording'), { target: { files: [new File(['a'], 'pascal_normal_246.wav', { type: 'audio/wav' })] } });
  fireEvent.click(screen.getByRole('button', { name: /analyze recording/i }));
  await screen.findByText('ANALYSIS COMPLETE');
  return view;
}

beforeEach(() => { try { localStorage.clear(); } catch { /* ignore */ } });
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('Analyze results (monitor UI)', () => {
  it('renders the summary table before the channel grid', async () => {
    await renderAnalyzed();
    const table = screen.getByTestId('summary-table'), grid = screen.getByTestId('channel-grid');
    expect(table.compareDocumentPosition(grid) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    const upload = screen.getByTestId('drop-zone');
    expect(upload.compareDocumentPosition(table) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it('grid has 5 cards and the SVM group holds 2', async () => {
    await renderAnalyzed();
    expect(screen.getByTestId('channel-grid').querySelectorAll('article.model-card')).toHaveLength(5);
    const svm = screen.getByTestId('svm-group');
    expect(within(svm).getByRole('heading', { name: 'SVM (2 kernels)' })).toBeInTheDocument();
    expect(svm.querySelectorAll('article.model-card')).toHaveLength(2);
  });

  it('each card shows label, score with type, threshold, reliability and the 3 MLflow links', async () => {
    await renderAnalyzed();
    for (const r of results) {
      const card = document.querySelector(`article.model-card[data-model-key="${r.model_key}"]`) as HTMLElement;
      const c = within(card);
      expect(card.querySelector('.prediction')!.textContent).toBe('Normal');
      expect(card.querySelector('.prediction svg[data-icon="check"]')).not.toBeNull();
      expect(c.getByTestId('score').textContent).toBe(fmtScore(r.score));
      expect(c.getByTestId('threshold').textContent).toBe(fmtThreshold(r.threshold));
      expect(c.getByText(r.score_type === 'decision_score' ? 'Decision score (positive = abnormal)' : 'Probability of abnormal')).toBeInTheDocument();
      const rel = c.getByTestId('reliability');
      expect(rel).toHaveTextContent('3-fold cross-validated test metrics');
      expect(rel).toHaveTextContent('not the confidence of this prediction');
      const s = r.benchmark_summary!;
      expect(rel).toHaveTextContent(`${fmtPct(s.mean_test_recall)} ± ${fmtPct(s.std_test_recall)}%`);
      expect(rel).toHaveTextContent(`${fmtPct(s.mean_test_specificity)} ± ${fmtPct(s.std_test_specificity)}%`);
      expect(rel).toHaveTextContent(`${fmtPct(s.mean_test_balanced_accuracy)} ± ${fmtPct(s.std_test_balanced_accuracy)}%`);
      expect(c.getByRole('link', { name: /3-fold results in MLflow/ })).toHaveAttribute('href', r.benchmark_run_url!);
      expect(c.getByRole('link', { name: /train vs test/ })).toHaveAttribute('href', r.benchmark_train_vs_test_url!);
      expect(c.getByRole('link', { name: /analysis in MLflow/ })).toHaveAttribute('href', analysis.mlflow_run_url!);
      expect(c.getByText('3-fold CV · fold 3 model')).toBeInTheDocument();
      expect(card.querySelector('details')).toHaveTextContent('v2-fold3');
    }
  });

  it('score marker position matches the score on the model axis', async () => {
    await renderAnalyzed();
    for (const r of results) {
      const card = document.querySelector(`[data-model-key="${r.model_key}"]`)!;
      const pos = Number(card.querySelector('[data-testid="score-marker"]')!.getAttribute('data-pos'));
      const expected = r.score_type === 'decision_score' ? (r.score + 3) / 6 * 100 : r.score * 100;
      expect(pos).toBeCloseTo(expected, 2);
      const thr = (card.querySelector('[data-testid="threshold-line"]') as HTMLElement).style.left;
      expect(parseFloat(thr)).toBeCloseTo(r.score_type === 'decision_score' ? 50 : r.threshold * 100, 6);
    }
  });

  it('summary table values equal the card values', async () => {
    await renderAnalyzed();
    for (const r of results) {
      const row = document.querySelector(`tr[data-row-key="${r.model_key}"]`)!;
      expect(row.querySelector('[data-col="score"]')!.textContent).toBe(fmtScore(r.score));
      expect(row.querySelector('[data-col="threshold"]')!.textContent).toBe(fmtThreshold(r.threshold));
    }
  });

  it('never renders a combined verdict', async () => {
    const { container } = await renderAnalyzed();
    expect(container.textContent).not.toMatch(/overall|combined verdict|majority|consensus|ensemble|agreement|\d ?\/ ?5 (models )?(say|agree)/i);
    expect(container.textContent).toMatch(/no combined result/i); // the explicit statement is allowed
  });

  it('distance tag shows its rule and is not a confidence', async () => {
    await renderAnalyzed();
    const tag = within(document.querySelector('[data-model-key="cnn_custom"]') as HTMLElement).getByTestId('distance-tag');
    expect(tag).toHaveTextContent('Far from threshold');
    expect(tag.getAttribute('title')).toMatch(/\|P\(abnormal\) − threshold\| ≥ 0\.15/);
    expect(tag.getAttribute('title')).toMatch(/not a probability of being correct and not a confidence level/);
  });
});

describe('i18n', () => {
  it('French dictionary has every key, with no empty value and standard terms', () => {
    expect(Object.keys(fr).sort()).toEqual(Object.keys(en).sort());
    for (const v of Object.values(fr)) expect(v.trim()).not.toBe('');
    expect(fr['rel.recall']).toMatch(/Rappel/);
    expect(fr['rel.specificity']).toBe('Spécificité');
    expect(fr['rel.balanced']).toBe('Exactitude équilibrée');
    expect(fr['card.threshold']).toBe('seuil');
    expect(fr['axis.decision_score']).toMatch(/Score de décision/);
    expect(fr['axis.probability']).toBe("Probabilité d'anomalie");
  });

  it('switching to French changes the visible strings and keeps the MLflow links', async () => {
    await renderAnalyzed();
    const hrefBefore = [...document.querySelectorAll('a')].map(a => a.getAttribute('href'));
    fireEvent.click(screen.getByRole('button', { name: 'FR' }));
    expect(await screen.findByText('ANALYSE TERMINÉE')).toBeInTheDocument();
    // raw API values shown as code (label rule, version) are data, not UI strings
    const clone = document.body.cloneNode(true) as HTMLElement;
    clone.querySelectorAll('dd.mono').forEach(n => n.remove());
    const text = clone.textContent ?? '';
    // no raw dictionary key leaks, and no English-only UI string remains
    for (const key of Object.keys(en)) expect(text).not.toContain(key.includes('.') && key.length > 8 ? `${key} ` : '\u0000');
    const englishOnly = Object.entries(en).filter(([k, v]) => v !== fr[k as keyof typeof en] && v.length > 6 && !v.includes('{')).map(([, v]) => v);
    for (const s of englishOnly) expect(text, `untranslated: ${s}`).not.toContain(s);
    expect([...document.querySelectorAll('a')].map(a => a.getAttribute('href'))).toEqual(hrefBefore);
    expect(localStorage.getItem('heart-lab-lang')).toBe('fr');
  });

  it('translate fills variables', () => {
    expect(translate('fr', 'card.modelLabel', { fold: 3 })).toBe('Validation croisée à 3 plis · modèle du pli 3');
  });
});

describe('exports contain exactly the values shown', () => {
  it('CSV rows match the cards', async () => {
    await renderAnalyzed();
    const rows = csvRows(analysis);
    expect(rows).toHaveLength(5);
    for (const row of rows) {
      const card = document.querySelector(`article.model-card[data-model-key="${row.model_key}"]`) as HTMLElement;
      expect(row.score).toBe(within(card).getByTestId('score').textContent);
      expect(row.threshold).toBe(within(card).getByTestId('threshold').textContent);
      expect(row.label).toBe(card.querySelector('.prediction')!.textContent);
      expect(card).toHaveTextContent(row.inference_time);
      const rel = within(card).getByTestId('reliability');
      expect(rel).toHaveTextContent(`${row.test_recall_pct} ± ${row.test_recall_std_pct}%`);
      expect(rel).toHaveTextContent(`${row.test_specificity_pct} ± ${row.test_specificity_std_pct}%`);
      expect(rel).toHaveTextContent(`${row.test_balanced_accuracy_pct} ± ${row.test_balanced_accuracy_std_pct}%`);
      expect(within(card).getByRole('link', { name: /3-fold results in MLflow/ })).toHaveAttribute('href', row.mlflow_3fold_results);
      expect(row.mlflow_analysis).toBe(analysis.mlflow_run_url);
      expect(fmtMs(results.find(r => r.model_key === row.model_key)!.latency_ms)).toBe(row.inference_time);
    }
    const csv = toCsv(analysis).trim().split('\n');
    expect(csv[0]).toBe(CSV_COLUMNS.join(','));
    expect(csv).toHaveLength(6);
  });

  it('JSON is the API response without the large images', () => {
    const parsed = JSON.parse(toJson(analysis));
    expect(parsed.results).toEqual(analysis.results);
    expect(parsed.filename).toBe(analysis.filename);
    expect(parsed.mel_spectrogram_png_base64).toBeUndefined();
  });
});

it('the disclaimer stays visible on every tab', async () => {
  vi.stubGlobal('fetch', fakeFetch());
  render(<App />);
  for (const tab of ['Benchmark', 'History', 'Models', 'Analyze']) {
    fireEvent.click(screen.getByRole('button', { name: tab }));
    expect(screen.getByRole('note')).toHaveTextContent('not a medical device');
  }
});
