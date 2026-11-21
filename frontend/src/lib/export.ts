// Exports are generated in the browser from the exact API response on screen. No value is recomputed:
// the CSV uses the same formatters as the cards and the table, the JSON is the API response itself.
import type { Analysis } from '../types';
import { fmtMs, fmtPct, fmtScore, fmtThreshold, sortResults } from './display';

export const CSV_COLUMNS = [
  'model_key', 'model_name', 'prediction', 'label', 'score_type', 'score', 'threshold', 'inference_time',
  'model_version', 'test_recall_pct', 'test_recall_std_pct', 'test_specificity_pct', 'test_specificity_std_pct',
  'test_balanced_accuracy_pct', 'test_balanced_accuracy_std_pct', 'mlflow_3fold_results', 'mlflow_train_vs_test',
  'mlflow_analysis',
] as const;

export function csvRows(a: Analysis): Record<(typeof CSV_COLUMNS)[number], string>[] {
  return sortResults(a.results).map(r => {
    const s = r.benchmark_summary;
    return {
      model_key: r.model_key, model_name: r.model_name, prediction: String(r.prediction), label: r.prediction_label,
      score_type: r.score_type ?? '', score: fmtScore(r.score), threshold: fmtThreshold(r.threshold),
      inference_time: fmtMs(r.latency_ms), model_version: r.model_version,
      test_recall_pct: fmtPct(s?.mean_test_recall), test_recall_std_pct: fmtPct(s?.std_test_recall),
      test_specificity_pct: fmtPct(s?.mean_test_specificity), test_specificity_std_pct: fmtPct(s?.std_test_specificity),
      test_balanced_accuracy_pct: fmtPct(s?.mean_test_balanced_accuracy),
      test_balanced_accuracy_std_pct: fmtPct(s?.std_test_balanced_accuracy),
      mlflow_3fold_results: r.benchmark_run_url ?? '', mlflow_train_vs_test: r.benchmark_train_vs_test_url ?? '',
      mlflow_analysis: a.mlflow_run_url ?? '',
    };
  });
}

const cell = (v: string) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
export function toCsv(a: Analysis): string {
  return [CSV_COLUMNS.join(','), ...csvRows(a).map(row => CSV_COLUMNS.map(c => cell(row[c])).join(','))].join('\n') + '\n';
}
export function toJson(a: Analysis): string {
  // drop the large base64 images; everything else is the API response unchanged
  const { mel_spectrogram_png_base64: _m, waveform_png_base64: _w, waveform_points: _p, ...rest } = a;
  return JSON.stringify(rest, null, 2);
}

export function baseName(filename: string) { return filename.replace(/\.[^.]+$/, '') || 'analysis'; }
export function download(name: string, text: string, type: string) {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
