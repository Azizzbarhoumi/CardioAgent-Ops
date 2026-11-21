export type ModelResult = {
  model_name: string;
  model_key: string;
  model_version: string;
  model_run_id?: string | null;
  prediction: 0 | 1;
  prediction_label: string;
  confidence_or_score: number;
  confidence_type: string;
  score: number;
  score_type: string;
  threshold: number;
  latency_ms: number;
  kernel?: string | null;
  label_rule?: string | null;
  benchmark_run_url?: string | null;
  benchmark_folds_urls?: string[];
  benchmark_train_vs_test_url?: string | null;
  benchmark_summary?: BenchmarkSummary | null;
  benchmark_warning?: string | null;
};

export type BenchmarkSummary = {
  n_folds: number; mean_test_recall: number; std_test_recall: number;
  mean_test_accuracy: number; std_test_accuracy: number; parent_run_id: string;
  // read-only, computed by the API from the confusion matrices logged in MLflow
  mean_test_specificity?: number | null; std_test_specificity?: number | null;
  mean_test_balanced_accuracy?: number | null; std_test_balanced_accuracy?: number | null;
  metrics_source?: string | null; app_fold?: string | number | null;
};
export type ConfusionMatrix = { labels: string[]; matrix: number[][]; tn: number; fp: number; fn: number; tp: number; total: number };
export type Sample = { id: string; label: string; file: string; source?: string };
export type SplitMetrics = Record<string, number | null>;
export type BenchmarkFold = { fold: number; run_id: string; run_url: string; train: SplitMetrics; validation: SplitMetrics; test: SplitMetrics; confusion_matrices?: { test?: ConfusionMatrix; train?: ConfusionMatrix } };
export type BenchmarkModel = {
  model_key: string; display_name: string; kernel?: string | null; parent_run_id: string;
  benchmark_run_url: string; train_vs_test_url: string; benchmark_folds_urls: string[]; app_fold?: string;
  mean: Record<'train' | 'validation' | 'test', SplitMetrics>; std: Record<'train' | 'validation' | 'test', SplitMetrics>;
  folds: BenchmarkFold[];
  confusion_matrix_warning?: string | null;
};

export type ModelGroupData = { key: string; name: string; models: ModelResult[] };
export type Analysis = {
  filename: string;
  duration_s: number;
  source_sample_rate: number;
  sample_rate_after_resampling: number;
  results: ModelResult[];
  model_groups: ModelGroupData[];
  waveform_points: [number, number][];
  mel_spectrogram_png_base64: string;
  mel_spectrogram_axes?: { left: number; right: number; top: number; bottom: number; time_start_s: number; time_end_s: number } | null;
  waveform_png_base64: string;
  total_processing_ms: number;
  mlflow_run_id?: string | null;
  mlflow_run_url?: string | null;
  warning?: string | null;
};

export type Health = { status: string; models_loaded: boolean; mlflow_reachable: boolean; loaded_models: string[] };
export type ModelInfo = {
  key: string; name: string; loaded: boolean; version?: string | null; framework?: string;
  kernel?: string | null; input?: string; feature_version?: string; score_type: string; threshold: number;
  label_rule?: string; app_fold?: number; model_run_id?: string | null; metrics: Record<string, number>;
  label_convention?: Record<string, string>;
};
export type HistoryItem = {
  run_id: string; run_name?: string; run_url: string; start_time: number; filename?: string;
  metrics: Record<string, number>; tags: Record<string, string>;
};
