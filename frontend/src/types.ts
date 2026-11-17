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
  waveform_png_base64: string;
  total_processing_ms: number;
  mlflow_run_id?: string | null;
  mlflow_run_url?: string | null;
  warning?: string | null;
};

export type Health = { status: string; models_loaded: boolean; mlflow_reachable: boolean; loaded_models: string[] };
export type ModelInfo = {
  key: string; name: string; loaded: boolean; version?: string | null; source: string;
  type: string; kernel?: string; input_features?: string[]; input_shape?: number[];
  score_type: string; threshold: number; metrics: Record<string, number>; label_convention: Record<string, string>;
  mel_settings?: Record<string, number>;
};
export type HistoryItem = {
  run_id: string; run_name?: string; run_url: string; start_time: number; filename?: string;
  metrics: Record<string, number>; tags: Record<string, string>;
};
