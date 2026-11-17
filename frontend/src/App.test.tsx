import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import App from './App';

const models = ['svm_linear', 'svm_rbf', 'decision_tree', 'cnn_custom'].map((key, index) => ({
  model_name: ['SVM (linear)', 'SVM (RBF)', 'Decision Tree', 'CNN'][index],
  model_key: key,
  model_version: `version ${key}`,
  model_run_id: null,
  prediction: (index === 3 ? 1 : 0) as 0 | 1,
  prediction_label: index === 3 ? 'Abnormal' : 'Normal',
  confidence_or_score: 0.25,
  confidence_type: index < 2 ? 'decision_score' : 'probability',
  score: index < 2 ? -0.25 : 0.25,
  score_type: index < 2 ? 'decision_score' : 'probability',
  threshold: index < 2 ? 0 : 0.5,
  latency_ms: 2.5,
  kernel: index === 0 ? 'linear' : index === 1 ? 'rbf' : null,
}));

const analysis = {
  filename: 'sample.wav',
  duration_s: 1.2,
  source_sample_rate: 4000,
  sample_rate_after_resampling: 2000,
  results: models,
  model_groups: [
    { key: 'svm', name: 'SVM (2 kernels)', models: models.slice(0, 2) },
    { key: 'decision_tree', name: 'Decision Tree', models: models.slice(2, 3) },
    { key: 'cnn', name: 'CNN', models: models.slice(3) },
  ],
  waveform_points: [[0, 0], [0.4, 0.2], [0.8, -0.1], [1.2, 0]] as [number, number][],
  mel_spectrogram_png_base64: 'aGVsbG8=',
  waveform_png_base64: '',
  total_processing_ms: 30,
  mlflow_run_id: 'abc',
  mlflow_run_url: 'http://localhost:5000/run/abc',
  warning: null,
};

const apiResponse = (body: unknown, status = 200) =>
  ({ ok: status < 400, status, json: async () => body }) as Response;

function createFetch(analyze: () => Response | Promise<Response> = () => apiResponse(analysis)) {
  return vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith('/health')) {
      return apiResponse({ status: 'ok', models_loaded: true, mlflow_reachable: true, loaded_models: [] });
    }
    if (url.endsWith('/models')) return apiResponse({ models: [], warning: '' });
    if (init?.method === 'POST') return analyze();
    return apiResponse({ analyses: [], warning: '' });
  });
}

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('Heart Sound Model Lab', () => {
  it('uploads a WAV and displays separate model groups and its MLflow link', async () => {
    vi.stubGlobal('fetch', createFetch());
    render(<App />);

    const wav = new File(['audio'], 'sample.wav', { type: 'audio/wav' });
    fireEvent.change(screen.getByLabelText('Upload WAV recording'), { target: { files: [wav] } });
    fireEvent.click(screen.getByRole('button', { name: /analyze recording/i }));

    expect(await screen.findByRole('heading', { name: 'SVM (2 kernels)' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Decision Tree' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'CNN' })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /open this analysis in mlflow/i })).toHaveAttribute(
      'href',
      'http://localhost:5000/run/abc',
    );
    expect(screen.queryByText(/overall prediction|combined verdict/i)).not.toBeInTheDocument();
  });

  it('shows a loading state while the analysis request is pending', async () => {
    let resolveAnalysis!: (response: Response) => void;
    const pending = new Promise<Response>(resolve => {
      resolveAnalysis = resolve;
    });
    vi.stubGlobal('fetch', createFetch(() => pending));
    render(<App />);

    const wav = new File(['audio'], 'sample.wav', { type: 'audio/wav' });
    fireEvent.change(screen.getByLabelText('Upload WAV recording'), { target: { files: [wav] } });
    fireEvent.click(screen.getByRole('button', { name: /analyze recording/i }));

    expect(await screen.findByRole('status')).toHaveTextContent('Running the four independent model analyses');
    expect(screen.getByRole('button', { name: /analyzing/i })).toBeDisabled();
    resolveAnalysis(apiResponse(analysis));
  });

  it('shows the API error message when analysis fails', async () => {
    vi.stubGlobal('fetch', createFetch(() => apiResponse({ detail: 'Audio is too short.' }, 422)));
    render(<App />);

    const wav = new File(['audio'], 'sample.wav', { type: 'audio/wav' });
    fireEvent.change(screen.getByLabelText('Upload WAV recording'), { target: { files: [wav] } });
    fireEvent.click(screen.getByRole('button', { name: /analyze recording/i }));

    expect(await screen.findByRole('alert')).toHaveTextContent('Audio is too short.');
  });
});
