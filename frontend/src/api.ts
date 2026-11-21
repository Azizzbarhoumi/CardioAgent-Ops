import type { Analysis, BenchmarkModel, Health, HistoryItem, ModelInfo, Sample } from './types';

export const API_URL = (window.__APP_CONFIG__?.apiUrl || import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`${API_URL}${path}`, init); }
  catch { throw new Error(`Could not reach the API at ${API_URL}. Start the API and try again.`); }
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try { const body = await response.json(); message = body.detail || message; } catch { /* response may not be JSON */ }
    throw new Error(message);
  }
  return response.json() as Promise<T>;
}

export const getHealth = () => request<Health>('/health');
export const getModels = () => request<{ models: ModelInfo[]; warning?: string }>('/models');
export const getBenchmark = () => request<{ models: BenchmarkModel[]; evaluation: string }>('/benchmark');
export const getHistory = (limit = 20) => request<{ analyses: HistoryItem[]; warning?: string }>(`/history?limit=${limit}`);
export const analyzeFile = (file: File) => {
  const body = new FormData(); body.append('file', file);
  return request<Analysis>('/analyze_pcg', { method: 'POST', body });
};
export const getSamples = () => request<{ samples: Sample[]; warning?: string | null }>('/samples');
export async function fetchSample(sample: Sample): Promise<File> {
  let response: Response;
  try { response = await fetch(`${API_URL}/samples/${encodeURIComponent(sample.id)}`); }
  catch { throw new Error(`Could not reach the API at ${API_URL}. Start the API and try again.`); }
  if (!response.ok) throw new Error(`Sample unavailable (${response.status})`);
  return new File([await response.blob()], sample.file, { type: 'audio/wav' });
}
