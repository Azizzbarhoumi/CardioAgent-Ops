import type { Analysis, Health, HistoryItem, ModelInfo } from './types';

export const API_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8000').replace(/\/$/, '');

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
export const getHistory = (limit = 20) => request<{ analyses: HistoryItem[]; warning?: string }>(`/history?limit=${limit}`);
export const analyzeFile = (file: File) => {
  const body = new FormData(); body.append('file', file);
  return request<Analysis>('/analyze_pcg', { method: 'POST', body });
};
