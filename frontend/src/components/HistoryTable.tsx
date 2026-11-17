import type { HistoryItem } from '../types';
import type { KeyboardEvent, MouseEvent } from 'react';
const labels = [['svm_linear.prediction','Linear SVM'],['svm_rbf.prediction','RBF SVM'],['decision_tree.prediction','Decision Tree'],['cnn_custom.prediction','CNN']] as const;

function openRun(url: string) {
  window.open(url, '_blank', 'noopener,noreferrer');
}

function handleRowClick(event: MouseEvent<HTMLTableRowElement>, url: string) {
  if (event.target instanceof Element && event.target.closest('a')) return;
  openRun(url);
}

function handleRowKeyDown(event: KeyboardEvent<HTMLTableRowElement>, url: string) {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    openRun(url);
  }
}

export function HistoryTable({ items }: { items: HistoryItem[] }) {
  if (!items.length) return <div className="empty-state">No analyses found. Analyze a WAV recording to create a history entry.</div>;
  return <div className="panel table-scroll"><table><thead><tr><th>Date</th><th>Recording</th>{labels.map(([,name]) => <th key={name}>{name}</th>)}<th>MLflow</th></tr></thead><tbody>{items.map(item => { const recording = item.filename || item.run_name || 'Recording'; return <tr key={item.run_id} tabIndex={0} aria-label={`Open MLflow run for ${recording}`} onClick={event => handleRowClick(event, item.run_url)} onKeyDown={event => handleRowKeyDown(event, item.run_url)}><td>{new Date(item.start_time).toLocaleString()}</td><td>{recording}</td>{labels.map(([key,name]) => <td key={name}>{item.metrics[key] == null ? '—' : Number(item.metrics[key]) === 1 ? 'Abnormal' : 'Normal'}</td>)}<td><a href={item.run_url} target="_blank" rel="noreferrer">Open run ↗</a></td></tr>; })}</tbody></table></div>;
}
