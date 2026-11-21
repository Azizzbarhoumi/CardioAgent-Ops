import type { KeyboardEvent, MouseEvent } from 'react';
import type { Analysis, Health, HistoryItem, ModelInfo } from '../types';
import { labelText, modelName, useI18n, type Lang } from '../i18n';
import { fmtThreshold, scoreKind, sortResults } from '../lib/display';

const mlflowUrl = (window.__APP_CONFIG__?.mlflowPublicUrl || import.meta.env.VITE_MLFLOW_PUBLIC_URL || 'http://localhost:5000').replace(/\/$/, '');
export type Tab = 'Analyze' | 'Benchmark' | 'History' | 'Models';
const TAB_KEYS = { Analyze: 'tabs.analyze', Benchmark: 'tabs.benchmark', History: 'tabs.history', Models: 'tabs.models' } as const;

export function Header({ health, tab, onTab }: { health: Health | null; tab: Tab; onTab(t: Tab): void }) {
  const { t, lang, setLang } = useI18n();
  return <>
    <header className="top">
      <div className="brand"><span className="brand-mark" aria-hidden="true">♥</span><div><h1>{t('app.title')}</h1><span className="brand-sub">{t('app.subtitle')}</span></div></div>
      <div className="st">
        <span className={health?.models_loaded ? 'on' : 'off'}>{t(health?.models_loaded ? 'status.ready' : 'status.apiDown')}</span>
        <span className={health?.mlflow_reachable ? 'on' : 'off'}>{t(health?.mlflow_reachable ? 'status.mlflowUp' : 'status.mlflowDown')}</span>
        <a href={mlflowUrl} target="_blank" rel="noreferrer">{t('status.openMlflow')}</a>
      </div>
      <div className="lang" role="group" aria-label={t('lang.label')}>
        {(['en', 'fr'] as Lang[]).map(l => <button key={l} type="button" aria-pressed={lang === l} className={lang === l ? 'on' : ''} onClick={() => setLang(l)}>{l.toUpperCase()}</button>)}
      </div>
      <nav aria-label={t('tabs.label')}>{(Object.keys(TAB_KEYS) as Tab[]).map(name =>
        <button key={name} type="button" data-tab={name} aria-current={tab === name ? 'page' : undefined} className={tab === name ? 'on' : ''} onClick={() => onTab(name)}>{t(TAB_KEYS[name])}</button>)}</nav>
    </header>
    <div className="disc" role="note"><i aria-hidden="true">{t('disclaimer.tag')}</i>{t('disclaimer.full')}</div>
  </>;
}

const HIST = [['svm_linear.prediction', 'svm_linear'], ['svm_rbf.prediction', 'svm_rbf'], ['decision_tree.prediction', 'decision_tree'], ['cnn_custom.prediction', 'cnn_custom'], ['vgg16_transfer.prediction', 'vgg16_transfer']] as const;
export function HistoryView({ items, warning, onRefresh }: { items: HistoryItem[]; warning?: string; onRefresh(): void }) {
  const { t, lang } = useI18n();
  const open = (url: string) => window.open(url, '_blank', 'noopener,noreferrer');
  const click = (e: MouseEvent<HTMLTableRowElement>, url: string) => { if (e.target instanceof Element && e.target.closest('a')) return; open(url); };
  const key = (e: KeyboardEvent<HTMLTableRowElement>, url: string) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(url); } };
  return <div className="page">
    <section className="pnl"><h2 className="h"><span>{t('history.eyebrow')}</span><button type="button" className="btn sm" onClick={onRefresh}>{t('history.refresh')}</button></h2><h1 className="page-title">{t('history.title')}</h1><p className="desc">{t('history.desc')}</p></section>
    {warning && <div className="alert warning" role="alert">{warning}</div>}
    {!items.length ? <div className="empty">{t('history.empty')}</div> :
      <section className="pnl ts"><table><thead><tr><th>{t('history.date')}</th><th>{t('history.recording')}</th>{HIST.map(([, k]) => <th key={k}>{modelName(t, k, k, true)}</th>)}<th>MLflow</th></tr></thead>
        <tbody>{items.map(item => { const name = item.filename || item.run_name || t('history.recording');
          return <tr key={item.run_id} tabIndex={0} aria-label={t('history.openAria', { name })} onClick={e => click(e, item.run_url)} onKeyDown={e => key(e, item.run_url)}>
            <td>{new Date(item.start_time).toLocaleString(lang === 'fr' ? 'fr-FR' : 'en-GB')}</td><td className="mono">{name}</td>
            {HIST.map(([m, k]) => <td key={k}>{item.metrics[m] == null ? '—' : t(Number(item.metrics[m]) === 1 ? 'label.abnormal' : 'label.normal')}</td>)}
            <td><a href={item.run_url} target="_blank" rel="noreferrer">{t('history.open')}</a></td></tr>; })}</tbody></table></section>}
  </div>;
}

const MODEL_METRICS = [['rel.recall', 'mean_test_recall', 'std_test_recall'], ['rel.specificity', 'mean_test_specificity', 'std_test_specificity'], ['rel.balanced', 'mean_test_balanced_accuracy', 'std_test_balanced_accuracy'], ['metric.f1', 'mean_test_f1', 'std_test_f1']] as const;
export function ModelsView({ models, warning }: { models: ModelInfo[]; warning?: string }) {
  const { t } = useI18n();
  return <div className="page">
    <section className="pnl"><h2 className="h"><span>{t('models.eyebrow')}</span></h2><h1 className="page-title">{t('models.title')}</h1><p className="desc">{t('models.desc')}</p></section>
    {warning && <div className="alert warning" role="alert">{warning}</div>}
    <div className="chs">{models.map((m, i) => <article className="ch info" key={m.key} data-info-key={m.key}>
      <div className="ch-id"><span className="ch-no">{t('channel', { n: i + 1 })}</span><span className={`dot ${m.loaded ? 'on' : 'off'}`}>{t(m.loaded ? 'models.loaded' : 'models.notLoaded')}</span></div>
      <h3>{modelName(t, m.key, m.name)}</h3><span className="muted">{m.framework}</span>
      <dl><dt>{t('models.input')}</dt><dd>{m.input}</dd>
        <dt>{t('models.output')}</dt><dd>{t(`axis.${scoreKind(m.score_type)}`)} · {t('card.threshold')} <span className="mono">{fmtThreshold(m.threshold)}</span></dd>
        <dt>{t('models.labelRule')}</dt><dd className="mono">{m.label_rule}</dd><dt>{t('models.labels')}</dt><dd>{t('card.labelMappingValue')}</dd>
        {m.app_fold != null && <><dt>{t('models.appModel')}</dt><dd>{t('models.appModelValue', { n: m.app_fold })}</dd></>}</dl>
      {Object.keys(m.metrics ?? {}).length > 0 && <div className="rel"><div className="rel-title">{t('models.metricsTitle')}</div>{MODEL_METRICS.map(([label, mean, std]) => m.metrics[mean] != null &&
        <div className="rel-row" key={mean}><span>{t(label)}</span><span /><b>{(m.metrics[mean] * 100).toFixed(1)} ± {((m.metrics[std] ?? 0) * 100).toFixed(1)}%</b></div>)}</div>}
    </article>)}</div>
  </div>;
}

/** Shown only when printing (print stylesheet). One page: file, 5 results, figures, disclaimer, MLflow links. */
export function PrintHeader({ analysis }: { analysis: Analysis }) {
  const { t, lang } = useI18n();
  return <div className="print-only print-head">
    <h1>{t('app.title')} · {t('print.title')}</h1>
    <p className="mono">{analysis.filename}</p>
    <p>{t('print.generated', { date: new Date().toLocaleString(lang === 'fr' ? 'fr-FR' : 'en-GB') })}</p>
  </div>;
}
export function PrintLinks({ analysis }: { analysis: Analysis }) {
  const { t, lang } = useI18n();
  return <section className="print-only print-links"><h2>MLflow</h2><ul>
    {analysis.mlflow_run_url && <li>{t('link.analysis').replace(' ↗', '')}: {analysis.mlflow_run_url}</li>}
    {sortResults(analysis.results).map(r => <li key={r.model_key}>{modelName(t, r.model_key, r.model_name)} ({labelText(t, r.prediction, r.prediction_label, lang)}): {r.benchmark_run_url ?? '–'}</li>)}
  </ul><p className="print-disc">{t('disclaimer.full')}</p></section>;
}
