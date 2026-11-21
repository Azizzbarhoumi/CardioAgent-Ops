import type { ModelResult } from '../types';
import { modelName, useI18n } from '../i18n';
import { fmtMs, fmtScore, fmtThreshold, foldFromVersion, scoreKind, sortResults } from '../lib/display';
import { DistanceTag, PredictionBadge, Reliability, ScoreScale } from './parts';

export function SummaryTable({ results }: { results: ModelResult[] }) {
  const { t } = useI18n();
  return <section className="pnl" data-testid="summary-table" aria-labelledby="summary-title">
    <h2 className="h" id="summary-title"><span>{t('summary.title')}</span><span className="h-note">{t('summary.note')}</span></h2>
    <div className="ts"><table>
      <thead><tr><th>{t('col.ch')}</th><th>{t('col.model')}</th><th>{t('col.prediction')}</th><th>{t('col.scoreType')}</th><th className="r">{t('col.score')}</th><th className="r">{t('col.threshold')}</th><th className="r">{t('col.time')}</th></tr></thead>
      <tbody>{sortResults(results).map((r, i) => <tr key={r.model_key} data-row-key={r.model_key}>
        <td className="mono">{i + 1}</td><td>{modelName(t, r.model_key, r.model_name)}</td><td><PredictionBadge result={r} className="table-badge" /></td>
        <td>{t(`scoreType.${scoreKind(r.score_type)}`)}</td><td className="n" data-col="score">{fmtScore(r.score)}</td><td className="n" data-col="threshold">{fmtThreshold(r.threshold)}</td><td className="n">{fmtMs(r.latency_ms)}</td></tr>)}</tbody>
    </table></div>
  </section>;
}

export function ChannelCard({ result, channel, analysisUrl }: { result: ModelResult; channel: number; analysisUrl?: string | null }) {
  const { t } = useI18n();
  const kind = scoreKind(result.score_type);
  const fold = foldFromVersion(result);
  const rule = result.label_rule ?? (kind === 'decision_score' ? 'decision_score > 0' : 'P(abnormal) >= threshold');
  return <article className="model-card ch" data-model-key={result.model_key}>
    <div className="ch-id"><span className="ch-no">{t('channel', { n: channel })}</span><PredictionBadge result={result} /></div>
    <h3>{modelName(t, result.model_key, result.model_name)}</h3>
    <div className="read">
      <div><div className="read-u">{t(`axis.${kind}`)}</div><div className="read-v" data-testid="score">{fmtScore(result.score)}</div></div>
      <div className="read-thr">{t('card.threshold')}<b data-testid="threshold">{fmtThreshold(result.threshold)}</b></div>
    </div>
    <ScoreScale score={result.score} threshold={result.threshold} type={kind} />
    <div className="ch-row"><span>{t('card.inference')} <b className="mono">{fmtMs(result.latency_ms)}</b></span><DistanceTag score={result.score} threshold={result.threshold} type={kind} /></div>
    <Reliability result={result} />
    <div className="lk">
      {result.benchmark_run_url
        ? <a className="cv-link" href={result.benchmark_run_url} target="_blank" rel="noreferrer">{t('link.cv')}</a>
        : <span className="muted">{result.benchmark_warning ?? t('link.cvUnavailable')}</span>}
      {result.benchmark_train_vs_test_url && <a href={result.benchmark_train_vs_test_url} target="_blank" rel="noreferrer">{t('link.tvt')}</a>}
      {analysisUrl && <a href={analysisUrl} target="_blank" rel="noreferrer">{t('link.analysis')}</a>}
    </div>
    <div className="ch-model">{fold != null ? t('card.modelLabel', { fold }) : t('card.modelUnknown')}</div>
    <details className="ch-details"><summary>{t('card.details')}</summary><dl>
      <dt>{t('card.rawVersion')}</dt><dd className="mono">{result.model_version}</dd>
      <dt>{t('card.labelRule')}</dt><dd className="mono">{rule}</dd>
      <dt>{t('card.labelMapping')}</dt><dd>{t('card.labelMappingValue')}</dd>
      {result.model_run_id && <><dt>{t('card.trainingRun')}</dt><dd className="mono">{result.model_run_id}</dd></>}
      {(result.benchmark_folds_urls ?? []).length > 0 && <><dt>{t('card.foldRuns')}</dt><dd>{(result.benchmark_folds_urls ?? []).map((url, i) => <a key={url} href={url} target="_blank" rel="noreferrer">{t('card.fold', { n: i + 1 })} </a>)}</dd></>}
    </dl></details>
  </article>;
}

/** 5 independent channels. The SVM pair is a visual group only; there is no shared or combined result. */
export function ChannelGrid({ results, analysisUrl }: { results: ModelResult[]; analysisUrl?: string | null }) {
  const { t } = useI18n();
  const sorted = sortResults(results);
  const svm = sorted.filter(r => r.model_key.startsWith('svm_'));
  const others = sorted.filter(r => !r.model_key.startsWith('svm_'));
  const no = (r: ModelResult) => sorted.indexOf(r) + 1;
  return <section aria-labelledby="channels-title">
    <h2 className="h" id="channels-title"><span>{t('channels.title')}</span><span className="h-note">{t('channels.note')}</span></h2>
    <div className="chs" data-testid="channel-grid">
      {svm.length > 0 && <div className="svm-group" data-testid="svm-group" role="group" aria-labelledby="svm-group-title">
        <h3 className="gh" id="svm-group-title">{t('group.svm')}</h3>
        {svm.map(r => <ChannelCard key={r.model_key} result={r} channel={no(r)} analysisUrl={analysisUrl} />)}
      </div>}
      {others.map(r => <ChannelCard key={r.model_key} result={r} channel={no(r)} analysisUrl={analysisUrl} />)}
    </div>
  </section>;
}
