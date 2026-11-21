import { useCallback, useEffect, useState } from 'react';
import { analyzeFile, fetchSample, getBenchmark, getHealth, getHistory, getModels, getSamples } from './api';
import type { Analysis, BenchmarkModel, Health, HistoryItem, ModelInfo, Sample } from './types';
import { I18nProvider, useI18n } from './i18n';
import { baseName, download, toCsv, toJson } from './lib/export';
import { UploadPanel } from './monitor/Upload';
import { ChannelGrid, SummaryTable } from './monitor/Results';
import { SignalPanel } from './monitor/Signal';
import { BenchmarkView } from './monitor/Benchmark';
import { Header, HistoryView, ModelsView, PrintHeader, PrintLinks, type Tab } from './monitor/Pages';
import './monitor/monitor.css';

function Lab() {
  const { t, lang } = useI18n();
  const loc = lang === 'fr' ? 'fr-FR' : 'en-US';
  const [health, setHealth] = useState<Health | null>(null);
  const [tab, setTab] = useState<Tab>('Analyze');
  const [file, setFile] = useState<File | null>(null);
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [analyzedFile, setAnalyzedFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [samples, setSamples] = useState<Sample[]>([]);
  const [samplesWarning, setSamplesWarning] = useState<string | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]), [modelsWarning, setModelsWarning] = useState('');
  const [bench, setBench] = useState<BenchmarkModel[]>([]), [benchWarning, setBenchWarning] = useState(''), [benchLoading, setBenchLoading] = useState(false);
  const [history, setHistory] = useState<HistoryItem[]>([]), [historyWarning, setHistoryWarning] = useState('');

  const refresh = useCallback(async () => { try { setHealth(await getHealth()); } catch { setHealth(null); } }, []);
  useEffect(() => {
    void refresh();
    void getModels().then(r => { setModels(r.models); setModelsWarning(r.warning || ''); }).catch(e => setModelsWarning(e.message));
    void getSamples().then(r => { setSamples(r.samples); setSamplesWarning(r.warning ?? null); }).catch(e => setSamplesWarning(e.message));
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const loadHistory = useCallback(() => void getHistory().then(r => { setHistory(r.analyses); setHistoryWarning(r.warning || ''); }).catch(e => setHistoryWarning(e.message)), []);
  useEffect(() => {
    if (tab === 'Benchmark') { setBenchLoading(true); void getBenchmark().then(r => { setBench(r.models); setBenchWarning(''); }).catch(e => setBenchWarning(e.message)).finally(() => setBenchLoading(false)); }
    if (tab === 'History') loadHistory();
  }, [tab, loadHistory]);

  async function analyze(target: File | null = file) {
    if (!target) return;
    setBusy(true); setError('');
    try { const result = await analyzeFile(target); setAnalysis(result); setAnalyzedFile(target); void refresh(); }
    catch (e) { setError(e instanceof Error ? e.message : t('error.analysis')); }
    finally { setBusy(false); }
  }
  async function trySample(sample: Sample) {
    setError('');
    try { const f = await fetchSample(sample); setFile(f); setAnalysis(null); await analyze(f); }
    catch (e) { setError(e instanceof Error ? e.message : t('error.analysis')); }
  }

  return <div className="lab">
    <Header health={health} tab={tab} onTab={setTab} />
    <main className="wrap">
      {tab === 'Analyze' && <div className="page">
        <UploadPanel file={file} busy={busy} samples={samples} samplesWarning={samplesWarning}
          onFile={f => { setFile(f); setAnalysis(null); setError(''); }} onAnalyze={() => void analyze()} onSample={s => void trySample(s)} />
        {error && <div className="alert error" role="alert">{error}</div>}
        {busy && <div className="loading" role="status"><span className="spinner" />{t('loading')}</div>}
        {analysis && <>
          <PrintHeader analysis={analysis} />
          <section className="pnl result-head">
            <div><div className="h"><span>{t('result.complete')}</span></div>
              <p className="mono file-name">{analysis.filename}</p>
              <p className="muted">{t('result.meta', { dur: analysis.duration_s.toFixed(2), src: analysis.source_sample_rate.toLocaleString(loc), rs: analysis.sample_rate_after_resampling.toLocaleString(loc), ms: analysis.total_processing_ms.toFixed(0) })}</p></div>
            <div className="actions no-print">
              {analysis.mlflow_run_url ? <a className="mlflow-link" href={analysis.mlflow_run_url} target="_blank" rel="noreferrer">{t('result.openRun')}</a> : <span className="muted">{t('result.noRun')}</span>}
              <button type="button" className="btn sm" onClick={() => download(`${baseName(analysis.filename)}_analysis.json`, toJson(analysis), 'application/json')}>{t('export.json')}</button>
              <button type="button" className="btn sm" onClick={() => download(`${baseName(analysis.filename)}_analysis.csv`, toCsv(analysis), 'text/csv')}>{t('export.csv')}</button>
              <button type="button" className="btn sm" onClick={() => window.print()}>{t('export.print')}</button>
            </div>
          </section>
          {analysis.warning && <div className="alert warning">{analysis.warning}</div>}
          <SummaryTable results={analysis.results} />
          <SignalPanel analysis={analysis} file={analyzedFile} />
          <ChannelGrid results={analysis.results} analysisUrl={analysis.mlflow_run_url} />
          <PrintLinks analysis={analysis} />
        </>}
        {!analysis && !busy && <div className="empty">{t('empty')}</div>}
      </div>}
      {tab === 'Benchmark' && <BenchmarkView models={bench} warning={benchWarning} loading={benchLoading} />}
      {tab === 'History' && <HistoryView items={history} warning={historyWarning} onRefresh={loadHistory} />}
      {tab === 'Models' && <ModelsView models={models} warning={modelsWarning} />}
    </main>
    <footer className="foot">{t('app.footer')}</footer>
  </div>;
}

export default function App() { return <I18nProvider><Lab /></I18nProvider>; }
