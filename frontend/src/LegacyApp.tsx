import { useCallback, useEffect, useState } from 'react';
import { analyzeFile, getBenchmark, getHealth, getHistory, getModels } from './api';
import type { Analysis, BenchmarkModel, Health, HistoryItem, ModelInfo } from './types';
import { AppHeader } from './components/AppHeader';
import { DisclaimerBanner } from './components/DisclaimerBanner';
import { UploadCard } from './components/UploadCard';
import { ErrorAlert } from './components/ErrorAlert';
import { LoadingState } from './components/LoadingState';
import { ModelGroup } from './components/ModelGroup';
import { ResultsSummaryTable } from './components/ResultsSummaryTable';
import { AudioCharts } from './components/AudioCharts';
import { MlflowLink } from './components/MlflowLink';
import { HistoryTable } from './components/HistoryTable';
import { ModelsInfoPanel } from './components/ModelsInfoPanel';
import { BenchmarkTab } from './components/BenchmarkTab';

type Tab = 'Analyze' | 'Benchmark' | 'History' | 'Models';
export default function LegacyApp() {
  const [health,setHealth]=useState<Health|null>(null), [models,setModels]=useState<ModelInfo[]>([]), [history,setHistory]=useState<HistoryItem[]>([]);
  const [tab,setTab]=useState<Tab>('Analyze'), [file,setFile]=useState<File|null>(null), [analysis,setAnalysis]=useState<Analysis|null>(null);
  const [busy,setBusy]=useState(false), [error,setError]=useState(''), [historyWarning,setHistoryWarning]=useState(''), [modelsWarning,setModelsWarning]=useState('');
  const [bench,setBench]=useState<BenchmarkModel[]>([]), [benchWarning,setBenchWarning]=useState('');
  const refresh=useCallback(async()=>{ try{setHealth(await getHealth());}catch{setHealth(null);} },[]);
  useEffect(()=>{void refresh(); void getModels().then(r=>{setModels(r.models);setModelsWarning(r.warning||'');}).catch(e=>setModelsWarning(e.message)); const timer=window.setInterval(()=>void refresh(),15000); return()=>window.clearInterval(timer);},[refresh]);
  useEffect(()=>{if(tab==='Benchmark')void getBenchmark().then(r=>{setBench(r.models);setBenchWarning('');}).catch(e=>setBenchWarning(e.message));},[tab]);
  useEffect(()=>{if(tab==='History')void getHistory().then(r=>{setHistory(r.analyses);setHistoryWarning(r.warning||'');}).catch(e=>setHistoryWarning(e.message));},[tab]);
  async function analyze(){if(!file)return;setBusy(true);setError('');try{const result=await analyzeFile(file);setAnalysis(result);void refresh();}catch(e){setError(e instanceof Error?e.message:'Analysis failed.');}finally{setBusy(false);}}
  const groups=analysis?.model_groups?.length?analysis.model_groups:analysis?[
    {key:'svm',name:'SVM (2 kernels)',models:analysis.results.filter(r=>r.model_key.startsWith('svm_'))},
    {key:'decision_tree',name:'Decision Tree',models:analysis.results.filter(r=>r.model_key==='decision_tree')},
    {key:'cnn',name:'CNN',models:analysis.results.filter(r=>r.model_key==='cnn_custom')},
    {key:'transfer_learning',name:'Transfer Learning (VGG16)',models:analysis.results.filter(r=>r.model_key==='vgg16_transfer')},
  ]:[];
  return <main className="page-shell"><AppHeader health={health}/><DisclaimerBanner/><nav className="tabs" aria-label="Main sections">{(['Analyze','Benchmark','History','Models'] as Tab[]).map(name=><button key={name} className={tab===name?'active':''} onClick={()=>setTab(name)}>{name}</button>)}</nav>
    {tab==='Analyze'&&<div className="page-content"><UploadCard file={file} busy={busy} onFile={chosen=>{setFile(chosen);setAnalysis(null);setError('');}} onAnalyze={()=>void analyze()}/>{error&&<ErrorAlert>{error}</ErrorAlert>}{busy&&<LoadingState/>}
      {analysis&&<><section className="result-intro"><div><div className="eyebrow">ANALYSIS COMPLETE</div><h2>{analysis.filename}</h2><p>{analysis.duration_s.toFixed(2)} s <span>·</span> {analysis.source_sample_rate.toLocaleString()} Hz → {analysis.sample_rate_after_resampling.toLocaleString()} Hz <span>·</span> {analysis.total_processing_ms.toFixed(0)} ms total</p></div><MlflowLink url={analysis.mlflow_run_url}/></section>{analysis.warning&&<div className="alert warning">{analysis.warning}</div>}
        <section className="results-section"><div className="section-heading"><div><div className="eyebrow">INDEPENDENT RESULTS</div><h2>Model analyses</h2><p>Each model reports its own prediction. No combined result is calculated.</p></div></div><div className="groups-stack">{groups.map(g=><ModelGroup group={g} key={g.key} analysisUrl={analysis.mlflow_run_url}/>)}</div></section>
        <DisclaimerBanner compact/><ResultsSummaryTable results={analysis.results}/><section className="results-section"><div className="section-heading"><div><div className="eyebrow">RECORDING DETAILS</div><h2>Audio analysis</h2></div></div><AudioCharts result={analysis}/></section></>}
      {!analysis&&!busy&&<div className="empty-state">Your independent model results and audio analysis will appear here.</div>}</div>}
    {tab==='Benchmark'&&<BenchmarkTab models={bench} warning={benchWarning}/>}
    {tab==='History'&&<div className="page-content"><div className="section-heading"><div><div className="eyebrow">PAST ANALYSES</div><h2>History</h2><p>Recent analyses logged to MLflow.</p></div><button className="button secondary" onClick={()=>void getHistory().then(r=>{setHistory(r.analyses);setHistoryWarning(r.warning||'');}).catch(e=>setHistoryWarning(e.message))}>Refresh</button></div>{historyWarning&&<div className="alert warning">{historyWarning}</div>}<HistoryTable items={history}/></div>}
    {tab==='Models'&&<div className="page-content"><div className="section-heading"><div><div className="eyebrow">MODEL INFORMATION</div><h2>Models and inputs</h2><p>Exported v2 models (patient-grouped 3-fold benchmark) with their saved thresholds.</p></div></div>{modelsWarning&&<div className="alert warning">{modelsWarning}</div>}<ModelsInfoPanel models={models}/></div>}
    <footer>Heart Sound Model Lab <span>·</span> Educational use only</footer></main>;
}
