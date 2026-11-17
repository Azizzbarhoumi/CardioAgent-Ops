import { useEffect, useRef, useState } from 'react';
import { AudioPlayer } from './AudioPlayer';

type Props = { file: File | null; busy: boolean; onFile(file: File | null): void; onAnalyze(): void };
export function UploadCard({ file, busy, onFile, onAnalyze }: Props) {
  const input = useRef<HTMLInputElement>(null); const [dragging, setDragging] = useState(false); const [error, setError] = useState('');
  function accept(candidate?: File) {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith('.wav')) { setError('Choose a .wav recording.'); onFile(null); return; }
    if (candidate.size === 0) { setError('This file is empty. Choose a WAV recording with audio.'); onFile(null); return; }
    if (candidate.size > 20 * 1024 * 1024) { setError('The file exceeds the 20 MB upload limit.'); onFile(null); return; }
    setError(''); onFile(candidate);
  }
  return <section className="panel upload-panel"><div className="section-heading"><div><div className="eyebrow">STEP 1</div><h2>Choose a recording</h2><p>Upload a WAV file to compare each model independently.</p></div><span className="upload-icon">↑</span></div>
    <div className={`drop-zone ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]); }}>
      <input ref={input} type="file" aria-label="Upload WAV recording" accept=".wav,audio/wav" hidden onChange={e => accept(e.target.files?.[0])} />
      <div className="drop-mark">♫</div><div><strong>Drop your WAV recording here</strong><span>or browse your device · up to 20 MB</span></div>
      <button type="button" className="button secondary" onClick={() => input.current?.click()}>Browse files</button>
    </div>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {file && <><div className="file-row"><span className="file-type">WAV</span><div className="file-meta"><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(file.size < 1024 * 1024 ? 1 : 0)} {file.size < 1024 * 1024 ? 'KB' : 'MB'}</span></div><button className="icon-button" aria-label="Remove file" onClick={() => { onFile(null); if (input.current) input.current.value = ''; }}>×</button></div><AudioPlayer file={file} /></>}
    <div className="upload-actions"><button className="button primary" disabled={!file || busy} onClick={onAnalyze}>{busy ? 'Analyzing…' : 'Analyze recording'}<span aria-hidden="true">→</span></button>{file && <button className="button text-button" onClick={() => { onFile(null); if (input.current) input.current.value = ''; }}>Clear</button>}</div>
  </section>;
}
