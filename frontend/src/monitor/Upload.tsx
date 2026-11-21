import { useRef, useState } from 'react';
import type { Sample } from '../types';
import { useI18n } from '../i18n';

type Props = {
  file: File | null; busy: boolean; samples: Sample[]; samplesWarning?: string | null;
  onFile(file: File | null): void; onAnalyze(): void; onSample(sample: Sample): void;
};

/** One clean drop zone (no extra icon button, no empty bar). */
export function UploadPanel({ file, busy, samples, samplesWarning, onFile, onAnalyze, onSample }: Props) {
  const { t } = useI18n();
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState('');
  function accept(candidate?: File) {
    if (!candidate) return;
    if (!candidate.name.toLowerCase().endsWith('.wav')) { setError(t('upload.errType')); onFile(null); return; }
    if (candidate.size === 0) { setError(t('upload.errEmpty')); onFile(null); return; }
    if (candidate.size > 20 * 1024 * 1024) { setError(t('upload.errSize')); onFile(null); return; }
    setError(''); onFile(candidate);
  }
  const clear = () => { onFile(null); if (input.current) input.current.value = ''; };
  const byId = (id: string) => samples.find(s => s.id === id);
  return <section className="pnl upload" aria-labelledby="upload-title">
    <h2 className="h" id="upload-title"><span>{t('upload.title')}</span></h2>
    <div className="up-row">
      <div className={`drop ${dragging ? 'dragging' : ''}`} data-testid="drop-zone"
        onDragOver={e => { e.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)}
        onDrop={e => { e.preventDefault(); setDragging(false); accept(e.dataTransfer.files[0]); }}>
        <input ref={input} type="file" aria-label={t('upload.inputLabel')} accept=".wav,audio/wav" hidden onChange={e => accept(e.target.files?.[0])} />
        <strong className="drop-title">{t('upload.drop')}</strong>
        <span className="drop-sub">{t('upload.browse')}</span>
      </div>
      <button type="button" className="btn" onClick={() => input.current?.click()}>{t('upload.browseButton')}</button>
      <button type="button" className="btn p" disabled={!file || busy} onClick={onAnalyze}>{busy ? t('upload.analyzing') : t('upload.analyze')}</button>
    </div>
    <div className="samples"><span>{t('upload.try')}</span>
      {(['normal', 'abnormal'] as const).map(id => { const s = byId(id); return <button key={id} type="button" className="btn sm" disabled={!s || busy} data-testid={`sample-${id}`} onClick={() => s && onSample(s)}>{t(`upload.sample.${id}`)}</button>; })}
      {samples.length === 0 && <span className="muted">{samplesWarning ? t('upload.samplesMissing') : ''}</span>}
    </div>
    {error && <p className="inline-error" role="alert">{error}</p>}
    {file && <div className="file-row"><span className="mono file-name">{file.name}</span><span className="muted mono">{(file.size / 1024).toFixed(1)} KB</span>
      <button type="button" className="btn sm ghost" aria-label={t('upload.remove')} onClick={clear}>{t('upload.clear')}</button></div>}
  </section>;
}
