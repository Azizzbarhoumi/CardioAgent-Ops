import { useEffect, useMemo, useRef, useState } from 'react';
import type { Analysis } from '../types';
import { useI18n } from '../i18n';
import { playableUrl } from '../lib/playable';

const W = 900, H = 200, L = 44, R = 16, TOP = 14, AX = H - 30;

function niceStep(duration: number) {
  for (const s of [0.5, 1, 2, 5, 10, 15, 30]) if (duration / s <= 12) return s;
  return 60;
}

/** Waveform + Mel spectrogram stacked on one time axis, with a playhead driven by the audio element. */
export function SignalPanel({ analysis, file }: { analysis: Analysis; file: File | null }) {
  const { t, lang } = useI18n();
  const loc = lang === 'fr' ? 'fr-FR' : 'en-US';
  const audio = useRef<HTMLAudioElement>(null);
  const [url, setUrl] = useState('');
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const duration = Math.max(analysis.duration_s, 0.001);

  useEffect(() => {
    if (!file || typeof URL.createObjectURL !== 'function') { setUrl(''); return; }
    let next = '', alive = true;
    void playableUrl(file).then(u => { if (alive) { next = u; setUrl(u); } else if (u) URL.revokeObjectURL(u); });
    return () => { alive = false; if (next && typeof URL.revokeObjectURL === 'function') URL.revokeObjectURL(next); };
  }, [file]);

  // rAF loop while playing, so the playhead moves smoothly (timeupdate alone fires ~4x per second)
  useEffect(() => {
    const el = audio.current; if (!el) return;
    let frame = 0;
    const tick = () => { setTime(el.currentTime); if (!el.paused && !el.ended) frame = requestAnimationFrame(tick); };
    const start = () => { setPlaying(true); cancelAnimationFrame(frame); frame = requestAnimationFrame(tick); };
    const stop = () => setPlaying(false);
    el.addEventListener('pause', stop); el.addEventListener('ended', stop);
    const sync = () => setTime(el.currentTime);
    el.addEventListener('play', start); el.addEventListener('seeked', sync); el.addEventListener('timeupdate', sync); el.addEventListener('ended', sync);
    return () => { cancelAnimationFrame(frame); el.removeEventListener('pause', stop); el.removeEventListener('ended', stop); el.removeEventListener('play', start); el.removeEventListener('seeked', sync); el.removeEventListener('timeupdate', sync); el.removeEventListener('ended', sync); };
  }, [url]);

  const { path, ticks } = useMemo(() => {
    const pts = analysis.waveform_points;
    const peak = Math.max(...pts.map(p => Math.abs(p[1])), 0.001);
    const x = (s: number) => L + (s / duration) * (W - L - R);
    const y = (v: number) => (TOP + AX) / 2 - (v / peak) * ((AX - TOP) / 2 - 6);
    const step = niceStep(duration), list: number[] = [];
    for (let s = 0; s <= duration + 1e-9; s += step) list.push(Number(s.toFixed(3)));
    return { path: pts.map(([s, v], i) => `${i ? 'L' : 'M'}${x(s).toFixed(1)} ${y(v).toFixed(1)}`).join(' '), ticks: list.map(s => ({ s, x: x(s) })) };
  }, [analysis.waveform_points, duration]);

  const frac = Math.min(1, Math.max(0, time / duration));
  const phX = L + frac * (W - L - R);
  const box = analysis.mel_spectrogram_axes;
  const melLeft = box ? (box.left + frac * (box.right - box.left)) * 100 : null;

  return <section className="pnl signal" aria-labelledby="signal-title">
    <h2 className="h" id="signal-title"><span>{t('signal.title')}</span>
      <span className="transport">{url && <button type="button" className="btn p" data-testid="play" aria-pressed={playing}
        onClick={() => { const el = audio.current; if (!el) return; if (el.paused) void el.play(); else el.pause(); }}>{t(playing ? 'signal.pause' : 'signal.play')}</button>}
      <span className="timecode mono" data-testid="timecode" aria-live="off">{time.toFixed(1)} s / {analysis.duration_s.toFixed(2)} s</span></span></h2>
    {url
      ? <audio ref={audio} controls src={url} aria-label={t('signal.playerLabel', { name: analysis.filename })} data-testid="audio" />
      : <p className="muted">{t('signal.noAudio')}</p>}
    <div className="scope">
      <h3 className="sub-h">{t('signal.waveform', { rs: analysis.sample_rate_after_resampling.toLocaleString(loc) })}</h3>
      <svg className="wave" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={t('signal.waveformAria')}>
        <line className="ax" x1={L} x2={W - R} y1={AX} y2={AX} />
        <line className="zero" x1={L} x2={W - R} y1={(TOP + AX) / 2} y2={(TOP + AX) / 2} />
        <path className="sig" d={path} />
        {ticks.map(k => <g key={k.s}><line className="ax" x1={k.x} x2={k.x} y1={AX} y2={AX + 5} /><text x={k.x} y={AX + 18} textAnchor="middle">{k.s} s</text></g>)}
        <text x={4} y={TOP + 8}>+1</text><text x={4} y={AX - 2}>−1</text>
        <text x={W - R} y={H - 1} textAnchor="end" className="ax-title">{t('signal.time')}</text>
        <line className="ph" data-testid="wave-playhead" data-x={phX.toFixed(2)} x1={phX} x2={phX} y1={TOP} y2={AX} />
      </svg>
      <h3 className="sub-h">{t('signal.spectrogram')}</h3>
      <div className="mel-wrap">
        <img className="mel-image" src={`data:image/png;base64,${analysis.mel_spectrogram_png_base64}`} alt={t('signal.spectrogramAlt')} />
        {melLeft != null && box && <span className="mel-playhead" data-testid="mel-playhead" data-left={melLeft.toFixed(2)}
          style={{ left: `${melLeft}%`, top: `${box.top * 100}%`, bottom: `${(1 - box.bottom) * 100}%` }} />}
      </div>
      <p className="caption">{t('signal.spectrogramAxes')}</p>
    </div>
  </section>;
}
