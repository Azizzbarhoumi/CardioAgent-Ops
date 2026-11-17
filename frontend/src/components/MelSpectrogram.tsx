import type { Analysis } from '../types';

export function MelSpectrogram({ result }: { result: Analysis }) {
  return <div className="chart-card"><div className="chart-heading"><div><div className="eyebrow">CNN INPUT</div><h3>Mel spectrogram</h3></div><span>Log power (dB)</span></div><img className="mel-image" src={`data:image/png;base64,${result.mel_spectrogram_png_base64}`} alt="Mel spectrogram with time and Mel-frequency axes" /><div className="axis-caption">Time (seconds) · Mel frequency (Hz)</div></div>;
}
