import type { Analysis } from '../types';

export function WaveformChart({ result }: { result: Analysis }) {
  const points=result.waveform_points, values=points.map(p=>p[1]), peak=Math.max(...values.map(Math.abs),0.001);
  const path=points.map(([t,value],i)=>`${i?'L':'M'} ${36+(t/Math.max(result.duration_s,0.001))*720} ${112-(value/peak)*82}`).join(' ');
  return <div className="chart-card"><div className="chart-heading"><div><div className="eyebrow">SIGNAL VIEW</div><h3>Resampled waveform</h3></div><span>{result.sample_rate_after_resampling.toLocaleString()} Hz</span></div><svg className="waveform" viewBox="0 0 790 170" role="img" aria-label="Waveform with time axis in seconds"><line x1="36" y1="30" x2="36" y2="140"/><line x1="36" y1="140" x2="756" y2="140"/><line className="zero-line" x1="36" y1="112" x2="756" y2="112"/><path d={path}/><text x="36" y="160">0 s</text><text x="756" y="160" textAnchor="end">{result.duration_s.toFixed(1)} s</text><text x="12" y="35">+1</text><text x="12" y="139">−1</text></svg><div className="axis-caption">Time (seconds) · Amplitude</div></div>;
}
