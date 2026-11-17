import type { Analysis } from '../types';
import { WaveformChart } from './WaveformChart';
import { MelSpectrogram } from './MelSpectrogram';

export function AudioCharts({ result }: { result: Analysis }) {
  return <section className="analysis-grid"><WaveformChart result={result}/><MelSpectrogram result={result}/></section>;
}
