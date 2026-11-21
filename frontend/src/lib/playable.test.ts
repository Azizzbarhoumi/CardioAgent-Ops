import { expect, it } from 'vitest';
import { upsampleWav } from './playable';

function wav(rate: number, samples: number[]) {
  const v = new DataView(new ArrayBuffer(44 + samples.length * 2));
  const w = (o: number, s: string) => { for (let i = 0; i < 4; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, 'data'); v.setUint32(40, samples.length * 2, true); samples.forEach((s, i) => v.setInt16(44 + i * 2, s, true));
  return v;
}

it('2 kHz WAV is repeated x4 to 8 kHz for playback, same duration and values', () => {
  const out = new DataView(upsampleWav(wav(2000, [1, -2, 3]))!);
  expect(out.getUint32(24, true)).toBe(8000);
  const s = Array.from({ length: 12 }, (_, i) => out.getInt16(44 + i * 2, true));
  expect(s).toEqual([1, 1, 1, 1, -2, -2, -2, -2, 3, 3, 3, 3]);
});

it('already playable rates are left alone', () => {
  expect(upsampleWav(wav(44100, [1, 2]))).toBeNull();
});
