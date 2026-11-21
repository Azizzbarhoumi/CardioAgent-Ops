// Browsers refuse to play WAV files below ~3 kHz (Chromium: "no supported sources"), and PhysioNet recordings
// are 2 kHz. For LISTENING ONLY we re-encode such files at an integer multiple of the rate (sample repeat,
// so pitch and timing are unchanged). The file sent to the API for analysis is always the original.
const MIN_PLAYABLE_RATE = 8000;

export async function playableUrl(file: File): Promise<string> {
  if (typeof URL.createObjectURL !== 'function') return '';
  try {
    const buf = await file.arrayBuffer();
    const out = upsampleWav(new DataView(buf));
    return URL.createObjectURL(out ? new Blob([out], { type: 'audio/wav' }) : file);
  } catch {
    return URL.createObjectURL(file);
  }
}

/** Returns a new 16-bit PCM WAV at >= 8 kHz, or null when the file is already playable / not plain PCM. */
export function upsampleWav(v: DataView): ArrayBuffer | null {
  const tag = (o: number) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1), v.getUint8(o + 2), v.getUint8(o + 3));
  if (v.byteLength < 44 || tag(0) !== 'RIFF' || tag(8) !== 'WAVE') return null;
  let o = 12, fmt: { ch: number; rate: number; bits: number; format: number } | null = null, data: [number, number] | null = null;
  while (o + 8 <= v.byteLength) {
    const id = tag(o), size = v.getUint32(o + 4, true);
    if (id === 'fmt ') fmt = { format: v.getUint16(o + 8, true), ch: v.getUint16(o + 10, true), rate: v.getUint32(o + 12, true), bits: v.getUint16(o + 22, true) };
    if (id === 'data') { data = [o + 8, Math.min(size, v.byteLength - o - 8)]; break; }
    o += 8 + size + (size % 2);
  }
  if (!fmt || !data || fmt.format !== 1 || fmt.bits !== 16 || fmt.rate >= MIN_PLAYABLE_RATE) return null;
  const k = Math.ceil(MIN_PLAYABLE_RATE / fmt.rate);
  const frames = Math.floor(data[1] / (2 * fmt.ch));
  const outBytes = frames * k * fmt.ch * 2;
  const out = new DataView(new ArrayBuffer(44 + outBytes));
  const w = (off: number, s: string) => { for (let i = 0; i < 4; i++) out.setUint8(off + i, s.charCodeAt(i)); };
  const rate = fmt.rate * k;
  w(0, 'RIFF'); out.setUint32(4, 36 + outBytes, true); w(8, 'WAVE'); w(12, 'fmt '); out.setUint32(16, 16, true);
  out.setUint16(20, 1, true); out.setUint16(22, fmt.ch, true); out.setUint32(24, rate, true); out.setUint32(28, rate * fmt.ch * 2, true);
  out.setUint16(32, fmt.ch * 2, true); out.setUint16(34, 16, true); w(36, 'data'); out.setUint32(40, outBytes, true);
  let p = 44;
  for (let f = 0; f < frames; f++) for (let r = 0; r < k; r++) for (let c = 0; c < fmt.ch; c++) { out.setInt16(p, v.getInt16(data[0] + (f * fmt.ch + c) * 2, true), true); p += 2; }
  return out.buffer;
}
