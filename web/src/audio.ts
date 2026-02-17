export function clampNormalize(f32: Float32Array): Float32Array {
  let max = 0;
  for (let i = 0; i < f32.length; i++) max = Math.max(max, Math.abs(f32[i]));
  const scale = max > 1 ? 1 / max : 1;
  const out = new Float32Array(f32.length);
  for (let i = 0; i < f32.length; i++) out[i] = Math.max(-1, Math.min(1, f32[i] * scale));
  return out;
}

export function floatTo16BitPCM(float32: Float32Array): Int16Array {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function writeWavHeader(view: DataView, dataLength: number, sampleRate = 24000): void {
  const blockAlign = 2; // mono 16-bit
  const byteRate = sampleRate * blockAlign;

  // RIFF identifier
  writeString(view, 0, 'RIFF');
  // file length = 36 + data
  view.setUint32(4, 36 + dataLength, true);
  // RIFF type
  writeString(view, 8, 'WAVE');
  // format chunk identifier
  writeString(view, 12, 'fmt ');
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw)
  view.setUint16(20, 1, true);
  // channel count
  view.setUint16(22, 1, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, byteRate, true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, blockAlign, true);
  // bits per sample
  view.setUint16(34, 16, true);
  // data chunk identifier
  writeString(view, 36, 'data');
  // data chunk length
  view.setUint32(40, dataLength, true);
}

function writeString(view: DataView, offset: number, str: string): void {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
}

export function pcmToWavBlob(float32: Float32Array, sampleRate = 24000): Blob {
  const pcm16 = floatTo16BitPCM(float32);
  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);
  writeWavHeader(view, pcm16.length * 2, sampleRate);
  let offset = 44;
  for (let i = 0; i < pcm16.length; i++, offset += 2) view.setInt16(offset, pcm16[i], true);
  return new Blob([view], { type: 'audio/wav' });
}

export async function playInAudioElement(blob: Blob, el: HTMLAudioElement): Promise<void> {
  const url = URL.createObjectURL(blob);
  el.src = url;
  await el.play().catch(() => void 0);
}


