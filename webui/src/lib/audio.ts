export interface PreparedAudio {
  bytes: ArrayBuffer;
  channels: 1 | 2;
  sampleRate: number;
  seconds: number;
}

async function resampleBuffer(buffer: AudioBuffer, targetSampleRate: number): Promise<AudioBuffer> {
  if (buffer.sampleRate === targetSampleRate) {
    return buffer;
  }
  const length = Math.ceil((buffer.length * targetSampleRate) / buffer.sampleRate);
  const offline = new OfflineAudioContext(buffer.numberOfChannels, length, targetSampleRate);
  const source = offline.createBufferSource();
  source.buffer = buffer;
  source.connect(offline.destination);
  source.start(0);
  return offline.startRendering();
}

function downmixToStereo(buffer: AudioBuffer): Float32Array {
  const frames = buffer.length;
  const output = new Float32Array(frames * 2);
  const channels = buffer.numberOfChannels;
  const leftChannels: Float32Array[] = [];
  const rightChannels: Float32Array[] = [];

  for (let channel = 0; channel < channels; channel += 1) {
    const data = buffer.getChannelData(channel);
    if (channel % 2 === 0) {
      leftChannels.push(data);
    } else {
      rightChannels.push(data);
    }
  }
  if (!rightChannels.length) {
    rightChannels.push(leftChannels[0]);
  }

  for (let frame = 0; frame < frames; frame += 1) {
    let left = 0;
    let right = 0;
    for (const channel of leftChannels) {
      left += channel[frame];
    }
    for (const channel of rightChannels) {
      right += channel[frame];
    }
    output[frame * 2] = left / leftChannels.length;
    output[frame * 2 + 1] = right / rightChannels.length;
  }
  return output;
}

function encodeLittleEndianF32(data: Float32Array): ArrayBuffer {
  const buffer = new ArrayBuffer(data.length * 4);
  const view = new DataView(buffer);
  for (let index = 0; index < data.length; index += 1) {
    view.setFloat32(index * 4, data[index], true);
  }
  return buffer;
}

export async function prepareAudioFile(file: File, targetSampleRate: number): Promise<PreparedAudio> {
  const audioContext = new AudioContext();
  try {
    const decoded = await audioContext.decodeAudioData(await file.arrayBuffer());
    const resampled = await resampleBuffer(decoded, targetSampleRate);
    const seconds = resampled.length / resampled.sampleRate;
    if (resampled.numberOfChannels === 1) {
      const mono = new Float32Array(resampled.getChannelData(0));
      return {
        bytes: encodeLittleEndianF32(mono),
        channels: 1,
        sampleRate: targetSampleRate,
        seconds,
      };
    }
    const stereo = downmixToStereo(resampled);
    return {
      bytes: encodeLittleEndianF32(stereo),
      channels: 2,
      sampleRate: targetSampleRate,
      seconds,
    };
  } finally {
    await audioContext.close();
  }
}

export function base64ToBlob(base64: string, contentType: string): Blob {
  const raw = atob(base64);
  const bytes = new Uint8Array(raw.length);
  for (let index = 0; index < raw.length; index += 1) {
    bytes[index] = raw.charCodeAt(index);
  }
  return new Blob([bytes], { type: contentType });
}
