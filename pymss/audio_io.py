import numpy as np
import av


def _frame_to_audio(frame, mono):
    audio = frame.to_ndarray()
    if audio.ndim == 1:
        audio = audio[None, :]
    if mono and audio.shape[0] > 1:
        audio = audio.mean(axis=0, keepdims=True)
    return audio.astype(np.float32, copy=False)


def load_audio(path, sr=None, mono=False, offset=0.0, duration=None):
    chunks = []
    with av.open(path) as container:
        stream = container.streams.audio[0]
        out_rate = int(sr or stream.rate)
        resampler = None
        stop_samples = None if duration is None else int(round((offset + duration) * out_rate))
        decoded = 0

        for frame in container.decode(stream):
            if resampler is None:
                resampler = av.AudioResampler(format="fltp", layout=frame.layout.name, rate=out_rate)

            for out in resampler.resample(frame):
                audio = _frame_to_audio(out, mono)
                chunks.append(audio)
                decoded += audio.shape[-1]
            if stop_samples is not None and decoded >= stop_samples:
                break

        if resampler is not None:
            for out in resampler.resample(None):
                chunks.append(_frame_to_audio(out, mono))

    start = int(round(offset * out_rate))
    stop = None if duration is None else start + int(round(duration * out_rate))
    channels = 1 if mono else 0
    audio = np.concatenate(chunks, axis=-1) if chunks else np.empty((channels, 0), dtype=np.float32)
    audio = np.ascontiguousarray(audio[..., start:stop])
    if mono or audio.shape[0] == 1:
        audio = audio[0]
    return audio, out_rate


def _bitrate_to_int(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    value = str(value).strip().lower()
    if value.endswith("k"):
        return int(float(value[:-1]) * 1000)
    return int(value)


def _format_audio(audio, sample_format):
    audio = np.asarray(audio)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = np.ascontiguousarray(audio)

    if sample_format == "s16p":
        return np.ascontiguousarray((np.clip(audio, -1, 1) * 32767).astype(np.int16).T)
    if sample_format == "s32p":
        return np.ascontiguousarray((np.clip(audio, -1, 1) * (2 ** 31 - 1)).astype(np.int32).T)
    return np.ascontiguousarray(audio.astype(np.float32).T)


def save_audio(path, audio, sr, output_format, audio_params):
    output_format = output_format.lower()
    layout = "stereo" if np.asarray(audio).ndim > 1 and np.asarray(audio).shape[1] == 2 else "mono"

    if output_format == "mp3":
        codec, sample_format = "libmp3lame", "s16p"
    elif output_format == "flac":
        codec, sample_format = "flac", "s32p" if audio_params.get("flac_bit_depth") == "PCM_24" else "s16p"
    else:
        wav_codecs = {
            "PCM_16": ("pcm_s16le", "s16p"),
            "PCM_24": ("pcm_s24le", "s32p"),
            "FLOAT": ("pcm_f32le", "fltp"),
        }
        codec, sample_format = wav_codecs.get(audio_params.get("wav_bit_depth", "FLOAT"), wav_codecs["FLOAT"])

    with av.open(path, "w") as container:
        stream = container.add_stream(codec, rate=int(sr))
        stream.layout = layout
        if output_format == "mp3":
            stream.bit_rate = _bitrate_to_int(audio_params.get("mp3_bit_rate", "320k"))

        frame = av.AudioFrame.from_ndarray(_format_audio(audio, sample_format), format=sample_format, layout=layout)
        frame.sample_rate = int(sr)
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
