from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .audio_io import load_audio, save_audio
from .logger import get_separation_logger

ENSEMBLE_ALGORITHMS = (
    "avg_wave",
    "median_wave",
    "min_wave",
    "max_wave",
    "avg_fft",
    "median_fft",
    "min_fft",
    "max_fft",
)


def _as_channel_first(audio):
    audio = np.asarray(audio, dtype=np.float32)
    return audio[None, :] if audio.ndim == 1 else audio


def stft(wave, nfft=2048, hl=1024):
    wave = _as_channel_first(wave)
    return np.asfortranarray([librosa.stft(np.asfortranarray(channel), n_fft=nfft, hop_length=hl) for channel in wave])


def istft(spec, hl=1024, length=None):
    return np.asfortranarray([librosa.istft(np.asfortranarray(channel), hop_length=hl, length=length) for channel in spec])


def absmax(a, *, axis):
    dims = list(a.shape)
    dims.pop(axis)
    indices = np.ogrid[tuple(slice(0, d) for d in dims)]
    argmax = np.abs(a).argmax(axis=axis)
    indices.insert((len(a.shape) + axis) % len(a.shape), argmax)
    return a[tuple(indices)]


def lambda_min(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmin(key(arr), axis)
    if axis is None:
        return arr.flatten()[idxs]
    idxs = np.expand_dims(idxs, axis)
    result = np.take_along_axis(arr, idxs, axis)
    return result if keepdims else np.squeeze(result, axis=axis)


def lambda_max(arr, axis=None, key=None, keepdims=False):
    idxs = np.argmax(key(arr), axis)
    if axis is None:
        return arr.flatten()[idxs]
    idxs = np.expand_dims(idxs, axis)
    result = np.take_along_axis(arr, idxs, axis)
    return result if keepdims else np.squeeze(result, axis=axis)


def average_waveforms(pred_track, weights=None, algorithm="avg_wave"):
    """
    Combine waveforms using one of the MSST ensemble algorithms.

    Parameters:
        pred_track: array-like with shape (files, channels, samples)
        weights: optional sequence with shape (files,)
        algorithm: one of ENSEMBLE_ALGORITHMS

    Returns:
        Combined waveform with shape (channels, samples).
    """
    if algorithm not in ENSEMBLE_ALGORITHMS:
        raise ValueError(f"Unknown ensemble algorithm: {algorithm}")

    pred_track = np.asarray(pred_track, dtype=np.float32)
    if pred_track.ndim != 3:
        raise ValueError("pred_track must have shape (files, channels, samples)")

    if weights is None:
        weights = np.ones(pred_track.shape[0], dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (pred_track.shape[0],):
        raise ValueError("weights length must match number of input files")
    if algorithm in {"avg_wave", "avg_fft"} and np.isclose(weights.sum(), 0.0):
        raise ValueError("weights must not sum to zero for average ensemble algorithms")

    final_length = pred_track.shape[-1]
    mod_track = []
    for idx in range(pred_track.shape[0]):
        if algorithm == "avg_wave":
            mod_track.append(pred_track[idx] * weights[idx])
        elif algorithm in {"median_wave", "min_wave", "max_wave"}:
            mod_track.append(pred_track[idx])
        elif algorithm in {"avg_fft", "median_fft", "min_fft", "max_fft"}:
            spec = stft(pred_track[idx], nfft=2048, hl=1024)
            mod_track.append(spec * weights[idx] if algorithm == "avg_fft" else spec)

    pred_track = np.asarray(mod_track)
    if algorithm == "avg_wave":
        return pred_track.sum(axis=0) / weights.sum()
    if algorithm == "median_wave":
        return np.median(pred_track, axis=0)
    if algorithm == "min_wave":
        return lambda_min(pred_track, axis=0, key=np.abs)
    if algorithm == "max_wave":
        return lambda_max(pred_track, axis=0, key=np.abs)
    if algorithm == "avg_fft":
        return istft(pred_track.sum(axis=0) / weights.sum(), hl=1024, length=final_length)
    if algorithm == "min_fft":
        return istft(lambda_min(pred_track, axis=0, key=np.abs), hl=1024, length=final_length)
    if algorithm == "max_fft":
        return istft(absmax(pred_track, axis=0), hl=1024, length=final_length)
    if algorithm == "median_fft":
        return istft(np.median(pred_track, axis=0), hl=1024, length=final_length)

    raise AssertionError("unreachable")


def ensemble_audios(files, algorithm="avg_wave", weights=None, logger=None):
    if len(files) < 2:
        raise ValueError("at least two input files are required")

    if weights is None:
        weights = np.ones(len(files), dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (len(files),):
        raise ValueError("weights length must match number of input files")

    data = []
    sample_rate = None
    for file in files:
        path = Path(file)
        if not path.is_file():
            raise FileNotFoundError(f"input audio file not found: {path}")
        audio, sr = load_audio(str(path), sr=None, mono=False)
        audio = _as_channel_first(audio)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError(f"sample rate mismatch: {path} has {sr}, expected {sample_rate}")
        data.append(audio)
        if logger is not None:
            logger.debug("read %s, waveform shape=%s, sample_rate=%s", path, audio.shape, sr)

    channel_counts = {item.shape[0] for item in data}
    if len(channel_counts) != 1:
        raise ValueError("all input files must have the same channel count")

    lengths = [item.shape[-1] for item in data]
    min_length = min(lengths)
    if len(set(lengths)) > 1:
        if logger is not None:
            logger.warning("Input audio files have different lengths. Truncating all to the shortest length.")
        data = [item[..., :min_length] for item in data]

    result = average_waveforms(np.asarray(data), weights=weights, algorithm=algorithm)
    if logger is not None:
        logger.debug("ensemble result shape=%s", result.shape)
    return result.T, sample_rate


def save_ensemble_audio(
    files,
    output,
    algorithm="avg_wave",
    weights=None,
    output_format=None,
    audio_params=None,
    logger=None,
):
    result, sample_rate = ensemble_audios(files, algorithm=algorithm, weights=weights, logger=logger)
    output_path = Path(output)
    if not output_path.suffix and not output_format:
        output_path = output_path.with_suffix(".wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_format = output_format or output_path.suffix.lstrip(".").lower() or "wav"
    save_audio(str(output_path), result, sample_rate, output_format, audio_params or {})
    return output_path


def audio_ensemble(args):
    logger = get_separation_logger()
    output_path = save_ensemble_audio(
        args.files,
        args.output,
        algorithm=args.algorithm,
        weights=args.weights,
        output_format=args.output_format,
        audio_params={
            "wav_bit_depth": args.wav_bit_depth,
            "flac_bit_depth": args.flac_bit_depth,
            "mp3_bit_rate": args.mp3_bit_rate,
            "m4a_bit_rate": args.m4a_bit_rate,
            "m4a_codec": args.m4a_codec,
            "m4a_aac_at_quality": args.m4a_aac_at_quality,
        },
        logger=logger,
    )
    logger.info(f"Saved ensemble audio to {output_path}")
    return 0
