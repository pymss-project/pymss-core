from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from numpy.typing import NDArray
from typing import Dict

from .config import load_config


def get_model_from_config(model_type, config_path):
    config = load_config(config_path)

    if model_type == 'mdx23c':
        from .modules.mdx23c_tfc_tdf_v3 import TFC_TDF_net
        return TFC_TDF_net(config), config
    elif model_type == 'htdemucs':
        from .modules.demucs4ht import get_model
        return get_model(config), config
    elif model_type == 'mel_band_roformer':
        from .modules.bs_roformer import MelBandRoformer
        return MelBandRoformer(**dict(config.model)), config
    elif model_type == 'bs_roformer':
        from .modules.bs_roformer import BSRoformer
        return BSRoformer(**dict(config.model)), config
    elif model_type == 'bs_roformer_hyperace':
        from .modules.bs_roformer import BSRoformerHyperACE
        return BSRoformerHyperACE(**dict(config.model)), config
    elif model_type == 'bandit':
        from .modules.bandit.core.model import MultiMaskMultiSourceBandSplitRNNSimple
        return MultiMaskMultiSourceBandSplitRNNSimple(**config.model), config
    elif model_type == 'bandit_v2':
        from .modules.bandit_v2.bandit import Bandit
        return Bandit(**config.kwargs), config
    elif model_type == 'scnet':
        from .modules.scnet import SCNet
        return SCNet(**config.model), config
    elif model_type == 'apollo':
        from .modules.look2hear.apollo import Apollo
        return Apollo(**config.model), config
    raise ValueError(f"Model type {model_type} not supported")

def _getWindowingArray(window_size, fade_size):
    if fade_size <= 0:
        return torch.ones(window_size)

    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window = torch.ones(window_size)
    window[-fade_size:] *= fadeout
    window[:fade_size] *= fadein
    return window


def _build_chunk_plan(total_length, chunk_size, step, fade_size):
    starts = list(range(0, total_length, step))
    windows = []
    normal_window = _getWindowingArray(chunk_size, fade_size)

    for start in starts:
        length = min(chunk_size, total_length - start)
        window = normal_window
        if start == 0 or start + length >= total_length:
            window = normal_window.clone()
            if start == 0:
                window[:fade_size] = 1
            if start + length >= total_length:
                fade_start = max(0, length - fade_size)
                window[fade_start:length] = 1
        windows.append(window)

    return starts, windows


def _get_inference_step(config, chunk_size):
    overlap_size = int(config.inference.get('overlap_size', chunk_size // 2))
    if overlap_size < 0 or overlap_size >= chunk_size:
        raise ValueError("inference.overlap_size must be >= 0 and < audio.chunk_size")
    return chunk_size - overlap_size


def _complete_chunk_count(total_length, chunk_size, step):
    if total_length < chunk_size:
        return 0
    return (total_length - chunk_size) // step + 1


def _fold_windows(counter, windows, step, start_offset=0):
    n_chunks = windows.shape[0]
    if n_chunks == 0:
        return

    chunk_size = windows.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    counter_columns = windows.transpose(0, 1).unsqueeze(0)
    folded_counter = nn.functional.fold(
        counter_columns,
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    counter[..., start_offset:start_offset + output_length] += folded_counter.view(1, 1, output_length)


def _fold_chunk_batch(result, chunks, windows, step, start_offset=0):
    n_chunks = chunks.shape[0]
    if n_chunks == 0:
        return

    chunk_size = chunks.shape[-1]
    output_length = (n_chunks - 1) * step + chunk_size
    n_sources, n_channels = chunks.shape[1:3]

    weighted = chunks * windows[:, None, None, :]
    columns = weighted.permute(1, 2, 3, 0).reshape(1, n_sources * n_channels * chunk_size, n_chunks)
    folded = nn.functional.fold(
        columns,
        output_size=(1, output_length),
        kernel_size=(1, chunk_size),
        stride=(1, step),
    )
    result[..., start_offset:start_offset + output_length] += folded.view(n_sources, n_channels, output_length)


def _ensure_source_dim(x, chunk_batch):
    if x.ndim == chunk_batch.ndim:
        return x.unsqueeze(1)
    return x


def _fit_tensor_length(x, length):
    if x.shape[-1] > length:
        return x[..., :length]
    if x.shape[-1] < length:
        return nn.functional.pad(x, (0, length - x.shape[-1]))
    return x


def _autocast(device, enabled):
    if torch.device(device).type == 'cuda' and enabled:
        return torch.amp.autocast('cuda')
    return nullcontext()


def _source_names(config):
    if config.training.target_instrument is None:
        return config.training.instruments
    return [config.training.target_instrument]


def _source_count(config):
    return len(_source_names(config))


def _sources_to_dict(config, estimated_sources):
    return {k: v for k, v in zip(_source_names(config), estimated_sources)}


def _prepare_mix_for_chunks(mix, border):
    length_init = mix.shape[-1]
    if mix.ndim == 1:
        mix = mix.unsqueeze(0)
    if length_init > 2 * border and border > 0:
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
    return mix, length_init


def _init_overlap_buffers(config, mix, device, use_fast_path):
    req_shape = (_source_count(config),) + tuple(mix.shape)
    result_device = device if use_fast_path else 'cpu'
    counter_shape = (1, 1, mix.shape[1]) if use_fast_path else req_shape
    result = torch.zeros(req_shape, dtype=torch.float32, device=result_device)
    counter = torch.zeros(counter_shape, dtype=torch.float32, device=result_device)
    return result, counter


def _model_mix(mix, device):
    return mix.to(device) if torch.device(device).type != 'cpu' else mix


def _run_model_chunk(model, arr, chunk_size):
    x = _ensure_source_dim(model(arr), arr).float()
    return _fit_tensor_length(x, chunk_size)


def _extract_chunk(mix, start, chunk_size):
    length = min(chunk_size, mix.shape[1] - start)
    part = mix[:, start:start + chunk_size]
    if length == chunk_size:
        return part, length
    if length > chunk_size // 2 + 1:
        part = nn.functional.pad(part, (0, chunk_size - length), mode='reflect')
    else:
        part = nn.functional.pad(part, (0, chunk_size - length, 0, 0), mode='constant', value=0)
    return part, length


def _add_weighted_chunk(result, counter, chunk, window, start, length):
    device = result.device
    window = window.to(device=device, dtype=torch.float32)[:length]
    result[..., start:start + length] += chunk[..., :length].to(device=device, dtype=torch.float32) * window
    counter[..., start:start + length] += window


def _run_complete_chunks(model, mix, windows, result, counter, chunk_size, step, batch_size, progress_bar):
    n_chunks = _complete_chunk_count(mix.shape[1], chunk_size, step)
    if n_chunks == 0:
        return 0

    inputs = mix.unfold(-1, chunk_size, step).permute(1, 0, 2)[:n_chunks]
    fold_windows = torch.stack(windows[:n_chunks], dim=0).to(device=result.device, dtype=torch.float32)
    _fold_windows(counter, fold_windows, step)

    for batch_start in range(0, n_chunks, batch_size):
        batch_end = min(batch_start + batch_size, n_chunks)
        chunks = _run_model_chunk(model, inputs[batch_start:batch_end].contiguous(), chunk_size)
        _fold_chunk_batch(
            result,
            chunks,
            fold_windows[batch_start:batch_end],
            step,
            start_offset=batch_start * step,
        )
        if progress_bar:
            progress_bar.update(step * (batch_end - batch_start))

    return n_chunks


def _run_tail_chunks(model, mix, starts, windows, result, counter, chunk_size, step, batch_size, first_chunk, progress_bar):
    for batch_start in range(first_chunk, len(starts), batch_size):
        batch_indices = range(batch_start, min(batch_start + batch_size, len(starts)))
        batch_data = []
        batch_locations = []

        for idx in batch_indices:
            chunk, length = _extract_chunk(mix, starts[idx], chunk_size)
            batch_data.append(chunk)
            batch_locations.append((idx, starts[idx], length))

        chunks = _run_model_chunk(model, torch.stack(batch_data, dim=0), chunk_size)
        for j, (idx, start, length) in enumerate(batch_locations):
            _add_weighted_chunk(result, counter, chunks[j], windows[idx], start, length)

        if progress_bar:
            progress_bar.update(step * len(batch_data))


def _finalize_overlap(result, counter, length_init, border):
    estimated_sources = (result / counter).cpu().numpy()
    np.nan_to_num(estimated_sources, copy=False, nan=0.0)
    if length_init > 2 * border and border > 0:
        estimated_sources = estimated_sources[..., border:-border]
    return estimated_sources


def demix_track(config, model, mix, device, pbar=False):
    C = config.audio.chunk_size
    step = _get_inference_step(config, C)
    border = C - step
    fade_size = min(C // 10, border)
    batch_size = config.inference.batch_size

    mix, length_init = _prepare_mix_for_chunks(mix, border)
    chunk_starts, chunk_windows = _build_chunk_plan(mix.shape[1], C, step, fade_size)
    device_type = torch.device(device).type
    use_complete_fast_path = device_type in ('cuda', 'cpu')
    mix_device = _model_mix(mix, device)

    with _autocast(device, config.training.get('use_amp', True)):
        with torch.inference_mode():
            result, counter = _init_overlap_buffers(config, mix, device, use_complete_fast_path)
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            complete_chunks = 0
            if use_complete_fast_path:
                complete_chunks = _run_complete_chunks(
                    model, mix_device, chunk_windows, result, counter, C, step, batch_size, progress_bar
                )

            _run_tail_chunks(
                model, mix_device, chunk_starts, chunk_windows, result, counter, C, step, batch_size, complete_chunks, progress_bar
            )


            if progress_bar:
                progress_bar.close()

            estimated_sources = _finalize_overlap(result, counter, length_init, border)

    return _sources_to_dict(config, estimated_sources)


def demix_track_demucs(config, model, mix, device, pbar=False):
    S = len(config.training.instruments)
    C = config.training.samplerate * config.training.segment
    batch_size = config.inference.batch_size
    step = _get_inference_step(config, C)
    # logger.info(S, C, step, mix.shape, mix.device)

    with _autocast(device, config.training.get('use_amp', True)):
        with torch.inference_mode():
            req_shape = (S, ) + tuple(mix.shape)
            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            progress_bar = tqdm(total=mix.shape[1], desc="Processing audio chunks", leave=False) if pbar else None

            while i < mix.shape[1]:
                # logger.info(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step


                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)
                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu()
                        counter[..., start:start+l] += 1.
                    batch_data = []
                    batch_locations = []

                if progress_bar:
                    progress_bar.update(step)

            if progress_bar:
                progress_bar.close()

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

    return {k: v for k, v in zip(config.training.instruments, estimated_sources)} if S > 1 else estimated_sources

def demix(config, model, mix: NDArray, device, pbar=False, model_type: str = None) -> Dict[str, NDArray]:
    mix = torch.tensor(mix, dtype=torch.float32)
    if model_type == 'htdemucs':
        return demix_track_demucs(config, model, mix, device, pbar=pbar)
    return demix_track(config, model, mix, device, pbar=pbar)
