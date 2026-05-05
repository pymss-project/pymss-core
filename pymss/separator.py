import gc
import os
import logging
import torch
import numpy as np
import platform
import subprocess
from time import time
from tqdm import tqdm

from .audio_io import load_audio, save_audio
from .utils import demix, get_model_from_config
from .logger import get_separation_logger, set_log_level


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def _configure_torch_compile_cache(cache_dir):
    cache_dir = cache_dir or '.torchinductor_cache'
    os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR', cache_dir)
    return cache_dir


def _patch_inductor_duplicate_kernel_imports():
    try:
        import torch._dynamo.utils as dynamo_utils
    except Exception:
        return False

    if getattr(dynamo_utils.import_submodule, '_pymss_inductor_patch', False):
        return True

    skip_kernel_modules = {'flex_attention', 'flex_decoding', 'mm_scaled_grouped'}

    def patched_import_submodule(mod):
        import importlib

        base = os.path.dirname(mod.__file__)
        for filename in sorted(os.listdir(base)):
            if not filename.endswith('.py') or filename[0] == '_':
                continue
            name = filename[:-3]
            if mod.__name__ == 'torch._inductor.kernel' and name in skip_kernel_modules:
                continue
            importlib.import_module(f'{mod.__name__}.{name}')

    patched_import_submodule._pymss_inductor_patch = True
    dynamo_utils.import_submodule = patched_import_submodule
    return True


def _select_device(device, device_ids, logger):
    if device not in ['cpu', 'cuda', 'mps']:
        if torch.cuda.is_available():
            logger.debug("CUDA is available in Torch, setting Torch device to CUDA")
            return f'cuda:{device_ids[0]}'
        if torch.backends.mps.is_available():
            logger.debug("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
            return "mps"
        return "cpu"

    if device == "cpu":
        logger.warning("No hardware acceleration could be configured, running in CPU mode")
    return device


def _load_state_dict(model_type, model_path, device):
    if model_type in ['htdemucs', 'apollo']:
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        for key in ('state', 'state_dict'):
            if key in state_dict:
                state_dict = state_dict[key]
        return state_dict
    return torch.load(model_path, map_location=device, weights_only=True)


def _runtime_model_type(model_type, state_dict):
    if model_type == 'bs_roformer' and any('.segm.' in key for key in state_dict):
        return 'bs_roformer_hyperace'
    return model_type


def _model_is_stereo(model_type, config):
    return config.model.get("stereo", True) if model_type in ['bs_roformer', 'bs_roformer_hyperace', 'mel_band_roformer'] else True


def _prepare_mix_channels(mix, is_stereo, logger):
    if is_stereo and len(mix.shape) == 1:
        logger.warning("Track is mono, but model is stereo, adding a second channel.")
        return np.stack([mix, mix], axis=0)
    if is_stereo and len(mix.shape) > 2:
        logger.warning("Track has more than 2 channels, taking mean of all channels and adding a second channel.")
        mono = np.mean(mix, axis=0)
        return np.stack([mono, mono], axis=0)
    if not is_stereo and len(mix.shape) != 1:
        logger.warning("Track has more than 1 channels, but model is mono, taking mean of all channels.")
        return np.mean(mix, axis=0)
    return mix


def _normalize_mix(mix, enabled, logger):
    if not enabled:
        return mix, None

    mono = mix.mean(0)
    mean = mono.mean()
    std = mono.std()
    logger.debug(f"Normalize mix with mean: {mean}, std: {std}")
    return (mix - mean) / std, (mean, std)


def _denormalize(estimates, stats):
    return estimates if stats is None else estimates * stats[1] + stats[0]


def _tta_variants(mix, use_tta, logger):
    if not use_tta:
        return [mix.copy()]
    variants = [mix.copy(), mix[::-1].copy(), -1. * mix.copy()]
    logger.debug(f"User needs to apply TTA, total tracks: {len(variants)}")
    return variants


def _merge_tta_results(results):
    waveforms = results[0]
    for index, result in enumerate(results[1:], start=1):
        for stem, audio in result.items():
            waveforms[stem] += audio[::-1].copy() if index == 1 else -1.0 * audio

    for stem in waveforms:
        waveforms[stem] = waveforms[stem] / len(results)
    return waveforms


def _build_results(waveforms, instruments, mix_orig, config, norm_stats, logger):
    results = {
        instr: _denormalize(waveforms[instr].T, norm_stats)
        for instr in instruments
    }

    target_instrument = config.training.target_instrument
    if target_instrument is None:
        return results

    other_instruments = [instr for instr in config.training.instruments if instr != target_instrument]
    logger.debug(f"target_instrument is not null, extracting instrumental from {target_instrument}, other_instruments: {other_instruments}")
    if other_instruments:
        secondary = other_instruments[0]
        waveforms[secondary] = mix_orig - waveforms[target_instrument]
        results[secondary] = _denormalize(waveforms[secondary].T, norm_stats)
    return results


class MSSeparator:
    def __init__(
            self,
            model_type,
            model_path,
            config_path = None,
            device = 'auto',
            device_ids = [0],
            output_format = 'wav',
            use_tta = False,
            store_dirs = 'results', # str for single folder, dict with instrument keys for multiple folders
            audio_params = {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"},
            logger = get_separation_logger(),
            debug = False,
            inference_params = {
                "batch_size": None,
                "overlap_size": None,
                "chunk_size": None,
                "normalize": None,
                "mask_mode": None,
                "torch_compile": None,
                "torch_compile_mode": None,
                "torch_compile_cache_dir": None,
            }
    ):

        if not model_type:
            raise ValueError('model_type is required')
        if not model_path:
            raise ValueError('model_path is required')

        self.model_type = model_type

        self.model_path = model_path
        self.config_path = config_path if config_path else (model_path + '.yaml')
        self.output_format = output_format
        self.use_tta = use_tta
        self.store_dirs = store_dirs
        self.audio_params = audio_params
        self.logger = logger
        self.debug = debug
        self.inference_params = inference_params

        if self.debug:
            set_log_level(logger, logging.DEBUG)
        else:
            set_log_level(logger, logging.INFO)

        self.log_system_info()
        self.check_ffmpeg_installed()

        self.device_ids = device_ids
        self.device = _select_device(device, self.device_ids, self.logger)

        torch.backends.cudnn.benchmark = True
        self.logger.info(f'Using device: {self.device}, device_ids: {self.device_ids}')

        self.model, self.config = self.load_model()

        if type(self.store_dirs) == str:
            self.store_dirs = {k: self.store_dirs for k in self.config.training.instruments}

        for key in list(self.store_dirs.keys()):
            if key not in self.config.training.instruments and key.lower() not in self.config.training.instruments:
                self.store_dirs.pop(key)
                self.logger.warning(f"Invalid instrument key: {key}, removing from store_dirs")
                self.logger.warning(f"Valid instrument keys: {self.config.training.instruments}")

    def log_system_info(self):
        os_name = platform.system()
        os_version = platform.version()
        self.logger.debug(f"Operating System: {os_name} {os_version}")

        python_version = platform.python_version()
        self.logger.debug(f"Python Version: {python_version}")

        pytorch_version = torch.__version__
        self.logger.debug(f"PyTorch Version: {pytorch_version}")

    def check_ffmpeg_installed(self):
        try:
            ffmpeg_version_output = subprocess.check_output(["ffmpeg", "-version"], text=True)
            first_line = ffmpeg_version_output.splitlines()[0]
            self.logger.debug(f"FFmpeg installed: {first_line}")
        except FileNotFoundError:
            self.logger.warning("FFmpeg is not installed. Please install FFmpeg to use this package.")

    def load_model(self):
        start_time = time()
        state_dict = _load_state_dict(self.model_type, self.model_path, self.device)
        model_type = _runtime_model_type(self.model_type, state_dict)

        model, config = get_model_from_config(model_type, self.config_path)

        self.update_inference_params(config, self.inference_params)
        self.apply_model_inference_config(model, config)

        self.logger.info(f"Separator params: model_type: {model_type}, model_path: {self.model_path}, config_path: {self.config_path}, output_folder: {self.store_dirs}")
        self.logger.info(f"Audio params: output_format: {self.output_format}, audio_params: {self.audio_params}")
        self.logger.info(f"Model params: instruments: {config.training.get('instruments', None)}, target_instrument: {config.training.get('target_instrument', None)}")
        self.logger.debug(f"Model params: batch_size: {config.inference.get('batch_size', None)}, overlap_size: {config.inference.get('overlap_size', None)}, chunk_size: {config.audio.get('chunk_size', None)}, normalize: {config.inference.get('normalize', None)}, use_tta: {self.use_tta}")

        model.load_state_dict(state_dict)

        if len(self.device_ids) > 1:
            model = torch.nn.DataParallel(model, device_ids=self.device_ids)
        model = model.to(self.device)
        model.eval()
        model = self.maybe_compile_model(model, config)

        self.logger.debug(f"Loading model completed, duration: {time() - start_time:.2f} seconds")
        return model, config

    def apply_model_inference_config(self, model, config):
        if hasattr(model, 'set_mask_mode'):
            model.set_mask_mode(config.inference.get('mask_mode', 'no_segm'))

    def maybe_compile_model(self, model, config):
        compile_value = config.inference.get('torch_compile', False)
        if not _as_bool(compile_value):
            return model

        device_type = torch.device(self.device).type
        if device_type != 'cuda':
            self.logger.warning("torch_compile is only enabled for CUDA inference; skipping compile")
            return model

        mode = config.inference.get('torch_compile_mode', None)
        if isinstance(compile_value, str) and compile_value.strip().lower() not in ('1', 'true', 'yes', 'on'):
            mode = compile_value
        mode = mode or 'default'
        if mode == 'reduce-overhead':
            self.logger.warning("torch_compile_mode='reduce-overhead' uses CUDA graphs and is unstable for this model; using 'default'")
            mode = 'default'

        cache_dir = _configure_torch_compile_cache(config.inference.get('torch_compile_cache_dir', '.torchinductor_cache'))
        try:
            import torch._inductor.config as inductor_config
            inductor_config.triton.cudagraphs = False
            inductor_config.triton.cudagraph_trees = False
        except Exception:
            pass

        _patch_inductor_duplicate_kernel_imports()
        torch.set_float32_matmul_precision('high')

        self.logger.info(f"Enabling torch.compile: mode={mode}, cache_dir={cache_dir}")
        return torch.compile(model, mode=mode, fullgraph=False)
    
    def update_inference_params(self, config, params):
        for key, value in {
            'batch_size': 'inference',
            'overlap_size': 'inference',
            'chunk_size': 'audio',
            'normalize': 'inference',
            'mask_mode': 'inference',
            'torch_compile': 'inference',
            'torch_compile_mode': 'inference',
            'torch_compile_cache_dir': 'inference',
        }.items():
            if params.get(key) is not None:
                if key in ('normalize', 'torch_compile'):
                    config[value][key] = params[key]
                elif key in ('mask_mode', 'torch_compile_mode', 'torch_compile_cache_dir'):
                    config[value][key] = params[key]
                else:
                    config[value][key] = int(params[key])
        return config

    def process_folder(self, input_folder):
        if not os.path.isdir(input_folder):
            raise ValueError(f"Input folder '{input_folder}' does not exist.")

        all_mixtures_path = [os.path.join(input_folder, f) for f in os.listdir(input_folder)]

        sample_rate = 44100
        if 'sample_rate' in self.config.audio:
            sample_rate = self.config.audio['sample_rate']
        self.logger.info(f"Input_folder: {input_folder}, Total files found: {len(all_mixtures_path)}, Use sample rate: {sample_rate}")

        if not self.debug:
            all_mixtures_path = tqdm(all_mixtures_path, desc="Total progress")

        success_files = []
        for path in all_mixtures_path:
            if not self.debug:
                all_mixtures_path.set_postfix({'track': os.path.basename(path)})
            try:
                mix, sr = load_audio(path, sr=sample_rate, mono=False)
            except Exception as e:
                self.logger.warning(f'Cannot process track: {path}, error: {str(e)}')
                continue

            self.logger.debug(f"Starting separation process for audio_file: {path}")
            results = self.separate(mix)
            self.logger.debug(f"Separation audio_file: {path} completed. Starting to save results.")

            file_name, _ = os.path.splitext(os.path.basename(path))

            for instr in results.keys():
                save_dir = self.store_dirs.get(instr, "")
                if save_dir and type(save_dir) == str:
                    os.makedirs(save_dir, exist_ok=True)
                    self.save_audio(results[instr], sr, f"{file_name}_{instr}", save_dir)
                    self.logger.debug(f"Saved {instr} for {file_name}_{instr}.{self.output_format} in {save_dir}")
                elif save_dir and type(save_dir) == list:
                    for dir in save_dir:
                        os.makedirs(dir, exist_ok=True)
                        self.save_audio(results[instr], sr, f"{file_name}_{instr}", dir)
                        self.logger.debug(f"Saved {instr} for {file_name}_{instr}.{self.output_format} in {dir}")

            success_files.append(os.path.basename(path))
            del mix, results
            gc.collect()
        return success_files

    def separate(self, mix):
        mix = _prepare_mix_channels(mix, _model_is_stereo(self.model_type, self.config), self.logger)
        target = self.config.training.target_instrument
        instruments = [target] if target is not None else self.config.training.instruments.copy()
        if target is not None:
            self.logger.debug("Target instrument is not null, set primary_stem to target_instrument, secondary_stem will be calculated by mix - target_instrument")

        mix_orig = mix.copy()
        mix, norm_stats = _normalize_mix(mix, self.config.inference.get('normalize', False), self.logger)
        full_result = [
            demix(self.config, self.model, track, self.device, pbar=True, model_type=self.model_type)
            for track in _tta_variants(mix, self.use_tta, self.logger)
        ]

        self.logger.debug("Finished demixing tracks.")
        waveforms = _merge_tta_results(full_result)
        self.logger.debug(f"Starting to extract waveforms for instruments: {instruments}")
        results = _build_results(waveforms, instruments, mix_orig, self.config, norm_stats, self.logger)
        self.logger.debug("Separation process completed.")
        return results

    def save_audio(self, audio, sr, file_name, store_dir):
        output_format = self.output_format.lower()
        file = os.path.join(store_dir, f"{file_name}.{output_format}")
        save_audio(file, audio, sr, output_format, self.audio_params)

    def del_cache(self):
        self.logger.debug("Running garbage collection...")
        gc.collect()
        if "mps" in self.device:
            self.logger.debug("Clearing MPS cache...")
            torch.mps.empty_cache()
        if "cuda" in self.device:
            self.logger.debug("Clearing CUDA cache...")
            torch.cuda.empty_cache()
