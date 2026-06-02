from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from ..config import load_config
from ..logger import get_separation_logger
from ..model_download import download_model
from ..model_registry import create_separator, resolve_model
from ..separator import INFERENCE_PARAM_TARGETS
from .config import ServerConfig


VR_SUPPORTED_PARAMETERS = {
    "aggression",
    "batch_size",
    "enable_post_process",
    "enable_tta",
    "fuse_conv_bn",
    "high_end_process",
    "mps_model_backend",
    "mps_model_compute_dtype",
    "normalize",
    "post_process_threshold",
    "use_amp",
    "use_channels_last",
    "window_size",
}


class InferenceParameterError(ValueError):
    pass


class RequestLimiter:
    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self.active = 0
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    async def release(self):
        async with self.lock:
            self.active = max(0, self.active - 1)


@dataclass
class LoadedModel:
    separator: object
    entry: object
    resolved: dict
    requested_model: str
    model_id: str
    sample_rate: int
    instruments: tuple[str, ...]
    device: str
    inference_params: dict
    supported_parameters: dict[str, list[str]]
    audio_params: dict = field(default_factory=dict)

    def is_model_id(self, model):
        return str(model or "") == self.model_id


@dataclass
class ServerState:
    config: ServerConfig
    logger: object
    limiter: RequestLimiter
    model_lock: asyncio.Lock
    inference_lock: asyncio.Lock
    loaded: LoadedModel | None = None
    model_loading: bool = False
    model_loading_target: str | None = None

    def is_loaded_model(self, model):
        return self.loaded is not None and self.loaded.is_model_id(model)


def _section(config, section):
    if config is None:
        return None
    if isinstance(config, dict):
        return config.get(section)
    return getattr(config, section, None)


def _contains(section, key):
    if section is None:
        return False
    if isinstance(section, dict):
        return key in section
    return hasattr(section, key)


def _is_parameter_supported(config, model_type, key, section_name):
    if model_type == "vr" and key in VR_SUPPORTED_PARAMETERS:
        return True
    return _contains(_section(config, section_name), key)


def supported_parameters(config, model_type):
    grouped: dict[str, list[str]] = {}
    for key, section_name in INFERENCE_PARAM_TARGETS.items():
        if not _is_parameter_supported(config, model_type, key, section_name):
            continue
        grouped.setdefault(section_name, []).append(key)
    return grouped


def validate_inference_params(params, config, model_type):
    for key in params:
        section_name = INFERENCE_PARAM_TARGETS.get(key)
        if section_name is None:
            raise InferenceParameterError(f"Unknown inference parameter: {key}")
        if not _is_parameter_supported(config, model_type, key, section_name):
            raise InferenceParameterError(f"Inference parameter {key!r} is not supported by this model")


def _preload_config(resolved):
    model_type = resolved["model_type"]
    if model_type == "vr":
        return None
    config_path = resolved.get("config_path")
    return load_config(config_path) if config_path else None


def _resolve_existing_or_download(model, model_dir, source, endpoint):
    try:
        return resolve_model(model, model_dir=model_dir, require_supported=True, require_exists=True)
    except FileNotFoundError:
        download_model(model, model_dir=model_dir, source=source, endpoint=endpoint)
        return resolve_model(model, model_dir=model_dir, require_supported=True, require_exists=True)


def load_model(config, model, source=None, endpoint=None, inference_params=None):
    source = source or config.source
    endpoint = config.endpoint if endpoint is None else endpoint
    params = dict(config.inference_params or {})
    if inference_params is not None:
        params.update(inference_params)

    resolved = _resolve_existing_or_download(model, config.model_dir, source, endpoint)
    pre_config = _preload_config(resolved)
    validate_inference_params(params, pre_config, resolved["model_type"])

    separator = create_separator(
        model,
        model_dir=config.model_dir,
        device=config.device,
        device_ids=config.device_ids or [0],
        output_format="wav",
        store_dirs="results",
        logger=get_separation_logger(),
        debug=config.debug,
        inference_params=params,
    )
    instruments = tuple(str(item) for item in separator.config.training.instruments)
    sample_rate = int(separator.config.audio.get("sample_rate", 44100))
    entry = resolved["entry"]
    model_type = getattr(separator, "model_type", resolved["model_type"])
    return LoadedModel(
        separator=separator,
        entry=entry,
        resolved=resolved,
        requested_model=model,
        model_id=entry.name,
        sample_rate=sample_rate,
        instruments=instruments,
        device=separator.device,
        inference_params=params,
        supported_parameters=supported_parameters(separator.config, model_type),
        audio_params=dict(getattr(separator, "audio_params", {}) or {}),
    )


def close_loaded_model(loaded):
    if loaded is None:
        return
    separator = loaded.separator
    close = getattr(separator, "close", None)
    if close is not None:
        close()


def load_state(config):
    logger = get_separation_logger()
    state = ServerState(
        config=config,
        logger=logger,
        limiter=RequestLimiter(config.max_queue_size),
        model_lock=asyncio.Lock(),
        inference_lock=asyncio.Lock(),
    )
    if config.model:
        state.loaded = load_model(config, config.model)
    return state


def model_card(loaded):
    entry = loaded.entry
    return {
        "id": loaded.model_id,
        "object": "model",
        "created": 0,
        "owned_by": "pymss",
        "pymss": {
            "catalog_name": entry.name,
            "model_type": entry.model_type,
            "architecture": entry.architecture,
            "category": entry.category_path or entry.primary_category,
            "catalog_target_stem": entry.target_stem,
            "supported": entry.supported,
            "sample_rate": loaded.sample_rate,
            "instruments": list(loaded.instruments),
            "instruments_source": "separator.config.training.instruments",
            "supported_parameters": loaded.supported_parameters,
        },
    }
