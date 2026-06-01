from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ..logger import get_separation_logger
from ..model_download import download_model
from ..model_registry import create_separator, resolve_model
from .config import ServerConfig


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
class ServerState:
    config: ServerConfig
    separator: object
    entry: object
    resolved: dict
    served_model_names: tuple[str, ...]
    sample_rate: int
    instruments: tuple[str, ...]
    device: str
    logger: object
    limiter: RequestLimiter
    inference_lock: asyncio.Lock

    def is_served_model(self, model):
        return str(model or "") in self.served_model_names


def _dedupe(values):
    result = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def load_state(config):
    logger = get_separation_logger()
    try:
        resolved = resolve_model(config.model, model_dir=config.model_dir, require_supported=True, require_exists=True)
    except FileNotFoundError:
        download_model(config.model, model_dir=config.model_dir, source=config.source, endpoint=config.endpoint)
        resolved = resolve_model(config.model, model_dir=config.model_dir, require_supported=True, require_exists=True)

    separator = create_separator(
        config.model,
        model_dir=config.model_dir,
        device=config.device,
        device_ids=config.device_ids or [0],
        output_format="wav",
        store_dirs="results",
        logger=logger,
        debug=config.debug,
        inference_params=config.inference_params,
    )
    instruments = tuple(str(item) for item in separator.config.training.instruments)
    sample_rate = int(separator.config.audio.get("sample_rate", 44100))
    served_model_names = tuple(_dedupe(config.served_model_names or [config.model]))
    return ServerState(
        config=config,
        separator=separator,
        entry=resolved["entry"],
        resolved=resolved,
        served_model_names=served_model_names,
        sample_rate=sample_rate,
        instruments=instruments,
        device=separator.device,
        logger=logger,
        limiter=RequestLimiter(config.max_queue_size),
        inference_lock=asyncio.Lock(),
    )


def model_card(state, model_id):
    entry = state.entry
    return {
        "id": model_id,
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
            "sample_rate": state.sample_rate,
            "instruments": list(state.instruments),
            "instruments_source": "separator.config.training.instruments",
        },
    }
