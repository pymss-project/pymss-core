"""Device types that actually exist in this process.

Vendor builds expose their accelerator under a private type (``supa`` for BIREN,
``npu`` for Ascend, ``gcu``, ``mlu``), so a hardcoded ``["cpu", "cuda"]``
parameterisation silently skips every domestic-GPU run.  Import ``DEVICES`` and
the suite exercises whatever the running stack provides.
"""
from __future__ import annotations

import torch

_CANDIDATES = ("cuda", "supa", "npu", "gcu", "mlu", "xpu")


def _accel_available(name: str) -> bool:
    if name == "cuda":
        return torch.cuda.is_available()
    module = getattr(torch, name, None)
    try:
        return bool(module is not None and module.is_available() and module.device_count() > 0)
    except Exception:  # noqa: BLE001 - a half-registered backend must not fail collection
        return False


DEVICES: tuple[str, ...] = ("cpu", *(name for name in _CANDIDATES if _accel_available(name)))

__all__ = ["DEVICES"]
