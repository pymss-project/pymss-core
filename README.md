# pymss-core

[中文文档](README_CN.md)

Core model, configuration, and checkpoint package for music source separation.

`pymss-core` is the shared low-level package for higher-level projects such as
`pymss` inference and `pymsst` training. It contains model definitions,
configuration loading, and checkpoint compatibility helpers. It intentionally
does not include inference DSP pipelines, chunked demixing, audio file I/O,
model downloads, catalog management, CLI, HTTP server, WebUI, datasets, losses,
or training loops.

## Install

```bash
pip install pymss-core
```

For local development:

```bash
uv sync --dev
```

Optional MLX backend on Apple Silicon:

```bash
pip install "pymss-core[mlx]"
```

## Public API

```python
from pymss_core import (
    get_model_from_config,
    load_config,
    load_model_weights,
)

model, config = get_model_from_config("bs_roformer", "config.yaml")
load_model_weights(model, "model.ckpt", model_type="bs_roformer", strict=True)

model.eval()
```

## Package Boundary

Included:

- YAML config loading with `AttrDict`
- PyTorch model definitions under `pymss_core.modules`
- Optional MLX backend implementations for supported model forward paths
- Model factory: `get_model_from_config(model_type, config_path)`
- Checkpoint helpers for common MSS checkpoint containers
- Small model-internal DSP math needed to construct model structures
- VR network structures and VR model parameter JSON files

Excluded:

- Audio file decoding/encoding
- Resampling, preprocessing, and full inference DSP pipelines
- Tensor-level chunked demixing runtime
- Model catalog, aliases, downloads, and cache management
- CLI, server, WebUI, and endpoint schemas
- Dataset, augmentation, loss, metrics, and trainer code
- Any default dependency on MLX, Librosa, tqdm, Lightning, FastAPI, Uvicorn,
  PyAV, WandB, or training extras

## Repository Roles

```text
pymss-core
  shared model/config/checkpoint layer

pymss
  user-facing inference package built on pymss-core, with audio I/O and demix

pymsst
  training package built on pymss-core, with training data/loss/runtime code
```
