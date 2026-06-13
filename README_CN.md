# pymss-core

[English](README.md)

`pymss-core` 是面向音乐源分离的底层模型、配置与 checkpoint 包。它会作为上层项目的共享基础层，例如：

- `pymss`：面向用户的推理包
- `pymsst`：面向训练的训练包

这个包只保留模型推理和训练都需要复用的核心能力：模型定义、配置加载、
checkpoint 兼容处理。它不包含推理 DSP pipeline、分块 demix、音频文件读写、
模型下载、模型 catalog、CLI、HTTP server、WebUI、dataset、loss 或训练循环。

## 安装

```bash
pip install pymss-core
```

本地开发：

```bash
uv sync --dev
```

Apple Silicon 上可选安装 MLX backend：

```bash
pip install "pymss-core[mlx]"
```

## 公共 API

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

## 包边界

包含：

- YAML 配置加载与 `AttrDict`
- `pymss_core.modules` 下的 PyTorch 模型定义
- 支持部分模型 forward 路径的可选 MLX backend 实现
- 模型工厂：`get_model_from_config(model_type, config_path)`
- 常见 MSS checkpoint 容器的加载与 state dict 解析
- 构造模型结构所需的少量内部 DSP 数学函数
- VR 网络结构和 `resources/vr_modelparams/*.json`

不包含：

- 音频文件解码/编码
- 重采样、预处理和完整推理 DSP pipeline
- tensor 级别的分块 demix runtime
- 模型 catalog、别名、下载和缓存管理
- CLI、server、WebUI 和接口 schema
- dataset、augmentation、loss、metric 和 trainer 代码
- 默认安装不会引入 MLX、Librosa、tqdm、Lightning、FastAPI、Uvicorn、
  PyAV、WandB 等训练或产品层依赖

## 关于 VR JSON

`pymss_core/resources/vr_modelparams/*.json` 不是新增的下载数据，而是从旧路径
`pymss/resources/vr_modelparams/*.json` 移动过来的 VR/UVR 模型结构参数。它们是
VR 网络结构的一部分，因此保留在 core 中。

已删除的 `model_catalog.json` 属于模型下载和 catalog 管理层，不属于 core。
完整的 `VRSeparator`、重采样、STFT/ISTFT pipeline 属于上层推理包。

## 仓库分工

```text
pymss-core
  共享的模型、配置、checkpoint 层

pymss
  基于 pymss-core 的用户推理包，包含音频 IO 和 demix

pymsst
  基于 pymss-core 的训练包，包含训练数据、loss 和 runtime
```
