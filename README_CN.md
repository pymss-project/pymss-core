# pymss
用于音乐源分离的 Python 包。
[English](./README.md)  [简体中文]
## 安装
使用 pip 安装 `pymss` 包的示例：
```sh
pip install pymss
```
## 用法
这是一个简单的例子。
```python
from pymss import MSSeparator, get_separation_logger
# 初始化
separator = MSSeparator(
    model_type='htdemucs',
    model_path='path/to/model',
    config_path='path/to/config',
    device='cuda',
    device_ids=[0],
    output_format='wav',
    use_tta=True,
    store_dirs={
        "vocals": "./output/vocals",
        "other": None # None 或缺少此音轨将导致不输出此音轨的文件。 此示例将在 ./output/vocals 中输出人声音轨，并忽略其他（乐器）音轨。 确保键与配置文件匹配。
    },
    audio_params={"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}, # 可以省略
    logger=get_separation_logger(), # 可以省略
    debug=False, # 可以省略
    inference_params={
        "batch_size": 4,
        "overlap_size": 512,
        "chunk_size": 1024,
        "normalize": True
    } # 可以省略
)
# 处理文件夹中的所有音频文件
separator.process_folder('path/to/input_folder')
```
### 参数
- model_type: 模型类型，例如 'htdemucs'。 必须是以下之一
    ['bs_roformer',
    'mel_band_roformer',
    'htdemucs',
    'mdx23c',
    'bandit',
    'bandit_v2',
    'scnet',
    'apollo',
    'vr']
- model_path: 模型文件路径。
- config_path: 配置文件路径。
- device: 设备类型，默认为 'auto'。 必须是以下之一 ['auto', 'cuda', 'mps', 'cpu']
- device_ids: 设备 ID 列表，默认为 [0]。
- output_format: 输出音频格式，默认为 'wav'。 必须是以下之一 ['wav', 'flac', 'mp3']
- use_tta: 是否使用 TTA（测试时增强），默认为 False。 使用 TTA 会使处理时间增加三倍，但质量会略有提高。
- store_dirs: 存储目录，可以是单个文件夹路径或带有乐器键的字典。
- audio_params: 音频参数，包括 wav_bit_depth、flac_bit_depth 和 mp3_bit_rate。 默认为 {"wav_bit_depth": "FLOAT", "flac_bit_depth": "PCM_24", "mp3_bit_rate": "320k"}。
- logger: Logger 实例。 默认为 pymss.get_separation_logger()
- debug: 是否启用调试模式，默认为 False。
- inference_params: 推理参数，包括 batch_size、overlap_size、chunk_size、normalize 和 `cuda_attention_backend`。 默认值均为 None（意味着所有参数都取决于配置文件）。`model_type='vr'` 支持 `batch_size`、`window_size`、`aggression`、`enable_tta`、`enable_post_process`、`post_process_threshold` 和 `high_end_process`。

### CUDA Attention 后端

RoFormer 系列模型在已安装 PyTorch 暴露 cuDNN attention 时默认使用 cuDNN attention，否则使用 PyTorch 默认 SDPA 路径。需要探测式回退时可通过 `inference_params={"cuda_attention_backend": "auto"}` 覆盖。可选值为 `auto`、`default`、`flash`、`cudnn`、`efficient`、`math` 和 `xformers`。`auto` 会优先尝试 cuDNN attention，然后回退到 PyTorch memory-efficient SDPA，再回退到 PyTorch 默认 SDPA。`xformers` 是本地可选安装项，不作为必需依赖。

### Apple Silicon MLX 后端

在 `device='mps'` 时，可以通过 `inference_params` 显式启用可选 MLX 完整 forward：

```python
inference_params={
    "mps_model_backend": "mlx_full",
    "mps_model_compute_dtype": "float16",
}
```

该后端需要本地安装 `mlx`，但当前不会作为 `setup.py` 必需依赖安装。默认推理仍使用 Torch 路径；缺少 MLX 或 backend 运行失败时，非 VR 模型会记录 `_pymss_mlx_full_backend_error` 并回退 Torch。

### 模型兼容性

Demucs 仅支持配置为 `model: htdemucs` 且 `htdemucs.cac: true` 的 HTDemucs checkpoint。当前无外部依赖推理路径不支持 classic `model: demucs`、`model: hdemucs` 和 non-CaC Wiener Demucs 配置。

UVR VR 可通过 `model_type='vr'` 使用，支持已适配的 UVR/VR 系列 `.pth` 权重。输出 stem 名称来自内置 VR 模型列表，例如 `Vocals`、`Instrumental`、`No Echo` 或 `Echo`。

```python
separator = MSSeparator(
    model_type='vr',
    model_path='pretrain/VR_Models/1_HP-UVR.pth',
    device='cuda',
    output_format='wav',
    store_dirs={
        "Vocals": "./output/vocals",
        "Instrumental": "./output/instrumental",
    },
    inference_params={
        "batch_size": 2,
        "window_size": 512,
        "aggression": 5,
    },
)
separator.process_folder('path/to/input_folder')
```

### Hugging Face 配置提醒
一些从 Hugging Face 或 MSST-WebUI 下载的模型配置使用 `inference.num_overlap`。当前优化后的 pymss 路径使用 `inference.overlap_size`。如果配置里只有 `num_overlap`，请手动添加 `overlap_size`，或通过 `inference_params` 传入；否则 pymss 会回退到 50% overlap，推理会慢很多。

推荐快速设置：

```yaml
audio:
  chunk_size: 480000
inference:
  batch_size: 2
  overlap_size: 24000  # chunk_size 的 5%
```

## 贡献
欢迎贡献！
