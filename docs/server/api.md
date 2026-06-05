# pymss Server API

server 以单模型方式运行，一个进程最多加载一个 `MSSeparator`。可以不指定模型空载启动，再通过 API 加载或切换模型。

## Endpoint 总览

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---:|---|
| `GET` | `/health` | 否 | 进程健康检查 |
| `GET` | `/v1/models` | 是 | 返回当前进程暴露的模型列表 |
| `GET` | `/v1/models/{model}` | 是 | 返回指定模型的元数据 |
| `GET` | `/v1/catalog/models` | 是 | 返回 pymss catalog 模型及本地状态 |
| `GET` | `/v1/catalog/models/{model}` | 是 | 返回指定 catalog 模型详情 |
| `GET` | `/v1/server/info` | 是 | 返回 WebUI/server 配置信息 |
| `POST` | `/v1/models/load` | 是 | 加载或切换模型 |
| `POST` | `/v1/models/download` | 是 | 下载单个 catalog 模型到本地 |
| `GET` | `/v1/download-source` | 是 | 返回当前默认下载源 |
| `POST` | `/v1/download-source` | 是 | 热切换当前默认下载源 |
| `POST` | `/v1/audio/separations` | 是 | 执行音频源分离 |
| `GET` | `/ui/` | 否 | 启用 `--webui` 时返回浏览器 WebUI |

如果启动时没有设置 `--api-key`，`/v1/*` endpoint 不要求鉴权。设置后，客户端必须发送：

```http
Authorization: Bearer <api-key>
```

## GET /health

响应示例：

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_loading": false,
  "model": "bs_roformer_voc_hyperacev2.ckpt",
  "device": "cuda:0"
}
```

`/health` 不要求鉴权。

空载或加载/切换期间，`model_loaded` 为 `false`。加载/切换期间 `model_loading` 为 `true`，且 `/health` 不等待模型操作完成。

## GET /v1/models

空载或加载/切换期间返回空列表。

响应示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "bs_roformer_voc_hyperacev2.ckpt",
      "object": "model",
      "created": 0,
      "owned_by": "pymss",
      "pymss": {
        "catalog_name": "bs_roformer_voc_hyperacev2.ckpt",
        "model_type": "bs_roformer",
        "architecture": "bs_roformer",
        "category": "vocal/vocal_extraction",
        "catalog_target_stem": "vocals",
        "supported": true,
        "sample_rate": 44100,
        "instruments": ["vocals", "instrument"],
        "instruments_source": "separator.config.training.instruments",
        "supported_parameters": {
          "inference": ["batch_size", "normalize"],
          "audio": ["chunk_size"]
        }
      }
    }
  ]
}
```

`id` 一律使用 catalog entry 的 `name` 字段，不使用请求加载时的 alias 或 stem。`pymss.instruments` 是 `/v1/audio/separations` 的 `stems` 可选值。它来自已加载模型的 `separator.config.training.instruments`，不是从 catalog 的 `target_stem` 推导。

## GET /v1/server/info

返回 WebUI 首屏所需的只读 server 信息。该 endpoint 遵守 `/v1/*` 的 Bearer token 鉴权规则。

```json
{
  "object": "server.info",
  "webui": {
    "enabled": true,
    "path": "/ui/"
  },
  "auth": {
    "api_key_required": true
  },
  "limits": {
    "max_audio_seconds": 600.0,
    "max_request_bytes": 536870912,
    "max_queue_size": 8,
    "request_timeout_seconds": 0.0
  },
  "download_source": {
    "source": "modelscope",
    "endpoint": null
  },
  "model_dir": "/home/user/.cache/pymss/models"
}
```

该 endpoint 不改变任何 server 状态。`limits` 只用于客户端展示和预检查，后端仍会在请求处理时执行真实校验。

## GET /v1/models/{model}

`model` 必须匹配当前已加载模型的 catalog `name`。找不到或当前空载时返回 `404 model_not_found`。

响应对象与 `/v1/models` 的 `data[]` 项一致。

## GET /v1/catalog/models

列出 pymss 包内 catalog 模型，并显示模型文件在当前 `model_dir` 下是否完整。该 endpoint 不表示当前进程已加载模型；当前 loaded model 仍只由 `/v1/models` 表示。

查询参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `supported` | `true` | `true`、`false` 或 `all` |
| `category` | 空 | 按 primary category、secondary category 或 `primary/secondary` 过滤 |
| `local` | `all` | `all`、`complete` 或 `missing` |
| `q` | 空 | 按 `name`、alias、architecture、target stem 做大小写不敏感搜索 |
| `include_files` | `false` | 是否返回每个模型的 required file 状态 |
| `source` | 当前默认源 | 仅用于响应中的远程 URL |
| `endpoint` | 当前默认 endpoint | 仅用于响应中的远程 URL |

响应示例：

```json
{
  "object": "list",
  "data": [
    {
      "id": "bs_roformer_voc_hyperacev2.ckpt",
      "object": "pymss.model_catalog_entry",
      "owned_by": "pymss",
      "pymss": {
        "name": "bs_roformer_voc_hyperacev2.ckpt",
        "aliases": ["bs_roformer_voc_hyperacev2.ckpt", "bs_roformer_voc_hyperacev2"],
        "model_type": "bs_roformer",
        "architecture": "bs_roformer",
        "category": "vocal/vocal_extraction",
        "primary_category": "vocal",
        "secondary_category": "vocal_extraction",
        "target_stem": "vocals",
        "supported": true,
        "unsupported_reason": "",
        "size_bytes": 123456789,
        "local": {
          "complete": false,
          "missing_count": 1,
          "model_dir": "/home/user/.cache/pymss/models"
        },
        "remote": {
          "available": true,
          "source": "modelscope",
          "endpoint": null
        }
      }
    }
  ],
  "pymss": {
    "model_dir": "/home/user/.cache/pymss/models",
    "source": "modelscope",
    "endpoint": null,
    "total": 1
  }
}
```

首版不做远程实时验证。`remote.available=true` 表示该模型存在于随包发布的 catalog。

响应字段说明：

- `id` 固定为 catalog entry 的 `name`。
- `local.complete` 只检查模型主文件、config 文件和 auxiliary 文件是否存在，不计算 sha256。
- `size_bytes` 对模型主文件使用 catalog 记录的大小；config 和 auxiliary 文件在 `pymss.files` 中按本地文件大小返回，缺失时为 `0`。
- 列表首版不分页。

## GET /v1/catalog/models/{model}

查询单个 catalog 模型。`model` 可以是 catalog `name`、alias 或 stem。响应 `id` 始终是 catalog `name`。

详情响应默认包含 `pymss.files`：

```json
[
  {
    "role": "model",
    "relpath": "vocal/vocal_extraction/bs_roformer_voc_hyperacev2.ckpt",
    "exists": true,
    "size_bytes": 123456789,
    "remote_url": "https://www.modelscope.cn/models/baicai1145/pymss/resolve/master/vocal/vocal_extraction/bs_roformer_voc_hyperacev2.ckpt"
  }
]
```

首版不返回绝对 `local_path`。客户端可使用 `pymss.local.model_dir` 和 `files[].relpath` 理解本地位置。

找不到 catalog 模型时返回 `404 model_not_found`。

## POST /v1/models/load

加载模型；如果已有模型，则先卸载当前模型，再加载目标模型。请求同步返回，但加载/卸载会在 worker 线程中执行，不阻塞 `/health`。

请求示例：

```json
{
  "model": "bs_roformer_voc_hyperacev2",
  "source": "modelscope",
  "endpoint": null,
  "inference_params": {
    "batch_size": 2
  }
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `model` | string | 是 | catalog `name`、catalog `aliases`，或 `model_registry` 支持的 stem 名 |
| `source` | string | 否 | 下载源，默认使用当前默认下载源 |
| `endpoint` | string/null | 否 | 下载 endpoint，默认使用当前默认 endpoint，传 `null` 表示本次不用自定义 endpoint |
| `inference_params` | object | 否 | 本次加载模型的推理参数覆盖 |

响应中的 `model.id` 始终是 catalog `name`。请求显式传入的 `source` / `endpoint` 只影响本次加载，不修改当前默认下载源。加载/切换期间，其它加载请求和推理请求返回 `409 model_operation_in_progress`。

## POST /v1/models/download

下载单个 catalog 模型需要的全部文件到当前 `model_dir`。该 endpoint 只下载文件，不加载模型、不创建 `MSSeparator`、不影响当前 loaded model。

请求示例：

```json
{
  "model": "bs_roformer_voc_hyperacev2",
  "source": "modelscope",
  "endpoint": null,
  "force": false,
  "verify": true,
  "timeout_seconds": 30
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `model` | string | 是 | catalog `name`、alias 或 stem |
| `source` | string | 否 | 本次下载源；省略时使用当前默认下载源 |
| `endpoint` | string/null | 否 | 本次 endpoint；省略时使用当前默认 endpoint，传 `null` 表示本次不用自定义 endpoint |
| `force` | bool | 否 | 是否重新下载已存在文件，默认 `false` |
| `verify` | bool | 否 | 是否使用下载逻辑的校验，默认 `true` |
| `timeout_seconds` | number | 否 | 单次网络请求超时，默认 `30` |

响应示例：

```json
{
  "object": "model.download",
  "model": {
    "id": "bs_roformer_voc_hyperacev2.ckpt",
    "object": "pymss.model_catalog_entry",
    "pymss": {
      "local": {
        "complete": true,
        "missing_count": 0,
        "model_dir": "/home/user/.cache/pymss/models"
      }
    }
  },
  "source": "modelscope",
  "endpoint": null,
  "downloaded": ["vocal/vocal_extraction/bs_roformer_voc_hyperacev2.ckpt"],
  "skipped": ["vocal/vocal_extraction/bs_roformer_voc_hyperacev2.yaml"]
}
```

`downloaded` 和 `skipped` 是当前 `model_dir` 下的相对路径。如果模型已完整下载且 `force=false`，返回 `200`，`downloaded=[]`，已有文件进入 `skipped`。

下载 unsupported catalog model 是允许的；后续 `/v1/models/load` 仍会拒绝 unsupported model。

行为与失败语义：

- 这是同步 API，请求在下载完成或失败后返回。
- 下载逻辑在线程池中执行，不阻塞 FastAPI event loop。
- 请求显式传入的 `source` / `endpoint` 只影响本次下载，不修改当前默认下载源。
- 另一个显式下载正在进行时返回 `409 model_download_in_progress`。
- 模型加载/切换正在进行时返回 `409 model_operation_in_progress`。
- catalog 中不存在时返回 `404 model_not_found`。
- `model` 为空时返回 `400 invalid_model`。
- `source` 非法或 `endpoint` 类型非法时返回 `400 invalid_download_source`。
- 下载失败、size/hash 不匹配或网络错误时返回 `500 model_download_failed`。

## GET /v1/download-source

返回当前默认下载源。

```json
{
  "object": "download.source",
  "source": "modelscope",
  "endpoint": null,
  "model_dir": "/home/user/.cache/pymss/models"
}
```

## POST /v1/download-source

热切换当前默认下载源，只影响后续下载和缺失文件自动下载，不影响当前已加载模型。

```json
{
  "source": "huggingface",
  "endpoint": null
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `source` | string | 是 | `modelscope`、`huggingface` 或 `hf-mirror` |
| `endpoint` | string/null | 否 | 省略时保留当前 endpoint；传 `null` 表示清空 |

响应与 `GET /v1/download-source` 一致。下载正在进行时返回 `409 model_download_in_progress`；模型加载/切换正在进行时返回 `409 model_operation_in_progress`。

## POST /v1/audio/separations

执行一次分离。当前实现支持两种输入传输方式：

| Content-Type | 音频位置 | 元数据位置 | 适用场景 |
|---|---|---|---|
| `application/json` | JSON `input.data` base64 | JSON 字段 | 调试、短音频、浏览器/脚本 |
| `application/octet-stream` | HTTP body raw PCM bytes | query 参数 | 长音频、生产调用、高吞吐 |

server 不读取文件路径、不拉取 URL、不解析 mp3/wav/flac/m4a 容器。客户端负责读取本地文件、解码、重采样，并发送 PCM。

### JSON/base64 请求

```json
{
  "model": "bs_roformer_voc_hyperacev2.ckpt",
  "input": {
    "format": "pcm_f32le",
    "sample_rate": 44100,
    "channels": 2,
    "data": "base64-encoded-little-endian-pcm"
  },
  "stems": ["vocals"],
  "response_format": "json",
  "output_audio_format": "pcm_f32le"
}
```

字段：

| 字段 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `model` | string | 是 | 当前已加载模型的 catalog `name` |
| `input.format` | string | 是 | `pcm_f32le` 或 `pcm_s16le` |
| `input.sample_rate` | integer | 是 | 必须等于模型 sample rate |
| `input.channels` | integer | 是 | `1` 或 `2` |
| `input.data` | string | 是 | base64 编码的 interleaved PCM bytes |
| `stems` | string[] 或 string | 否 | 要返回的 stem；省略或空值表示全部 |
| `response_format` | string | 否 | `json` 或 `zip`，默认 `json` |
| `output_audio_format` | string | 否 | `pcm_f32le`、`wav`、`flac`，默认 `pcm_f32le` |

当 `response_format=json` 时，当前实现只允许 `output_audio_format=pcm_f32le`。

### raw PCM binary 请求

```http
POST /v1/audio/separations?model=bs_roformer_voc_hyperacev2.ckpt&format=pcm_f32le&sample_rate=44100&channels=2&stems=vocals&response_format=zip&output_audio_format=wav
Content-Type: application/octet-stream

<raw interleaved little-endian PCM bytes>
```

query 参数：

| 参数 | 类型 | 必填 | 说明 |
|---|---|---:|---|
| `model` | string | 是 | 当前已加载模型的 catalog `name` |
| `format` | string | 是 | `pcm_f32le` 或 `pcm_s16le` |
| `sample_rate` | integer | 是 | 必须等于模型 sample rate |
| `channels` | integer | 是 | `1` 或 `2` |
| `stems` | string | 否 | 逗号分隔 stem；省略或空字符串表示全部 |
| `response_format` | string | 否 | `json` 或 `zip`，默认 `json` |
| `output_audio_format` | string | 否 | `pcm_f32le`、`wav`、`flac`，默认 `pcm_f32le` |

## PCM 格式

`pcm_f32le`：

- little-endian float32。
- stereo wire layout 为 interleaved：`L0, R0, L1, R1, ...`。
- 输入不得包含 NaN 或 Inf。

`pcm_s16le`：

- little-endian signed int16。
- server 内部会转换为 float32。

server 解码后会把 interleaved PCM 转换为 `MSSeparator.separate()` 使用的 channel-first numpy 数据。返回 `pcm_f32le` 时会转回 interleaved little-endian float32 bytes。

## stems 行为

- `stems` 省略、空字符串、空数组或只包含空白项：返回模型配置中的全部 instruments。
- stem 名称大小写不敏感；响应使用模型配置里的原始名称。
- 请求不存在的 stem 返回 `400 invalid_stem`。
- 响应顺序：指定 `stems` 时按请求顺序；省略时按模型 instruments 顺序。
- VR 模型当前会执行完整 VR 推理，再在 HTTP 层按请求的 `stems` 过滤响应内容。

## JSON 响应

当 `response_format=json` 时，响应中的音频始终是 base64 编码的 `pcm_f32le` bytes。

```json
{
  "id": "sep_5f16f0e383f948229f85eadd98ac2c64",
  "object": "audio.separation",
  "created": 1760000000,
  "model": "bs_roformer_voc_hyperacev2.ckpt",
  "outputs": [
    {
      "stem": "vocals",
      "audio": {
        "format": "pcm_f32le",
        "sample_rate": 44100,
        "channels": 2,
        "data": "base64-encoded-little-endian-pcm"
      }
    }
  ],
  "metadata": {
    "input_seconds": 30.0,
    "output_stems": ["vocals"],
    "device": "cuda:0"
  },
  "usage": {
    "type": "duration",
    "seconds": 30.0
  }
}
```

## ZIP 响应

当 `response_format=zip` 时：

- `Content-Type: application/zip`
- 每个 stem 一个音频文件。
- 一定包含 `manifest.json`。
- ZIP entry 使用 server 生成的安全文件名，不包含目录层级。

示例：

```text
manifest.json
0001-vocals.wav
0002-instrument.wav
```

`manifest.json` 示例：

```json
{
  "id": "sep_5f16f0e383f948229f85eadd98ac2c64",
  "object": "audio.separation",
  "model": "bs_roformer_voc_hyperacev2.ckpt",
  "outputs": [
    {
      "stem": "vocals",
      "filename": "0001-vocals.wav",
      "format": "wav",
      "sample_rate": 44100,
      "channels": 2
    }
  ],
  "usage": {
    "type": "duration",
    "seconds": 30.0
  }
}
```
