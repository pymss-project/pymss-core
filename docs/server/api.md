# pymss Server API

server 以单模型方式运行，一个进程最多加载一个 `MSSeparator`。可以不指定模型空载启动，再通过 API 加载或切换模型。

## Endpoint 总览

| 方法 | 路径 | 鉴权 | 说明 |
|---|---|---:|---|
| `GET` | `/health` | 否 | 进程健康检查 |
| `GET` | `/v1/models` | 是 | 返回当前进程暴露的模型列表 |
| `GET` | `/v1/models/{model}` | 是 | 返回指定模型的元数据 |
| `POST` | `/v1/models/load` | 是 | 加载或切换模型 |
| `POST` | `/v1/audio/separations` | 是 | 执行音频源分离 |

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

## GET /v1/models/{model}

`model` 必须匹配当前已加载模型的 catalog `name`。找不到或当前空载时返回 `404 model_not_found`。

响应对象与 `/v1/models` 的 `data[]` 项一致。

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
| `source` | string | 否 | 下载源，默认使用 server 启动配置 |
| `endpoint` | string/null | 否 | 下载 endpoint，默认使用 server 启动配置 |
| `inference_params` | object | 否 | 本次加载模型的推理参数覆盖 |

响应中的 `model.id` 始终是 catalog `name`。加载/切换期间，其它加载请求和推理请求返回 `409 model_operation_in_progress`。

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
