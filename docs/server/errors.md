# pymss Server 错误格式

server 返回错误对象格式：

```json
{
  "error": {
    "message": "Model 'foo' is not served by this process.",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

字段说明：

| 字段 | 说明 |
|---|---|
| `error.message` | 可读错误信息 |
| `error.type` | 错误类别，当前主要是 `invalid_request_error` 或 `server_error` |
| `error.param` | 相关参数名；无明确参数时为 `null` |
| `error.code` | 稳定错误码，客户端应优先按该字段处理 |

## 错误码

| HTTP | code | type | 说明 |
|---:|---|---|---|
| 400 | `invalid_request` | `invalid_request_error` | JSON 结构无效、字段类型错误或 `Content-Length` 非法 |
| 400 | `invalid_query_parameter` | `invalid_request_error` | raw PCM binary 请求的 query 参数格式无效 |
| 400 | `invalid_model` | `invalid_request_error` | `model` 为空 |
| 404 | `model_not_found` | `invalid_request_error` | `model` 不匹配当前进程的 served model name |
| 400 | `invalid_audio_format` | `invalid_request_error` | `format` 不是 `pcm_f32le` 或 `pcm_s16le` |
| 400 | `invalid_sample_rate` | `invalid_request_error` | 请求 sample rate 与模型 sample rate 不一致 |
| 400 | `invalid_channel_count` | `invalid_request_error` | `channels` 不是 `1` 或 `2` |
| 400 | `invalid_base64` | `invalid_request_error` | JSON `input.data` 不是合法 base64 |
| 400 | `empty_audio` | `invalid_request_error` | decoded PCM bytes 为空 |
| 400 | `invalid_audio_length` | `invalid_request_error` | PCM bytes 长度不能按 format/channels 对齐 |
| 400 | `invalid_audio_data` | `invalid_request_error` | `pcm_f32le` 输入包含 NaN 或 Inf |
| 400 | `missing_audio_metadata` | `invalid_request_error` | raw PCM binary 请求缺少 `format`、`sample_rate` 或 `channels` |
| 415 | `unsupported_content_type` | `invalid_request_error` | `Content-Type` 不是 `application/json` 或 `application/octet-stream` |
| 400 | `invalid_stem` | `invalid_request_error` | 请求了模型不支持的 stem，或 `stems` 类型非法 |
| 400 | `invalid_response_format` | `invalid_request_error` | `response_format` 不是 `json` 或 `zip` |
| 400 | `invalid_output_audio_format` | `invalid_request_error` | `output_audio_format` 不支持，或 JSON 响应请求了非 `pcm_f32le` |
| 413 | `request_too_large` | `invalid_request_error` | 请求体或音频时长超过 server 限制 |
| 401 | `invalid_api_key` | `invalid_request_error` | Bearer token 缺失或不匹配 |
| 429 | `server_overloaded` | `invalid_request_error` | 推理队列已满 |
| 504 | `separation_timeout` | `invalid_request_error` | 推理超过 `--request-timeout-seconds` |
| 500 | `separation_failed` | `server_error` | 推理或响应编码过程失败 |

## 资源限制相关错误

1. HTTP body 大小由 `--max-request-bytes` 限制。超过时返回 `413 request_too_large`。
2. decoded PCM 时长由 `--max-audio-seconds` 限制。超过时返回 `413 request_too_large`；设置为 `0` 表示不限制。

推理并发由单模型锁串行执行。`--max-queue-size` 限制正在处理和等待处理的请求数量，超过时返回 `429 server_overloaded`。

## 鉴权错误

只有启动时设置了 `--api-key`，`/v1/*` endpoint 才要求：

```http
Authorization: Bearer <api-key>
```

`/health` 不要求鉴权。
