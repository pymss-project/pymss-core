# pymss Server CLI

server 通过 `pymss serve [MODEL]` 启动，行为类似 vLLM。可以不指定模型空载启动，再通过 API 加载或切换模型。

## 安装可选依赖

server 依赖位于 `server` extra，不属于核心包依赖：

```bash
uv sync --extra server
```

或：

```bash
pip install "pymss[server]"
```

## 启动命令

```bash
uv run --extra server pymss serve bs_roformer_voc_hyperacev2 \
  --host 127.0.0.1 \
  --port 8000 \
  --device auto \
  --source modelscope
```

## 启动行为

当前实现的启动顺序：

1. 如果提供 `MODEL`，调用 `resolve_model(MODEL, require_supported=True, require_exists=True)` 检查 catalog 和本地文件。
2. 如果本地文件缺失且错误是 `FileNotFoundError`，调用 `download_model()` 下载模型、配置和辅助文件。
3. 下载后再次 `resolve_model(..., require_exists=True)`。
4. 创建唯一的 `MSSeparator` 实例。
5. 模型加载成功后才开始监听 HTTP 端口。

下载失败或模型加载失败时，server 启动失败，不监听端口。

如果没有提供 `MODEL`，server 不加载模型并直接监听端口。此时 `/health` 返回 `model_loaded=false`，`/v1/models` 返回空列表，推理请求返回 `503 model_not_loaded`。之后可通过 `POST /v1/models/load` 加载模型。

运行中的 server 可以通过 `POST /v1/download-source` 热切换默认下载源。该设置只影响后续 `POST /v1/models/download`，以及 `/v1/models/load` 在本地缺少文件时触发的自动下载；不会卸载、重载或迁移当前已加载模型。

## CLI 参数

全局参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model-dir` | pymss 默认模型目录 | 模型缓存目录；也可通过 `PYMSS_MODEL_DIR` 控制 |

`serve` 子命令参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `MODEL` | 空 | catalog 模型名或 alias；省略时空载启动 |
| `--source` | `modelscope` | 下载源：`modelscope`、`huggingface`、`hf-mirror` |
| `--endpoint` | 空 | 自定义下载 endpoint |
| `--device` | `auto` | `auto`、`cpu`、`cuda`、`mps`、`mlx` |
| `--device-id` | `0` | CUDA device id，可重复 |
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8000` | 监听端口 |
| `--api-key` | 空 | 可选 Bearer token；设置后 `/v1/*` 需要鉴权 |
| `--debug` | `false` | 启用 debug 日志 |
| `--param` | 空 | 推理参数覆盖，格式为 `key=value`，可重复 |
| `--max-audio-seconds` | `600.0` | 单请求 decoded PCM 秒数上限；`0` 表示不限制 |
| `--max-request-bytes` | `536870912` | HTTP body 大小上限，默认 512 MiB |
| `--max-queue-size` | `8` | 正在处理和等待处理的请求数量上限 |
| `--request-timeout-seconds` | `0.0` | 单请求推理超时；`0` 表示不限制 |

## 测试脚本环境变量

`test/server/start_server.sh` 支持以下环境变量：

| 环境变量 | 默认值 | 说明 |
|---|---:|---|
| `MODEL` | `bs_roformer_voc_hyperacev2` | 启动模型；设置为显式空值 `MODEL=` 时空载启动 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `DEVICE` | `auto` | 推理设备 |
| `SOURCE` | `modelscope` | 下载源 |
| `MODEL_DIR` | 空 | 模型缓存目录 |
| `ENDPOINT` | 空 | 自定义下载 endpoint |
| `API_KEY` | 空 | Bearer token |

示例：

```bash
PORT=8010 API_KEY=test-token ./test/server/start_server.sh
```

脚本会把额外参数原样透传给 `pymss serve`：

```bash
./test/server/start_server.sh --max-audio-seconds 120 --request-timeout-seconds 60
```
