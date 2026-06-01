# pymss Server CLI

server 通过 `pymss serve MODEL` 启动，行为类似 vLLM，当前不支持运行时热切换。

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

1. 调用 `resolve_model(MODEL, require_supported=True, require_exists=True)` 检查 catalog 和本地文件。
2. 如果本地文件缺失且错误是 `FileNotFoundError`，调用 `download_model()` 下载模型、配置和辅助文件。
3. 下载后再次 `resolve_model(..., require_exists=True)`。
4. 创建唯一的 `MSSeparator` 实例。
5. 模型加载成功后才开始监听 HTTP 端口。

下载失败或模型加载失败时，server 启动失败，不监听端口。

## CLI 参数

全局参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model-dir` | pymss 默认模型目录 | 模型缓存目录；也可通过 `PYMSS_MODEL_DIR` 控制 |

`serve` 子命令参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `MODEL` | 必填 | catalog 模型名或 alias |
| `--source` | `modelscope` | 下载源：`modelscope`、`huggingface`、`hf-mirror` |
| `--endpoint` | 空 | 自定义下载 endpoint |
| `--served-model-name` | `MODEL` | API 暴露名，可重复；多个名称都指向同一个加载模型 |
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
| `MODEL` | `bs_roformer_voc_hyperacev2` | 启动模型 |
| `HOST` | `127.0.0.1` | 监听地址 |
| `PORT` | `8000` | 监听端口 |
| `DEVICE` | `auto` | 推理设备 |
| `SOURCE` | `modelscope` | 下载源 |
| `MODEL_DIR` | 空 | 模型缓存目录 |
| `ENDPOINT` | 空 | 自定义下载 endpoint |
| `SERVED_MODEL_NAME` | 空 | API 暴露名 |
| `API_KEY` | 空 | Bearer token |

示例：

```bash
PORT=8010 API_KEY=test-token ./test/server/start_server.sh
```

脚本会把额外参数原样透传给 `pymss serve`：

```bash
./test/server/start_server.sh --max-audio-seconds 120 --request-timeout-seconds 60
```
