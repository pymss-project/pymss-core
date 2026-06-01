#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-bs_roformer_voc_hyperacev2}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
DEVICE="${DEVICE:-auto}"
SOURCE="${SOURCE:-modelscope}"

args=(
  --extra server
  pymss serve "$MODEL"
  --host "$HOST"
  --port "$PORT"
  --device "$DEVICE"
  --source "$SOURCE"
)

if [[ -n "${MODEL_DIR:-}" ]]; then
  args+=(--model-dir "$MODEL_DIR")
fi

if [[ -n "${ENDPOINT:-}" ]]; then
  args+=(--endpoint "$ENDPOINT")
fi

if [[ -n "${SERVED_MODEL_NAME:-}" ]]; then
  args+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "${API_KEY:-}" ]]; then
  args+=(--api-key "$API_KEY")
fi

exec uv run "${args[@]}" "$@"
