#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STARVLA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
STARVLA_HOST="${STARVLA_HOST:-127.0.0.1}"
STARVLA_PORT="${STARVLA_PORT:-10093}"

if [[ $# -lt 1 ]]; then
    echo "Usage: bash deployment/gento/run_model_server.sh <checkpoint.pt>" >&2
    exit 2
fi

CHECKPOINT="$1"
shift
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Checkpoint not found: ${CHECKPOINT}" >&2
    exit 2
fi

COMMAND=(
    "${STARVLA_PYTHON}"
    -m deployment.model_server.server_policy
    --ckpt_path "${CHECKPOINT}"
    --host "${STARVLA_HOST}"
    --port "${STARVLA_PORT}"
    --use_bf16
)
if [[ -n "${STARVLA_BASE_VLM_PATH:-}" ]]; then
    COMMAND+=(--base-vlm-path "${STARVLA_BASE_VLM_PATH}")
fi
if [[ -n "${STARVLA_DINO_MODEL_PATH:-}" ]]; then
    COMMAND+=(--dino-model-path "${STARVLA_DINO_MODEL_PATH}")
fi

cd "${STARVLA_ROOT}"
exec env -u DEBUG "${COMMAND[@]}" "$@"
