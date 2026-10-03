#!/usr/bin/env bash
set -euo pipefail

# === Paths (adapted for this workspace) ===
STARVLA_DIR=${STARVLA_DIR:-/root/htq/code/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/root/htq/code/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/root/htq/code/LDA-1B/.venv/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/root/htq/code/LIBERO/.venv/bin/python}

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}

# === Checkpoint ===
CKPT=${CKPT:-/root/htq/starVLA/playground/Checkpoints/libero/libero_qwenoft_mip_dino_fdm_state7_100k/checkpoints/steps_60000_pytorch_model.pt}

your_ckpt=${CKPT}   
gpu_id=${GPU_ID:-0}
port=${PORT:-6694}
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES="${gpu_id}" "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${your_ckpt}" \
    --port "${port}" \
    --use_bf16

# #################################
