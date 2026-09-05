#!/usr/bin/env bash
set -euo pipefail

# === Paths (adapted for this workspace) ===
STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/root/tianyi/code/LIBERO}
STARVLA_PYTHON=${STARVLA_PYTHON:-/root/tianyi/code/LDA-1B/.venv/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-/root/tianyi/code/LIBERO/.venv/bin/python}

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_HOME}:${PYTHONPATH:-}"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}

# === Checkpoint ===
CKPT=${CKPT:-/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}

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
