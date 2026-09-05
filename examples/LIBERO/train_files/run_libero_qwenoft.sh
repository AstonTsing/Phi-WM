#!/usr/bin/env bash
set -euo pipefail

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  NPROC_PER_NODE=8
fi
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

LDA_VENV=${LDA_VENV:-/root/tianyi/code/LDA-1B/.venv}
STARVLA_ROOT=${STARVLA_ROOT:-/root/tianyi/code/starVLA}

export HF_HOME=${HF_HOME:-/root/nas/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/root/nas/.cache/huggingface/hub}
export TMPDIR=${TMPDIR:-/root/nas/.cache/tmp}
mkdir -p "${HF_HUB_CACHE}" "${TMPDIR}"

export PATH="${LDA_VENV}/bin:${PATH}"
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARNING}
export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1000}

PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-16}
LOAD_ALL_DATA_FOR_TRAINING=${LOAD_ALL_DATA_FOR_TRAINING:-true}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-100}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
RESUME=${RESUME:-false}

Framework_name=${Framework_name:-QwenOFT}
base_vlm=${base_vlm:-/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
config_yaml=${config_yaml:-examples/LIBERO/train_files/starvla_qwenoft_libero.yaml}

data_root_dir=${data_root_dir:-/root/tianyi/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all}

action_dim=${action_dim:-7}
state_dim=${state_dim:-7}
action_horizon=${action_horizon:-8}
future_action_window_size=${future_action_window_size:-7}

freeze_module_list=${freeze_module_list:-''}
run_root_dir=${run_root_dir:-/root/tianyi/starVLA/playground/Checkpoints/libero}
run_id=${run_id:-libero_qwenoft_official_100k}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY="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"
wandb_entity=${wandb_entity:-mehdizhang2-tsinghua-university}
wandb_project=${wandb_project:-WAM}

echo "=== StarVLA QwenOFT LIBERO ==="
echo "STARVLA_ROOT: ${STARVLA_ROOT}"
echo "LDA_VENV: ${LDA_VENV}"
echo "base_vlm: ${base_vlm}"
echo "data_root_dir: ${data_root_dir}"
echo "data_mix: ${data_mix}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "LOAD_ALL_DATA_FOR_TRAINING: ${LOAD_ALL_DATA_FOR_TRAINING}"
echo "MAX_TRAIN_STEPS: ${MAX_TRAIN_STEPS}"
echo "NNODES: ${NNODES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "RESUME: ${RESUME}"

cd "${STARVLA_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE - 1)))}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

output_dir=${run_root_dir}/${run_id}
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
TRAIN_LOG=${TRAIN_LOG:-${output_dir}/train.log}
if [[ "${RESUME}" == "true" ]]; then
  TEE_MODE="-a"
else
  TEE_MODE=""
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines "${NNODES}" \
  --num_processes "$((NNODES * NPROC_PER_NODE))" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${config_yaml}" \
  --framework.name "${Framework_name}" \
  --framework.qwenvl.base_vlm "${base_vlm}" \
  --framework.action_model.action_model_type MLP \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.future_action_window_size "${future_action_window_size}" \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.load_all_data_for_training "${LOAD_ALL_DATA_FOR_TRAINING}" \
  --datasets.vla_data.include_state false \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.is_resume "${RESUME}" \
  --trainer.learning_rate.base 2.5e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.action_model 1e-4 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  --is_debug False \
  "$@" 2>&1 | tee ${TEE_MODE} "${TRAIN_LOG}"
