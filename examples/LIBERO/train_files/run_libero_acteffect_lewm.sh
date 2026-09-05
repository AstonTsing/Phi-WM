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

export HF_HOME=${HF_HOME:-/root/tianyi/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/root/tianyi/.cache/huggingface/hub}
export TMPDIR=${TMPDIR:-/root/tianyi/.cache/tmp/starvla_acteffect_lewm}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/root/tianyi/.cache/triton/starvla_acteffect_lewm}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/root/tianyi/.cache/torch_extensions/starvla_acteffect_lewm}
mkdir -p "${HF_HUB_CACHE}" "${TMPDIR}" "${TRITON_CACHE_DIR}" "${TORCH_EXTENSIONS_DIR}"

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

Framework_name=${Framework_name:-ActEffectLewm}
base_vlm=${base_vlm:-/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}
dino_model_path=${dino_model_path:-/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m}
config_yaml=${config_yaml:-examples/LIBERO/train_files/starvla_acteffect_lewm_libero.yaml}

data_root_dir=${data_root_dir:-/root/tianyi/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA}
data_mix=${data_mix:-libero_all_video_fdm}

action_dim=${action_dim:-7}
state_dim=${state_dim:-7}
action_horizon=${action_horizon:-8}
future_action_window_size=${future_action_window_size:-7}
mip_t=${mip_t:-0.9}
mip_action_stage0_weight=${mip_action_stage0_weight:-1.0}
mip_action_stage1_weight=${mip_action_stage1_weight:-1.0}
num_inference_timesteps=${num_inference_timesteps:-4}
dino_input_size=${dino_input_size:-224,224}
FDM_LOSS_WEIGHT=${FDM_LOSS_WEIGHT:-0.05}
FDM_STAGE0_WEIGHT=${FDM_STAGE0_WEIGHT:-0.25}
FDM_RANK_WEIGHT=${FDM_RANK_WEIGHT:-0.1}
FDM_RANK_MARGIN=${FDM_RANK_MARGIN:-0.0}
FDM_RANK_TAU=${FDM_RANK_TAU:-0.1}
FDM_HIDDEN_DIM=${FDM_HIDDEN_DIM:-768}
FDM_DEPTH=${FDM_DEPTH:-4}
FDM_NUM_HEADS=${FDM_NUM_HEADS:-8}
FDM_MLP_RATIO=${FDM_MLP_RATIO:-4.0}
FDM_DROPOUT=${FDM_DROPOUT:-0.0}
FDM_MAX_PATCHES=${FDM_MAX_PATCHES:-2048}
FDM_CONDITION_NUM_IMAGES=${FDM_CONDITION_NUM_IMAGES:-2}

freeze_module_list=${freeze_module_list:-dino_model}
run_root_dir=${run_root_dir:-/root/tianyi/starVLA/playground/Checkpoints/libero}
run_id=${run_id:-libero_acteffect_lewm_state7_100k}

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_API_KEY="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"
wandb_entity=${wandb_entity:-mehdizhang2-tsinghua-university}
wandb_project=${wandb_project:-WAM}

echo "=== StarVLA ActEffectLewm LIBERO ==="
echo "STARVLA_ROOT: ${STARVLA_ROOT}"
echo "LDA_VENV: ${LDA_VENV}"
echo "base_vlm: ${base_vlm}"
echo "dino_model_path: ${dino_model_path}"
echo "data_root_dir: ${data_root_dir}"
echo "data_mix: ${data_mix}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "LOAD_ALL_DATA_FOR_TRAINING: ${LOAD_ALL_DATA_FOR_TRAINING}"
echo "MAX_TRAIN_STEPS: ${MAX_TRAIN_STEPS}"
echo "MIP_T: ${mip_t}"
echo "MIP_STAGE0_WEIGHT: ${mip_action_stage0_weight}"
echo "MIP_STAGE1_WEIGHT: ${mip_action_stage1_weight}"
echo "DINO_INPUT_SIZE: ${dino_input_size}"
echo "FDM_LOSS_WEIGHT: ${FDM_LOSS_WEIGHT}"
echo "FDM_STAGE0_WEIGHT: ${FDM_STAGE0_WEIGHT}"
echo "FDM_RANK_WEIGHT: ${FDM_RANK_WEIGHT}"
echo "FDM_RANK_MARGIN: ${FDM_RANK_MARGIN}"
echo "FDM_RANK_TAU: ${FDM_RANK_TAU}"
echo "FDM_HIDDEN_DIM: ${FDM_HIDDEN_DIM}"
echo "FDM_DEPTH: ${FDM_DEPTH}"
echo "FDM_NUM_HEADS: ${FDM_NUM_HEADS}"
echo "FDM_CONDITION_NUM_IMAGES: ${FDM_CONDITION_NUM_IMAGES}"
echo "NUM_INFERENCE_TIMESTEPS: ${num_inference_timesteps}"
echo "NNODES: ${NNODES}"
echo "NPROC_PER_NODE: ${NPROC_PER_NODE}"
echo "MASTER: ${MASTER_ADDR}:${MASTER_PORT}"
echo "RESUME: ${RESUME}"

cd "${STARVLA_ROOT}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE - 1)))}
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

python - <<PY
import os
import torch

expected = int("${NPROC_PER_NODE}")
visible = os.environ.get("CUDA_VISIBLE_DEVICES")
print(f"CUDA preflight | CUDA_VISIBLE_DEVICES={visible}")
print(f"CUDA preflight | torch={torch.__version__}, cuda={torch.version.cuda}")
try:
    available = torch.cuda.is_available()
    count = torch.cuda.device_count()
except Exception as exc:
    raise SystemExit(f"CUDA preflight failed while querying devices: {exc}") from exc
print(f"CUDA preflight | available={available}, device_count={count}, expected_processes={expected}")
if not available or count <= 0:
    raise SystemExit("CUDA preflight failed: no CUDA device is available in this container.")
if count < expected:
    raise SystemExit(
        f"CUDA preflight failed: only {count} visible CUDA device(s), "
        f"but NPROC_PER_NODE={expected}. Choose a matching GPU job or set NPROC_PER_NODE/CUDA_VISIBLE_DEVICES."
    )
PY

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
  --framework.dinov3.enabled true \
  --framework.dinov3.model_path "${dino_model_path}" \
  --framework.dinov3.freeze_dino true \
  --framework.dinov3.input_size "[${dino_input_size}]" \
  --framework.fdm.enabled true \
  --framework.fdm.loss_weight "${FDM_LOSS_WEIGHT}" \
  --framework.fdm.stage0_weight "${FDM_STAGE0_WEIGHT}" \
  --framework.fdm.rank_weight "${FDM_RANK_WEIGHT}" \
  --framework.fdm.rank_margin "${FDM_RANK_MARGIN}" \
  --framework.fdm.rank_tau "${FDM_RANK_TAU}" \
  --framework.fdm.detach_dino true \
  --framework.fdm.detach_action false \
  --framework.fdm.hidden_dim "${FDM_HIDDEN_DIM}" \
  --framework.fdm.depth "${FDM_DEPTH}" \
  --framework.fdm.num_heads "${FDM_NUM_HEADS}" \
  --framework.fdm.mlp_ratio "${FDM_MLP_RATIO}" \
  --framework.fdm.dropout "${FDM_DROPOUT}" \
  --framework.fdm.max_patches "${FDM_MAX_PATCHES}" \
  --framework.fdm.condition_num_images "${FDM_CONDITION_NUM_IMAGES}" \
  --framework.action_model.head_type gr00t_mip \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.action_dim "${action_dim}" \
  --framework.action_model.state_dim "${state_dim}" \
  --framework.action_model.action_horizon "${action_horizon}" \
  --framework.action_model.future_action_window_size "${future_action_window_size}" \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.mip_t "${mip_t}" \
  --framework.action_model.mip_action_stage0_weight "${mip_action_stage0_weight}" \
  --framework.action_model.mip_action_stage1_weight "${mip_action_stage1_weight}" \
  --framework.action_model.num_inference_timesteps "${num_inference_timesteps}" \
  --datasets.vla_data.data_root_dir "${data_root_dir}" \
  --datasets.vla_data.data_mix "${data_mix}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.load_all_data_for_training "${LOAD_ALL_DATA_FOR_TRAINING}" \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules "${freeze_module_list}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.is_resume "${RESUME}" \
  --trainer.learning_rate.base 2.5e-5 \
  --trainer.learning_rate.qwen_vl_interface 1e-5 \
  --trainer.learning_rate.dino_projector 1e-4 \
  --trainer.learning_rate.fdm_predictor 1e-4 \
  --trainer.learning_rate.action_model 1e-4 \
  --run_root_dir "${run_root_dir}" \
  --run_id "${run_id}" \
  --wandb_project "${wandb_project}" \
  --wandb_entity "${wandb_entity}" \
  --is_debug False \
  "$@" 2>&1 | tee ${TEE_MODE} "${TRAIN_LOG}"
