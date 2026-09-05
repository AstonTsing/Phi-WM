#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
LIBERO_PLUS_HOME=${LIBERO_PLUS_HOME:-/root/tianyi/code/LIBERO-plus}
LIBERO_PLUS_PYTHON=${LIBERO_PLUS_PYTHON:-${LIBERO_PLUS_HOME}/.venv/bin/python}
STARVLA_PYTHON=${STARVLA_PYTHON:-/root/tianyi/code/LDA-1B/.venv/bin/python}

DEFAULT_CKPT=/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

ckpt=${1:-${CKPT:-${DEFAULT_CKPT}}}
task_suite_name=${2:-${TASK_SUITE_NAME:-libero_spatial}}
gpu_id=${3:-${GPU_ID:-0}}
base_port=${4:-${PORT:-6550}}
start_idx=${5:-${START_IDX:-0}}
end_idx=${6:-${END_IDX:-1}}
output_dir=${7:-${OUTPUT_DIR:-}}

host=${HOST:-127.0.0.1}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-1}
server_wait_timeout=${SERVER_WAIT_TIMEOUT:-900}
unnorm_key=${UNNORM_KEY:-franka}
image_history=${IMAGE_HISTORY:--1}
multiview_pack=${MULTIVIEW_PACK:-auto}
use_state=${LIBERO_EVAL_USE_STATE:-${USE_STATE:-auto}}
use_state_flag=()
case "${use_state}" in
    auto|Auto|AUTO) use_state_flag=(--args.use-state) ;;
    1|true|True|TRUE|yes|Yes|YES) use_state_flag=(--args.use-state) ;;
    0|false|False|FALSE|no|No|NO) ;;
    *) echo "[ERROR] LIBERO_EVAL_USE_STATE/USE_STATE must be auto, true, or false, got: ${use_state}" >&2; exit 1 ;;
esac
save_videos=${SAVE_VIDEOS:-false}

cd "${STARVLA_DIR}"

export LIBERO_HOME="${LIBERO_PLUS_HOME}"
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_PLUS_HOME}/.libero_config}
export PYTHONPATH="${STARVLA_DIR}:${LIBERO_PLUS_HOME}:${PYTHONPATH:-}"
export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

STANDALONE_ENV_CLEANUP=(
    -u WORLD_SIZE
    -u LOCAL_WORLD_SIZE
    -u RANK
    -u LOCAL_RANK
    -u NODE_RANK
    -u GROUP_RANK
    -u ROLE_RANK
    -u ROLE_WORLD_SIZE
    -u MASTER_ADDR
    -u MASTER_PORT
)

if [[ ! -f "${ckpt}" ]]; then
    echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
    exit 1
fi
if [[ ! -x "${STARVLA_PYTHON}" ]]; then
    echo "[ERROR] STARVLA_PYTHON not executable: ${STARVLA_PYTHON}" >&2
    exit 1
fi
if [[ ! -x "${LIBERO_PLUS_PYTHON}" ]]; then
    echo "[ERROR] LIBERO_PLUS_PYTHON not executable: ${LIBERO_PLUS_PYTHON}" >&2
    echo "[ERROR] Run examples/LIBERO-plus/eval_files/install_liberoplus_uv.sh first." >&2
    exit 1
fi
if [[ ! -f "${LIBERO_PLUS_HOME}/libero/libero/benchmark/task_classification.json" ]]; then
    echo "[ERROR] LIBERO-plus task classification file missing under ${LIBERO_PLUS_HOME}." >&2
    exit 1
fi

if [[ -z "${output_dir}" ]]; then
    if [[ "${ckpt}" == *"/checkpoints/"* ]]; then
        model_root=$(echo "${ckpt}" | awk -F'/checkpoints/' '{print $1}')
    else
        model_root=$(dirname "${ckpt}")
    fi
    ckpt_name=$(basename "${ckpt}" .pt)
    output_dir="${model_root}/libero_plus_eval/${ckpt_name}"
fi

mkdir -p "${output_dir}/logs/${task_suite_name}" "${output_dir}/videos/${task_suite_name}"

server_log="${output_dir}/logs/${task_suite_name}/server_${start_idx}_to_${end_idx}_gpu${gpu_id}_port${base_port}.log"
eval_log="${output_dir}/logs/${task_suite_name}/eval_${start_idx}_to_${end_idx}_gpu${gpu_id}_port${base_port}.log"

server_pid=""
cleanup() {
    local status=$?
    if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
        echo "Stopping policy server PID ${server_pid}"
        kill "${server_pid}" 2>/dev/null || true
        wait "${server_pid}" 2>/dev/null || true
    fi
    exit "${status}"
}
trap cleanup EXIT INT TERM

echo "=========================================="
echo " LIBERO-plus slice evaluation"
echo "=========================================="
echo " checkpoint      : ${ckpt}"
echo " task suite      : ${task_suite_name}"
echo " task slice      : [${start_idx}, ${end_idx})"
echo " GPU             : ${gpu_id}"
echo " port            : ${base_port}"
echo " trials/task     : ${num_trials_per_task}"
echo " save videos     : ${save_videos}"
echo " use state       : ${use_state}"
echo " output dir      : ${output_dir}"
echo " server log      : ${server_log}"
echo " eval log        : ${eval_log}"
echo "=========================================="

env "${STANDALONE_ENV_CLEANUP[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py \
    --ckpt_path "${ckpt}" \
    --port "${base_port}" \
    --use_bf16 \
    > "${server_log}" 2>&1 &
server_pid=$!

echo "Started policy server PID ${server_pid}; waiting for ${host}:${base_port} ..."
wait_start=$(date +%s)
while ! ss -tlnp 2>/dev/null | grep -q ":${base_port} "; do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
        echo "[ERROR] Policy server exited before port ${base_port} became ready." >&2
        tail -120 "${server_log}" >&2 || true
        exit 1
    fi
    if (( $(date +%s) - wait_start > server_wait_timeout )); then
        echo "[ERROR] Timed out waiting for policy server on port ${base_port} after ${server_wait_timeout}s." >&2
        tail -120 "${server_log}" >&2 || true
        exit 1
    fi
    sleep 5
done
echo "Policy server ready on port ${base_port}."

eval_args=(
    --args.pretrained-path "${ckpt}" \
    --args.host "${host}" \
    --args.port "${base_port}" \
    --args.task-suite-name "${task_suite_name}" \
    --args.num-trials-per-task "${num_trials_per_task}" \
    --args.start-idx "${start_idx}" \
    --args.end-idx "${end_idx}" \
    --args.output-dir "${output_dir}" \
    --args.unnorm-key "${unnorm_key}" \
    --args.image-history "${image_history}" \
    --args.multiview-pack "${multiview_pack}"
    "${use_state_flag[@]}"
)
if [[ "${save_videos}" == "true" || "${save_videos}" == "1" || "${save_videos}" == "yes" ]]; then
    eval_args+=(--args.save-videos)
fi

env "${STANDALONE_ENV_CLEANUP[@]}" \
    CUDA_VISIBLE_DEVICES="${gpu_id}" \
    "${LIBERO_PLUS_PYTHON}" ./examples/LIBERO-plus/eval_files/eval_libero.py \
    "${eval_args[@]}" \
    2>&1 | tee "${eval_log}"

echo "LIBERO-plus slice completed: ${task_suite_name} [${start_idx}, ${end_idx})"
