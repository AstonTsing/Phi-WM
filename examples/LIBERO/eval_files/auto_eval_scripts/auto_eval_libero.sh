#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
SCRIPT_PATH=${SCRIPT_PATH:-${STARVLA_DIR}/examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh}

cd "${STARVLA_DIR}"

DEFAULT_CKPT=/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

# Environment-variable friendly configuration for training platforms:
#   CKPT_PATHS_STR="ckpt1 ckpt2"
#   CKPT_PATHS_STR="/path/to/checkpoints"   # directories are expanded
#   CKPT_DIR="/path/to/checkpoints"         # preferred for evaluating all checkpoints
#   TASK_SUITES_STR="libero_spatial libero_object libero_goal libero_10"
#   GPU_LIST_STR="0 1 2 3"
#   BASE_PORT=6450
#   NUM_TRIALS_PER_TASK=50
#   MAX_TASKS=-1
#   LIBERO_EVAL_SEED=7
#   LIBERO_EVAL_SEEDS_STR="7 42 100"
CKPT_DIR=${CKPT_DIR:-}
CKPT_PATHS_STR=${CKPT_PATHS_STR:-${CKPT:-}}
TASK_SUITES_STR=${TASK_SUITES_STR:-"libero_spatial libero_object libero_goal libero_10"}
GPU_LIST_STR=${GPU_LIST_STR:-"0 1 2 3"}
BASE_PORT=${BASE_PORT:-6450}
SLEEP_BETWEEN=${SLEEP_BETWEEN:-20}
LIBERO_EVAL_SEEDS_STR=${LIBERO_EVAL_SEEDS_STR:-${LIBERO_EVAL_SEED:-${SEED:-7}}}
read -r -a EVAL_SEEDS <<< "${LIBERO_EVAL_SEEDS_STR}"

read -r -a TASK_SUITES <<< "${TASK_SUITES_STR}"
read -r -a GPU_LIST <<< "${GPU_LIST_STR}"

CKPT_LIST=()
if [[ -n "${CKPT_DIR}" ]]; then
    mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'steps_*.pt' | sort -V)
elif [[ -n "${CKPT_PATHS_STR}" ]]; then
    read -r -a RAW_CKPT_LIST <<< "${CKPT_PATHS_STR}"
    for ckpt_or_dir in "${RAW_CKPT_LIST[@]}"; do
        if [[ -d "${ckpt_or_dir}" ]]; then
            while IFS= read -r ckpt; do
                CKPT_LIST+=("${ckpt}")
            done < <(find "${ckpt_or_dir}" -maxdepth 1 -type f -name 'steps_*.pt' | sort -V)
        else
            CKPT_LIST+=("${ckpt_or_dir}")
        fi
    done
else
    CKPT_LIST=("${DEFAULT_CKPT}")
fi

if (( ${#CKPT_LIST[@]} == 0 )); then
    echo "[ERROR] No checkpoints configured. Set CKPT_PATHS_STR, CKPT, or CKPT_DIR." >&2
    exit 1
fi
if (( ${#TASK_SUITES[@]} == 0 )); then
    echo "[ERROR] No task suites configured. Set TASK_SUITES_STR." >&2
    exit 1
fi
if (( ${#GPU_LIST[@]} == 0 )); then
    echo "[ERROR] No GPUs configured. Set GPU_LIST_STR." >&2
    exit 1
fi
if (( ${#EVAL_SEEDS[@]} == 0 )); then
    echo "[ERROR] No eval seeds configured. Set LIBERO_EVAL_SEEDS_STR or LIBERO_EVAL_SEED." >&2
    exit 1
fi

num_gpus=${#GPU_LIST[@]}
job_index=0
pids=()
gpu_job_count=()
for ((i=0; i<num_gpus; i++)); do
    gpu_job_count[$i]=0
done

cleanup() {
    local status=$?
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    exit "${status}"
}
trap cleanup INT TERM

echo "=========================================="
echo " Auto Eval LIBERO"
echo "=========================================="
echo " Checkpoints : ${CKPT_LIST[*]}"
echo " Task suites : ${TASK_SUITES[*]}"
echo " GPU list    : ${GPU_LIST[*]}"
echo " Base port   : ${BASE_PORT}"
echo " Seeds       : ${EVAL_SEEDS[*]}"
echo " Script      : ${SCRIPT_PATH}"
echo "=========================================="

failed=0
for ckpt in "${CKPT_LIST[@]}"; do
    if [[ ! -f "${ckpt}" ]]; then
        echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
        exit 1
    fi

    ckpt_name=$(basename "${ckpt}" .pt)
    for eval_seed in "${EVAL_SEEDS[@]}"; do
        export LIBERO_EVAL_SEED="${eval_seed}"
        echo "--- Launching checkpoint: ${ckpt_name} seed=${LIBERO_EVAL_SEED} ---"
        pids=()

        for task in "${TASK_SUITES[@]}"; do
            gpu_idx=$((job_index % num_gpus))
            gpu_id=${GPU_LIST[$gpu_idx]}
            port=$((BASE_PORT + job_index))

            echo "[Job ${job_index}] GPU=${gpu_id} port=${port} ckpt=${ckpt_name} seed=${LIBERO_EVAL_SEED} task=${task}"
            bash "${SCRIPT_PATH}" "${ckpt}" "${task}" "${gpu_id}" "${port}" &
            pids+=($!)

            gpu_job_count[$gpu_idx]=$(( ${gpu_job_count[$gpu_idx]} + 1 ))
            job_index=$((job_index + 1))
            sleep "${SLEEP_BETWEEN}"
        done

        echo "--- Waiting for checkpoint ${ckpt_name} seed=${LIBERO_EVAL_SEED} (${#pids[@]} jobs) ---"
        for pid in "${pids[@]}"; do
            if ! wait "${pid}"; then
                failed=1
            fi
        done
        pids=()
        echo "--- Finished checkpoint: ${ckpt_name} seed=${LIBERO_EVAL_SEED} ---"
    done
done

echo "=========================================="
echo " All evaluations completed (${job_index} jobs total)."
echo " GPU job distribution:"
for ((i=0; i<num_gpus; i++)); do
    echo "   GPU ${GPU_LIST[$i]}: ${gpu_job_count[$i]} jobs"
done
echo "=========================================="

exit "${failed}"
