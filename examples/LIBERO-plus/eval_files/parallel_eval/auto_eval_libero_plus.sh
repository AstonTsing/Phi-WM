#!/usr/bin/env bash
set -euo pipefail

STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
SCRIPT_PATH=${SCRIPT_PATH:-${STARVLA_DIR}/examples/LIBERO-plus/eval_files/parallel_eval/eval_libero_plus_slice.sh}

cd "${STARVLA_DIR}"

DEFAULT_CKPT=/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt

CKPT_DIR=${CKPT_DIR:-}
CKPT_PATHS_STR=${CKPT_PATHS_STR:-${CKPT:-${DEFAULT_CKPT}}}
TASK_SUITES_STR=${TASK_SUITES_STR:-"libero_10 libero_goal libero_object libero_spatial"}
GPU_LIST_STR=${GPU_LIST_STR:-"0 1 2 3 4 5 6 7"}
SUITE_SLICES_STR=${SUITE_SLICES_STR:-"4 2 1 1"}
BASE_PORT=${BASE_PORT:-6550}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
MAX_TASKS_PER_SUITE=${MAX_TASKS_PER_SUITE:--1}
SAVE_VIDEOS=${SAVE_VIDEOS:-false}
RUN_ID=${RUN_ID:-$(date +"%Y%m%d_%H%M%S")}

declare -A SUITE_SIZES=(
    [libero_10]=2519
    [libero_goal]=2591
    [libero_object]=2518
    [libero_spatial]=2402
)

read -r -a TASK_SUITES <<< "${TASK_SUITES_STR}"
read -r -a GPU_LIST <<< "${GPU_LIST_STR}"
read -r -a SUITE_SLICES <<< "${SUITE_SLICES_STR}"

CKPT_LIST=()
if [[ -n "${CKPT_PATHS_STR}" ]]; then
    read -r -a CKPT_LIST <<< "${CKPT_PATHS_STR}"
elif [[ -n "${CKPT_DIR}" ]]; then
    mapfile -t CKPT_LIST < <(find "${CKPT_DIR}" -maxdepth 1 -type f -name 'steps_*.pt' | sort -V)
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
if (( ${#SUITE_SLICES[@]} != ${#TASK_SUITES[@]} )); then
    echo "[ERROR] SUITE_SLICES_STR must have the same length as TASK_SUITES_STR." >&2
    echo "        TASK_SUITES_STR=${TASK_SUITES_STR}" >&2
    echo "        SUITE_SLICES_STR=${SUITE_SLICES_STR}" >&2
    exit 1
fi

cleanup_pids=()
cleanup() {
    local status=$?
    for pid in "${cleanup_pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    exit "${status}"
}
trap cleanup INT TERM

echo "=========================================="
echo " Auto Eval LIBERO-plus"
echo "=========================================="
echo " Checkpoints       : ${CKPT_LIST[*]}"
echo " Task suites       : ${TASK_SUITES[*]}"
echo " Suite slices      : ${SUITE_SLICES[*]}"
echo " GPU list          : ${GPU_LIST[*]}"
echo " Base port         : ${BASE_PORT}"
echo " Trials/task       : ${NUM_TRIALS_PER_TASK}"
echo " Max tasks/suite   : ${MAX_TASKS_PER_SUITE}"
echo " Save videos       : ${SAVE_VIDEOS}"
echo " Run id            : ${RUN_ID}"
echo " Script            : ${SCRIPT_PATH}"
echo "=========================================="

failed=0
for ckpt in "${CKPT_LIST[@]}"; do
    if [[ ! -f "${ckpt}" ]]; then
        echo "[ERROR] Checkpoint not found: ${ckpt}" >&2
        exit 1
    fi

    if [[ "${ckpt}" == *"/checkpoints/"* ]]; then
        model_root=$(echo "${ckpt}" | awk -F'/checkpoints/' '{print $1}')
    else
        model_root=$(dirname "${ckpt}")
    fi
    ckpt_name=$(basename "${ckpt}" .pt)
    output_dir="${OUTPUT_DIR:-${model_root}/libero_plus_eval/${ckpt_name}_${RUN_ID}}"
    mkdir -p "${output_dir}"

    echo "=========================================="
    echo " Checkpoint: ${ckpt}"
    echo " Output dir: ${output_dir}"
    echo "=========================================="

    job_suite=()
    job_start=()
    job_end=()
    for suite_idx in "${!TASK_SUITES[@]}"; do
        suite=${TASK_SUITES[$suite_idx]}
        slices=${SUITE_SLICES[$suite_idx]}
        if [[ -z "${SUITE_SIZES[$suite]:-}" ]]; then
            echo "[ERROR] Unknown LIBERO-plus suite: ${suite}" >&2
            exit 1
        fi
        suite_size=${SUITE_SIZES[$suite]}
        if (( MAX_TASKS_PER_SUITE > 0 && MAX_TASKS_PER_SUITE < suite_size )); then
            suite_size=${MAX_TASKS_PER_SUITE}
        fi
        if (( slices <= 0 )); then
            echo "[ERROR] Invalid slice count ${slices} for suite ${suite}" >&2
            exit 1
        fi

        base_size=$((suite_size / slices))
        remainder=$((suite_size % slices))
        start_idx=0
        for ((slice=0; slice<slices; slice++)); do
            chunk=${base_size}
            if (( slice < remainder )); then
                chunk=$((chunk + 1))
            fi
            end_idx=$((start_idx + chunk))
            if (( end_idx > suite_size )); then
                end_idx=${suite_size}
            fi
            if (( start_idx < end_idx )); then
                job_suite+=("${suite}")
                job_start+=("${start_idx}")
                job_end+=("${end_idx}")
            fi
            start_idx=${end_idx}
        done
    done

    total_jobs=${#job_suite[@]}
    echo "--- Prepared ${total_jobs} LIBERO-plus slice job(s). ---"

    batch_pids=()
    for job_idx in "${!job_suite[@]}"; do
        slot=$((job_idx % ${#GPU_LIST[@]}))
        gpu_id=${GPU_LIST[$slot]}
        port=$((BASE_PORT + slot))
        suite=${job_suite[$job_idx]}
        start_idx=${job_start[$job_idx]}
        end_idx=${job_end[$job_idx]}

        if (( job_idx > 0 && slot == 0 )); then
            echo "--- Waiting for current GPU batch to finish... ---"
            for pid in "${batch_pids[@]}"; do
                if ! wait "${pid}"; then
                    failed=1
                fi
            done
            batch_pids=()
            cleanup_pids=()
        fi

        echo "[Job ${job_idx}/${total_jobs}] GPU=${gpu_id} port=${port} suite=${suite} slice=[${start_idx},${end_idx})"
        NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
        SAVE_VIDEOS="${SAVE_VIDEOS}" \
        bash "${SCRIPT_PATH}" "${ckpt}" "${suite}" "${gpu_id}" "${port}" "${start_idx}" "${end_idx}" "${output_dir}" &
        pid=$!
        batch_pids+=("${pid}")
        cleanup_pids+=("${pid}")
        sleep 5
    done

    echo "--- Waiting for final GPU batch to finish... ---"
    for pid in "${batch_pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    batch_pids=()
    cleanup_pids=()

    echo "--- Aggregating LIBERO-plus results for ${ckpt_name} ---"
    /root/tianyi/code/LDA-1B/.venv/bin/python \
        ./examples/LIBERO-plus/eval_files/parallel_eval/aggregate_results.py \
        --root_path "${output_dir}"
    echo "Aggregated results: ${output_dir}/overall_results.json"
done

exit "${failed}"
