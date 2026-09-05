#!/usr/bin/env bash
set -euo pipefail

# === Paths (adapted for this workspace) ===
STARVLA_DIR=${STARVLA_DIR:-/root/tianyi/code/starVLA}
LIBERO_HOME=${LIBERO_HOME:-/root/tianyi/code/LIBERO}
LIBERO_Python=${LIBERO_PYTHON:-/root/tianyi/code/LIBERO/.venv/bin/python}

cd "${STARVLA_DIR}"
# === Checkpoint ===
CKPT=${CKPT:-/root/tianyi/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt}

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/.libero_config}

export PYTHONPATH="${LIBERO_HOME}:${STARVLA_DIR}:${PYTHONPATH:-}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host=${HOST:-127.0.0.1}
base_port=${PORT:-6694}
unnorm_key=${UNNORM_KEY:-franka}
your_ckpt=${CKPT}

# export DEBUG=true

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# model_root: playground/Checkpoints/<run_id>
if [[ "${your_ckpt}" == *"/checkpoints/"* ]]; then
    model_root=$(echo "$your_ckpt" | awk -F'/checkpoints/' '{print $1}')
elif [[ "$(basename "$(dirname "${your_ckpt}")")" == "final_model" ]]; then
    model_root=$(dirname "$(dirname "${your_ckpt}")")
else
    model_root=$(dirname "${your_ckpt}")
fi
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=${TASK_SUITE_NAME:-libero_goal}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
max_tasks=${MAX_TASKS:--1}
eval_seed=${LIBERO_EVAL_SEED:-${SEED:-7}}
folder_name="${folder_name}_seed${eval_seed}"
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
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

"${LIBERO_Python}" ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.max-tasks "$max_tasks" \
    --args.seed "$eval_seed" \
    --args.unnorm-key "$unnorm_key" \
    --args.video-out-path "$video_out_path" \
    --args.image-history "$image_history" \
    --args.multiview-pack "$multiview_pack" \
    "${use_state_flag[@]}"
