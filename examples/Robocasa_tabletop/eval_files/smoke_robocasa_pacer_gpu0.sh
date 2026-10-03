#!/usr/bin/env bash
set -euo pipefail

cd /root/htq/code/starVLA
source /root/htq/code/LDA-1B/.venv/bin/activate

GPU_LIST_STR="0" \
TASK_LIMIT=1 \
RETRY_MISSING=false \
ROBOCASA_EVAL_SEED=7 \
bash examples/Robocasa_tabletop/eval_files/batch_eval_args.sh \
  /root/htq/starVLA/playground/Checkpoints/robocasa/robocasa_pacer_state58_200k/checkpoints/steps_200000_pytorch_model.pt \
  1 30 1 1
