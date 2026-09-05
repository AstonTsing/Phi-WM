#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export Framework_name=${Framework_name:-QwenOFTMIPDINOFDMActionQueryRank}
export config_yaml=${config_yaml:-examples/Robocasa_tabletop/train_files/starvla_qwenoft_mip_dino_fdm_action_query_rank_robocasa_gr1.yaml}
export state_dim=${state_dim:-58}
export run_id=${run_id:-robocasa_qwenoft_mip_dino_fdm_action_query_rank_state58_200k}

ACTION_QUERY_LOSS_WEIGHT=${ACTION_QUERY_LOSS_WEIGHT:-0.1}
ACTION_QUERY_HIDDEN_DIM=${ACTION_QUERY_HIDDEN_DIM:-1024}
ACTION_QUERY_RANK_WEIGHT=${ACTION_QUERY_RANK_WEIGHT:-0.1}
ACTION_QUERY_RANK_MARGIN=${ACTION_QUERY_RANK_MARGIN:-0.0}
ACTION_QUERY_RANK_TAU=${ACTION_QUERY_RANK_TAU:-0.1}
ACTION_QUERY_LR=${ACTION_QUERY_LR:-1e-4}
export FDM_STAGE0_WEIGHT=${FDM_STAGE0_WEIGHT:-0.0}

exec bash "${SCRIPT_DIR}/run_robocasa_qwenoft_mip_dino_fdm.sh" \
  --framework.action_query.enabled true \
  --framework.action_query.loss_weight "${ACTION_QUERY_LOSS_WEIGHT}" \
  --framework.action_query.hidden_dim "${ACTION_QUERY_HIDDEN_DIM}" \
  --framework.action_query.rank_weight "${ACTION_QUERY_RANK_WEIGHT}" \
  --framework.action_query.rank_margin "${ACTION_QUERY_RANK_MARGIN}" \
  --framework.action_query.rank_tau "${ACTION_QUERY_RANK_TAU}" \
  --trainer.learning_rate.action_query_head "${ACTION_QUERY_LR}" \
  "$@"
