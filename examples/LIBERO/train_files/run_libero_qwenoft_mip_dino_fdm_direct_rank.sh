#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export Framework_name=${Framework_name:-QwenOFTMIPDINOFDMDirectRank}
export config_yaml=${config_yaml:-examples/LIBERO/train_files/starvla_qwenoft_mip_dino_fdm_direct_rank_libero.yaml}
export state_dim=${state_dim:-7}
export run_id=${run_id:-libero_qwenoft_mip_dino_fdm_direct_rank_state7_100k}

DIRECT_ACTION_LOSS_WEIGHT=${DIRECT_ACTION_LOSS_WEIGHT:-0.1}
DIRECT_ACTION_HIDDEN_DIM=${DIRECT_ACTION_HIDDEN_DIM:-1024}
DIRECT_FDM_RECON_WEIGHT=${DIRECT_FDM_RECON_WEIGHT:-0.1}
DIRECT_FDM_RANK_WEIGHT=${DIRECT_FDM_RANK_WEIGHT:-0.1}
DIRECT_FDM_RANK_MARGIN=${DIRECT_FDM_RANK_MARGIN:-0.0}
DIRECT_FDM_RANK_TAU=${DIRECT_FDM_RANK_TAU:-0.1}
DIRECT_ACTION_LR=${DIRECT_ACTION_LR:-1e-4}

exec bash "${SCRIPT_DIR}/run_libero_qwenoft_mip_dino_fdm.sh" \
  --datasets.vla_data.include_state true \
  --framework.direct_action.enabled true \
  --framework.direct_action.loss_weight "${DIRECT_ACTION_LOSS_WEIGHT}" \
  --framework.direct_action.hidden_dim "${DIRECT_ACTION_HIDDEN_DIM}" \
  --framework.direct_action.fdm_recon_weight "${DIRECT_FDM_RECON_WEIGHT}" \
  --framework.direct_action.fdm_rank_weight "${DIRECT_FDM_RANK_WEIGHT}" \
  --framework.direct_action.fdm_rank_margin "${DIRECT_FDM_RANK_MARGIN}" \
  --framework.direct_action.fdm_rank_tau "${DIRECT_FDM_RANK_TAU}" \
  --trainer.learning_rate.direct_action_head "${DIRECT_ACTION_LR}" \
  "$@"
