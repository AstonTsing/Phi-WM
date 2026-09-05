#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export Framework_name=${Framework_name:-QwenOFTLayerwiseMIPDINOFDMDirectRank}
export config_yaml=${config_yaml:-examples/LIBERO/train_files/starvla_qwenoft_layerwise_mip_dino_fdm_direct_rank_libero.yaml}
export state_dim=${state_dim:-7}
export run_id=${run_id:-libero_qwenoft_layerwise_mip_dino_fdm_direct_rank_state7_100k}

ACTION_QUERY_FUSION_LR=${ACTION_QUERY_FUSION_LR:-1e-4}

exec bash "${SCRIPT_DIR}/run_libero_qwenoft_mip_dino_fdm_direct_rank.sh" \
  --trainer.learning_rate.action_query_fusion "${ACTION_QUERY_FUSION_LR}" \
  "$@"
