#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export Framework_name=${Framework_name:-QwenOFTMIPDINOFDMActionQueryRankLatentCoT}
export config_yaml=${config_yaml:-examples/Robocasa_tabletop/train_files/starvla_qwenoft_mip_dino_fdm_action_query_rank_latent_cot_robocasa_gr1.yaml}
export state_dim=${state_dim:-58}
export run_id=${run_id:-robocasa_qwenoft_mip_dino_fdm_action_query_rank_latent_cot_state58_200k}
export FDM_STAGE0_WEIGHT=${FDM_STAGE0_WEIGHT:-0.0}

LATENT_COT_LOSS_WEIGHT=${LATENT_COT_LOSS_WEIGHT:-0.1}
LATENT_COT_NUM_LATENTS=${LATENT_COT_NUM_LATENTS:-8}
LATENT_COT_HIDDEN_DIM=${LATENT_COT_HIDDEN_DIM:-768}
LATENT_COT_NUM_HEADS=${LATENT_COT_NUM_HEADS:-8}
LATENT_COT_DROPOUT=${LATENT_COT_DROPOUT:-0.0}
LATENT_COT_GATE_INIT=${LATENT_COT_GATE_INIT:-0.1}
LATENT_COT_LR=${LATENT_COT_LR:-1e-4}

exec bash "${SCRIPT_DIR}/run_robocasa_qwenoft_mip_dino_fdm_action_query_rank.sh" \
  --framework.latent_cot.enabled true \
  --framework.latent_cot.loss_weight "${LATENT_COT_LOSS_WEIGHT}" \
  --framework.latent_cot.num_latents "${LATENT_COT_NUM_LATENTS}" \
  --framework.latent_cot.hidden_dim "${LATENT_COT_HIDDEN_DIM}" \
  --framework.latent_cot.num_heads "${LATENT_COT_NUM_HEADS}" \
  --framework.latent_cot.dropout "${LATENT_COT_DROPOUT}" \
  --framework.latent_cot.gate_init "${LATENT_COT_GATE_INIT}" \
  --trainer.learning_rate.latent_cot_reasoner "${LATENT_COT_LR}" \
  --trainer.learning_rate.latent_cot_condition_projector "${LATENT_COT_LR}" \
  "$@"
