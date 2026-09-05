#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export Framework_name=${Framework_name:-QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoT}
export config_yaml=${config_yaml:-examples/Robocasa_tabletop/train_files/starvla_qwenoft_mip_dino_fdm_action_query_rank_vlm_latent_cot_robocasa_gr1.yaml}
export state_dim=${state_dim:-58}
export run_id=${run_id:-robocasa_qwenoft_mip_dino_fdm_action_query_rank_vlm_latent_cot16_state58_200k}
export FDM_STAGE0_WEIGHT=${FDM_STAGE0_WEIGHT:-0.0}
export PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-8}

LATENT_COT_LOSS_WEIGHT=${LATENT_COT_LOSS_WEIGHT:-0.1}
LATENT_COT_NUM_LATENTS=${LATENT_COT_NUM_LATENTS:-16}
LATENT_COT_FUTURE_IMAGE_INDICES=${LATENT_COT_FUTURE_IMAGE_INDICES:-1,3}
LATENT_COT_ACTION_GRADIENT=${LATENT_COT_ACTION_GRADIENT:-true}

exec bash "${SCRIPT_DIR}/run_robocasa_qwenoft_mip_dino_fdm_action_query_rank.sh" \
  --framework.latent_cot.enabled true \
  --framework.latent_cot.loss_weight "${LATENT_COT_LOSS_WEIGHT}" \
  --framework.latent_cot.num_latents "${LATENT_COT_NUM_LATENTS}" \
  --framework.latent_cot.future_image_indices "[${LATENT_COT_FUTURE_IMAGE_INDICES}]" \
  --framework.latent_cot.action_gradient_to_latent "${LATENT_COT_ACTION_GRADIENT}" \
  --framework.latent_cot.inference_use_cache true \
  "$@"
