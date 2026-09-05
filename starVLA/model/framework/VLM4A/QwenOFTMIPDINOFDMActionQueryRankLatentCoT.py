# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0 (the "License");
"""Action-query MIP-FDM with a compact pre-action future-DINO latent CoT."""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDMActionQueryRank import (
    QwenOFTMIPDINOFDMActionQueryRankDefaultConfig,
    Qwen_OFT_MIP_DINO_FDM_ActionQueryRank,
    _cfg_get,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model.FutureDinoLatentCoT import FutureDinoLatentCoT
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


class GatedLatentConditionProjector(nn.Module):
    def __init__(self, dino_dim: int, condition_dim: int, gate_init: float) -> None:
        super().__init__()
        if not 0.0 < gate_init < 1.0:
            raise ValueError("latent_cot.gate_init must be in (0, 1)")
        self.projection = nn.Sequential(nn.LayerNorm(dino_dim), nn.Linear(dino_dim, condition_dim))
        self.gate_logit = nn.Parameter(torch.tensor(math.log(gate_init / (1.0 - gate_init))))

    def forward(self, tokens: torch.Tensor):
        projected = self.projection(tokens)
        gate = torch.sigmoid(self.gate_logit).to(dtype=projected.dtype)
        return gate * projected, gate


@dataclass
class QwenOFTMIPDINOFDMActionQueryRankLatentCoTDefaultConfig(
    QwenOFTMIPDINOFDMActionQueryRankDefaultConfig
):
    name: str = "QwenOFTMIPDINOFDMActionQueryRankLatentCoT"
    latent_cot: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.1,
            "num_latents": 8,
            "hidden_dim": 768,
            "num_heads": 8,
            "dropout": 0.0,
            "gate_init": 0.1,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFTMIPDINOFDMActionQueryRankLatentCoT")
class Qwen_OFT_MIP_DINO_FDM_ActionQueryRank_LatentCoT(
    Qwen_OFT_MIP_DINO_FDM_ActionQueryRank
):
    """Reason over future visual latents, then generate and rank MIP actions."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenOFTMIPDINOFDMActionQueryRankLatentCoTDefaultConfig, config)
        super().__init__(config=config, **kwargs)
        self.config.framework.name = "QwenOFTMIPDINOFDMActionQueryRankLatentCoT"
        self.latent_cot_cfg = self.config.framework.get("latent_cot", {})
        if not bool(_cfg_get(self.latent_cot_cfg, "enabled", True)):
            raise ValueError("QwenOFTMIPDINOFDMActionQueryRankLatentCoT requires latent_cot.enabled=true")

        self.latent_cot_loss_weight = float(_cfg_get(self.latent_cot_cfg, "loss_weight", 0.1))
        self.latent_cot_num_latents = int(_cfg_get(self.latent_cot_cfg, "num_latents", 8))
        dino_dim = int(self.dino_model.config.hidden_size)
        self.latent_cot_reasoner = FutureDinoLatentCoT(
            context_dim=self.qwen_hidden_size,
            dino_dim=dino_dim,
            num_latents=self.latent_cot_num_latents,
            hidden_dim=int(_cfg_get(self.latent_cot_cfg, "hidden_dim", 768)),
            num_heads=int(_cfg_get(self.latent_cot_cfg, "num_heads", 8)),
            dropout=float(_cfg_get(self.latent_cot_cfg, "dropout", 0.0)),
        )
        gate_init = float(_cfg_get(self.latent_cot_cfg, "gate_init", 0.1))
        self.latent_cot_condition_projector = GatedLatentConditionProjector(
            dino_dim, self.qwen_hidden_size, gate_init
        )

    def _pool_future_dino(self, future_tokens: torch.Tensor) -> torch.Tensor:
        if future_tokens.ndim == 4:
            future_tokens = future_tokens.flatten(1, 2)
        if future_tokens.ndim != 3:
            raise ValueError(f"future DINO tokens must be [B,K,D] or [B,H,K,D], got {tuple(future_tokens.shape)}")
        pooled = F.adaptive_avg_pool1d(
            future_tokens.detach().float().transpose(1, 2), self.latent_cot_num_latents
        )
        return pooled.transpose(1, 2)

    def _append_latent_condition(self, condition, mask, latent_tokens):
        projector_dtype = next(self.latent_cot_condition_projector.parameters()).dtype
        projected, gate = self.latent_cot_condition_projector(latent_tokens.to(dtype=projector_dtype))
        projected = projected.to(device=condition.device, dtype=condition.dtype)
        if mask is None:
            mask = torch.ones(condition.shape[:2], device=condition.device, dtype=torch.bool)
        latent_mask = torch.ones(projected.shape[:2], device=projected.device, dtype=torch.bool)
        return torch.cat([condition, projected], dim=1), torch.cat([mask, latent_mask], dim=1), gate

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image") for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("LatentCoT requires sample['future_image']")
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        condition, mask, current_dino, patches, action_queries = self._build_action_query_condition(
            batch_images, instructions
        )
        future_dino, future_patches = self._encode_future_dino_tokens_for_fdm(
            future_images, condition.device, condition.dtype
        )
        if not torch.equal(patches, future_patches):
            raise ValueError("Current and future DINO patch indices must match")

        latent_targets = self._pool_future_dino(future_dino)
        predicted_latents = self.latent_cot_reasoner(
            condition, context_mask=mask, teacher_targets=latent_targets
        )
        latent_cot_loss = 1.0 - F.cosine_similarity(
            predicted_latents.float(), latent_targets.float(), dim=-1
        ).mean()
        action_condition, action_mask, latent_gate = self._append_latent_condition(
            condition, mask, predicted_latents
        )

        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=action_condition.device,
            dtype=action_condition.dtype,
        )[:, -self.action_horizon :]
        state_tensor = (
            torch.tensor(np.array(state), device=action_condition.device, dtype=action_condition.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            action_output = self.action_model(
                action_condition, actions, state_tensor, encoder_attention_mask=action_mask
            )
            query_dtype = next(self.action_query_head.parameters()).dtype
            query_actions = self.action_query_head(action_queries.to(dtype=query_dtype))
            action_query_l1 = F.l1_loss(query_actions.float(), actions.float())

        base_fdm_loss, metrics = self._compute_fdm_loss(
            action_output, current_dino, future_dino, patches
        )
        query_rank_loss, query_metrics = self._compute_action_query_rank_loss(
            action_output, query_actions, current_dino, future_dino, patches
        )
        fdm_loss = base_fdm_loss + self.action_query_rank_weight * query_rank_loss
        total_loss = (
            action_output["loss"]
            + self.action_query_loss_weight * action_query_l1.to(dtype=action_output["loss"].dtype)
            + self.latent_cot_loss_weight * latent_cot_loss.to(dtype=action_output["loss"].dtype)
            + self.fdm_loss_weight * fdm_loss
        )
        metrics.update(query_metrics)
        metrics.update({"fdm_loss_base": base_fdm_loss, "fdm_loss": fdm_loss})
        output = {
            "action_loss": total_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "action_query_l1": action_query_l1.detach(),
            "latent_cot_loss": latent_cot_loss.detach(),
            "latent_cot_cosine": (1.0 - latent_cot_loss).detach(),
            "latent_cot_gate": latent_gate.detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in metrics.items()})
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if not isinstance(examples, list):
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:
            batch_images = resize_images(batch_images, target_size=size)
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        condition, mask, _current, _patches, _queries = self._build_action_query_condition(
            batch_images, instructions
        )
        predicted_latents = self.latent_cot_reasoner(condition, context_mask=mask)
        condition, mask, _gate = self._append_latent_condition(condition, mask, predicted_latents)
        state_tensor = (
            torch.from_numpy(np.array(state)).to(condition.device, dtype=condition.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(
                condition, state_tensor, encoder_attention_mask=mask
            )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
