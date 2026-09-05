# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""QwenOFTMIPDINOFDM with direct action regression and three-level FDM ranking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenMIPDINO import _cfg_get
from starVLA.model.framework.VLM4A.QwenMIPDINOFDM import _cfg_get_bool
from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDM import (
    QwenOFTMIPDINOFDMDefaultConfig,
    Qwen_OFT_MIP_DINO_FDM,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model.DINOFeatureDynamics import fdm_distance_per_sample
from starVLA.model.tools import FRAMEWORK_REGISTRY


@dataclass
class QwenOFTMIPDINOFDMDirectRankDefaultConfig(QwenOFTMIPDINOFDMDefaultConfig):
    name: str = "QwenOFTMIPDINOFDMDirectRank"
    direct_action: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.1,
            "hidden_dim": 1024,
            "fdm_recon_weight": 0.1,
            "fdm_rank_weight": 0.1,
            "fdm_rank_margin": 0.0,
            "fdm_rank_tau": 0.1,
        }
    )


def three_level_fdm_loss(
    pred_direct: torch.Tensor,
    pred_stage0: torch.Tensor,
    pred_stage1: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    stage0_weight: float,
    direct_recon_weight: float,
    stage1_rank_weight: float,
    stage0_rank_weight: float,
    stage1_rank_margin: float,
    stage0_rank_margin: float,
    stage1_rank_tau: float,
    stage0_rank_tau: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Enforce stage1 < stage0 < direct in future-feature distance."""
    if stage1_rank_tau <= 0 or stage0_rank_tau <= 0:
        raise ValueError("FDM rank temperatures must be positive")

    direct_dist = fdm_distance_per_sample(pred_direct, target_tokens)
    stage0_dist = fdm_distance_per_sample(pred_stage0, target_tokens)
    stage1_dist = fdm_distance_per_sample(pred_stage1, target_tokens)

    direct_recon = direct_dist.mean()
    stage0_recon = stage0_dist.mean()
    stage1_recon = stage1_dist.mean()
    rank_stage1_stage0 = F.softplus(
        (stage1_dist - stage0_dist.detach() + float(stage1_rank_margin)) / float(stage1_rank_tau)
    ).mean()
    rank_stage0_direct = F.softplus(
        (stage0_dist - direct_dist.detach() + float(stage0_rank_margin)) / float(stage0_rank_tau)
    ).mean()

    total = (
        stage1_recon
        + float(stage0_weight) * stage0_recon
        + float(direct_recon_weight) * direct_recon
        + float(stage1_rank_weight) * rank_stage1_stage0
        + float(stage0_rank_weight) * rank_stage0_direct
    )
    return total, {
        "fdm_loss": total,
        "fdm_loss_direct": direct_recon,
        "fdm_loss_stage0": stage0_recon,
        "fdm_loss_stage1": stage1_recon,
        "fdm_rank": rank_stage1_stage0,
        "fdm_rank_stage1_stage0": rank_stage1_stage0,
        "fdm_rank_stage0_direct": rank_stage0_direct,
        "fdm_order_stage1_stage0": (stage1_dist < stage0_dist).float().mean(),
        "fdm_order_stage0_direct": (stage0_dist < direct_dist).float().mean(),
    }


@FRAMEWORK_REGISTRY.register("QwenOFTMIPDINOFDMDirectRank")
class Qwen_OFT_MIP_DINO_FDM_DirectRank(Qwen_OFT_MIP_DINO_FDM):
    """Auxiliary Qwen action regression plus ranked MIP/DINO dynamics."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenOFTMIPDINOFDMDirectRankDefaultConfig, config)
        super().__init__(config=config, **kwargs)
        self.config.framework.name = "QwenOFTMIPDINOFDMDirectRank"

        self.direct_action_cfg = self.config.framework.get("direct_action", {})
        self.direct_action_enabled = _cfg_get_bool(self.direct_action_cfg, "enabled", True)
        if not self.direct_action_enabled:
            raise ValueError("QwenOFTMIPDINOFDMDirectRank requires direct_action.enabled=true")
        self.direct_action_loss_weight = float(_cfg_get(self.direct_action_cfg, "loss_weight", 0.1))
        self.direct_fdm_recon_weight = float(_cfg_get(self.direct_action_cfg, "fdm_recon_weight", 0.1))
        self.direct_fdm_rank_weight = float(_cfg_get(self.direct_action_cfg, "fdm_rank_weight", 0.1))
        self.direct_fdm_rank_margin = float(_cfg_get(self.direct_action_cfg, "fdm_rank_margin", 0.0))
        self.direct_fdm_rank_tau = float(_cfg_get(self.direct_action_cfg, "fdm_rank_tau", 0.1))

        qwen_hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        direct_hidden_dim = int(_cfg_get(self.direct_action_cfg, "hidden_dim", 1024))
        action_dim = int(self.config.framework.action_model.action_dim)
        self.direct_action_head = nn.Sequential(
            nn.LayerNorm(qwen_hidden_dim),
            nn.Linear(qwen_hidden_dim, direct_hidden_dim),
            nn.GELU(),
            nn.Linear(direct_hidden_dim, action_dim),
        )

    def _encode_qwen_with_action_queries(self, batch_images, instructions):
        instructions = self._append_oft_action_prompt(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("QwenOFTMIPDINOFDMDirectRank requires input_ids")
        action_counts = (input_ids == self.action_token_id).sum(dim=1)
        if (action_counts < self.chunk_len).any():
            raise RuntimeError(
                "Insufficient OFT action placeholder tokens after tokenization: "
                f"counts={action_counts.tolist()}, required={self.chunk_len}"
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]

        action_queries = self._gather_action_queries(last_hidden, input_ids)
        return last_hidden, backbone_attention_mask, action_queries

    def _encode_qwen(self, batch_images, instructions):
        last_hidden, attention_mask, _action_queries = self._encode_qwen_with_action_queries(
            batch_images,
            instructions,
        )
        return last_hidden, attention_mask

    def _gather_action_queries(self, last_hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        mask = input_ids == self.action_token_id
        counts = mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            raise RuntimeError(
                f"Expected at least {self.chunk_len} action tokens per sample, got {counts.tolist()}"
            )
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        positions = torch.where(mask, positions, torch.full_like(positions, -1))
        selected = positions.topk(k=self.chunk_len, dim=-1).values.sort(dim=-1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        return last_hidden.gather(dim=1, index=gather_index)

    def _build_ranked_action_condition(self, batch_images, instructions):
        batch_images = self._select_current_condition_images(batch_images)
        qwen_tokens, qwen_mask, action_queries = self._encode_qwen_with_action_queries(
            batch_images,
            instructions,
        )
        raw_dino_tokens, dino_condition_tokens, patch_indices = self._encode_dino_raw_and_condition(
            batch_images,
            qwen_tokens.device,
            qwen_tokens.dtype,
        )
        if dino_condition_tokens is None:
            return qwen_tokens, qwen_mask, raw_dino_tokens, patch_indices, action_queries

        condition_tokens = torch.cat([qwen_tokens, dino_condition_tokens], dim=1)
        if qwen_mask is None:
            qwen_mask = torch.ones(qwen_tokens.shape[:2], device=qwen_tokens.device, dtype=torch.bool)
        dino_mask = torch.ones(
            dino_condition_tokens.shape[:2],
            device=dino_condition_tokens.device,
            dtype=torch.bool,
        )
        return (
            condition_tokens,
            torch.cat([qwen_mask, dino_mask], dim=1),
            raw_dino_tokens,
            patch_indices,
            action_queries,
        )

    def _compute_ranked_fdm_loss(
        self,
        action_output,
        direct_actions: torch.Tensor,
        current_dino_tokens: torch.Tensor,
        target_future_dino_tokens: torch.Tensor,
        patch_indices: torch.Tensor,
    ):
        if not self.fdm_enabled or self.fdm_loss_weight <= 0:
            zero = action_output["loss"].new_zeros(())
            return zero, {
                "fdm_loss": zero,
                "fdm_loss_direct": zero,
                "fdm_loss_stage0": zero,
                "fdm_loss_stage1": zero,
                "fdm_rank": zero,
                "fdm_rank_stage1_stage0": zero,
                "fdm_rank_stage0_direct": zero,
                "fdm_order_stage1_stage0": zero,
                "fdm_order_stage0_direct": zero,
            }

        required = ("pred_action_stage0", "pred_action_stage1")
        missing = [key for key in required if key not in action_output]
        if missing:
            raise KeyError(f"QwenOFTMIPDINOFDMDirectRank requires MIP predictions, missing {missing}")

        direct_actions = direct_actions.detach().float()
        stage0_actions = action_output["pred_action_stage0"].float()
        stage1_actions = action_output["pred_action_stage1"].float()
        if self.fdm_detach_action:
            stage0_actions = stage0_actions.detach()
            stage1_actions = stage1_actions.detach()

        current_tokens = current_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)
        target_tokens = target_future_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)
        if target_tokens.ndim not in (3, 4):
            raise ValueError(
                f"target future DINO tokens must be [B,K,D] or [B,H,K,D], got {tuple(target_tokens.shape)}"
            )
        expected_current_shape = (
            target_tokens.shape
            if target_tokens.ndim == 3
            else (target_tokens.shape[0], target_tokens.shape[2], target_tokens.shape[3])
        )
        if tuple(current_tokens.shape) != tuple(expected_current_shape):
            raise ValueError(
                "current/future DINO token shape mismatch: "
                f"current={tuple(current_tokens.shape)} future={tuple(target_tokens.shape)}"
            )

        num_horizons = 1 if target_tokens.ndim == 3 else target_tokens.shape[1]
        if len(self.fdm_horizon_weights) < num_horizons:
            raise ValueError(f"Need {num_horizons} horizon weights, got {self.fdm_horizon_weights}")
        weights = torch.tensor(
            self.fdm_horizon_weights[:num_horizons],
            device=stage1_actions.device,
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-8)

        totals = {}
        metrics = {}
        for horizon in range(num_horizons):
            horizon_index = None if target_tokens.ndim == 3 else horizon
            target = target_tokens if target_tokens.ndim == 3 else target_tokens[:, horizon]
            pred_direct = self.fdm_predictor(
                current_tokens,
                direct_actions,
                patch_indices=patch_indices,
                horizon_indices=horizon_index,
            )
            pred_stage0 = self.fdm_predictor(
                current_tokens,
                stage0_actions,
                patch_indices=patch_indices,
                horizon_indices=horizon_index,
            )
            pred_stage1 = self.fdm_predictor(
                current_tokens,
                stage1_actions,
                patch_indices=patch_indices,
                horizon_indices=horizon_index,
            )
            loss_h, metrics_h = three_level_fdm_loss(
                pred_direct,
                pred_stage0,
                pred_stage1,
                target.to(dtype=pred_stage1.dtype),
                stage0_weight=self.fdm_stage0_weight,
                direct_recon_weight=self.direct_fdm_recon_weight,
                stage1_rank_weight=self.fdm_rank_weight,
                stage0_rank_weight=self.direct_fdm_rank_weight,
                stage1_rank_margin=self.fdm_rank_margin,
                stage0_rank_margin=self.direct_fdm_rank_margin,
                stage1_rank_tau=self.fdm_rank_tau,
                stage0_rank_tau=self.direct_fdm_rank_tau,
            )
            weight = weights[horizon]
            for key, value in metrics_h.items():
                totals[key] = totals.get(key, value.new_zeros(())) + weight * value.float()
            if num_horizons > 1:
                metrics[f"fdm_loss_h{horizon}"] = loss_h
                metrics[f"fdm_rank_stage1_stage0_h{horizon}"] = metrics_h["fdm_rank_stage1_stage0"]
                metrics[f"fdm_rank_stage0_direct_h{horizon}"] = metrics_h["fdm_rank_stage0_direct"]

        metrics.update(totals)
        return totals["fdm_loss"].to(
            device=action_output["loss"].device,
            dtype=action_output["loss"].dtype,
        ), metrics

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image", None) for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("QwenOFTMIPDINOFDMDirectRank requires sample['future_image']")

        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        condition_tokens, condition_mask, current_dino_tokens, patch_indices, action_queries = (
            self._build_ranked_action_condition(batch_images, instructions)
        )
        target_future_dino_tokens, future_patch_indices = self._encode_future_dino_tokens_for_fdm(
            future_images,
            condition_tokens.device,
            condition_tokens.dtype,
        )
        if not torch.equal(patch_indices, future_patch_indices):
            raise ValueError("current and future DINO patch indices must match for full-token FDM")

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions),
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
            actions_target = actions_tensor[:, -self.action_horizon :, :]
            state_tensor = (
                torch.tensor(np.array(state), device=condition_tokens.device, dtype=condition_tokens.dtype)
                if state is not None
                else None
            )
            action_output = self.action_model(
                condition_tokens,
                actions_target,
                state_tensor,
                encoder_attention_mask=condition_mask,
            )
            head_dtype = next(self.direct_action_head.parameters()).dtype
            direct_actions = self.direct_action_head(action_queries.to(dtype=head_dtype))
            direct_action_loss = F.l1_loss(direct_actions.float(), actions_target.float())

        fdm_loss, fdm_metrics = self._compute_ranked_fdm_loss(
            action_output,
            direct_actions,
            current_dino_tokens,
            target_future_dino_tokens,
            patch_indices,
        )
        total_loss = (
            action_output["loss"]
            + self.direct_action_loss_weight * direct_action_loss.to(dtype=action_output["loss"].dtype)
            + self.fdm_loss_weight * fdm_loss
        )
        output = {
            "action_loss": total_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "direct_action_l1": direct_action_loss.detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in fdm_metrics.items()})
        return output
