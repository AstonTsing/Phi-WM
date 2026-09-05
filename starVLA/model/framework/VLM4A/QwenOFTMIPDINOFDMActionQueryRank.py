# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0 (the "License");
"""OFT action-query supervision with final-MIP-versus-query FDM ranking."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDM import (
    QwenOFTMIPDINOFDMDefaultConfig,
    Qwen_OFT_MIP_DINO_FDM,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model.DINOFeatureDynamics import fdm_distance_per_sample
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


@dataclass
class QwenOFTMIPDINOFDMActionQueryRankDefaultConfig(QwenOFTMIPDINOFDMDefaultConfig):
    """OFT MIP-DINO-FDM with supervised action-query trajectories."""

    name: str = "QwenOFTMIPDINOFDMActionQueryRank"
    action_query: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.1,
            "hidden_dim": 1024,
            "rank_weight": 0.1,
            "rank_margin": 0.0,
            "rank_tau": 0.1,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFTMIPDINOFDMActionQueryRank")
class Qwen_OFT_MIP_DINO_FDM_ActionQueryRank(Qwen_OFT_MIP_DINO_FDM):
    """Supervise OFT action queries, then rank final MIP actions above them."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenOFTMIPDINOFDMActionQueryRankDefaultConfig, config)
        super().__init__(config=config, **kwargs)
        self.config.framework.name = "QwenOFTMIPDINOFDMActionQueryRank"

        self.action_query_cfg = self.config.framework.get("action_query", {})
        if not bool(_cfg_get(self.action_query_cfg, "enabled", True)):
            raise ValueError("QwenOFTMIPDINOFDMActionQueryRank requires action_query.enabled=true")
        self.action_query_loss_weight = float(_cfg_get(self.action_query_cfg, "loss_weight", 0.1))
        self.action_query_rank_weight = float(_cfg_get(self.action_query_cfg, "rank_weight", 0.1))
        self.action_query_rank_margin = float(_cfg_get(self.action_query_cfg, "rank_margin", 0.0))
        self.action_query_rank_tau = float(_cfg_get(self.action_query_cfg, "rank_tau", 0.1))
        if self.action_query_rank_tau <= 0:
            raise ValueError("action_query.rank_tau must be positive")

        query_hidden_dim = int(_cfg_get(self.action_query_cfg, "hidden_dim", 1024))
        self.action_query_head = nn.Sequential(
            nn.LayerNorm(self.qwen_hidden_size),
            nn.Linear(self.qwen_hidden_size, query_hidden_dim),
            nn.GELU(),
            nn.Linear(query_hidden_dim, int(self.config.framework.action_model.action_dim)),
        )

    def _gather_action_queries(self, last_hidden: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        action_mask = input_ids == self.action_token_id
        counts = action_mask.sum(dim=1)
        if (counts < self.chunk_len).any():
            raise RuntimeError(f"Expected {self.chunk_len} action tokens per sample, got {counts.tolist()}")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand_as(input_ids)
        positions = torch.where(action_mask, positions, torch.full_like(positions, -1))
        selected = positions.topk(k=self.chunk_len, dim=-1).values.sort(dim=-1).values
        gather_index = selected.unsqueeze(-1).expand(-1, -1, last_hidden.shape[-1])
        return last_hidden.gather(dim=1, index=gather_index)

    def _encode_qwen_with_action_queries(self, batch_images, instructions):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=self._append_oft_action_prompt(instructions),
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("QwenOFTMIPDINOFDMActionQueryRank requires input_ids")
        action_counts = (input_ids == self.action_token_id).sum(dim=1)
        if (action_counts < self.chunk_len).any():
            raise RuntimeError(
                "Insufficient OFT action tokens after tokenization: "
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
        return last_hidden, attention_mask, self._gather_action_queries(last_hidden, input_ids)

    def _build_action_query_condition(self, batch_images, instructions):
        batch_images = self._select_current_condition_images(batch_images)
        qwen_tokens, qwen_mask, action_queries = self._encode_qwen_with_action_queries(batch_images, instructions)
        raw_dino_tokens, dino_condition_tokens, patch_indices = self._encode_dino_raw_and_condition(
            batch_images,
            qwen_tokens.device,
            qwen_tokens.dtype,
        )
        if dino_condition_tokens is None:
            return qwen_tokens, qwen_mask, raw_dino_tokens, patch_indices, action_queries
        if qwen_mask is None:
            qwen_mask = torch.ones(qwen_tokens.shape[:2], device=qwen_tokens.device, dtype=torch.bool)
        dino_mask = torch.ones(
            dino_condition_tokens.shape[:2],
            device=dino_condition_tokens.device,
            dtype=torch.bool,
        )
        return (
            torch.cat([qwen_tokens, dino_condition_tokens], dim=1),
            torch.cat([qwen_mask, dino_mask], dim=1),
            raw_dino_tokens,
            patch_indices,
            action_queries,
        )

    def _compute_action_query_rank_loss(
        self,
        action_output: Dict[str, torch.Tensor],
        action_query_actions: torch.Tensor,
        current_dino_tokens: torch.Tensor,
        target_future_dino_tokens: torch.Tensor,
        patch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Train FDM to rank final MIP actions above detached action-query actions.

        The query action and its FDM distance are detached, so the VLM/direct
        head receives only action regression supervision and the query branch
        is a fixed counterfactual anchor for final-stage refinement.
        """
        zero = action_output["loss"].new_zeros(())
        if not self.fdm_enabled or self.action_query_rank_weight <= 0:
            return zero, {"fdm_rank_stage1_query": zero, "fdm_order_stage1_query": zero}
        if "pred_action_stage1" not in action_output:
            raise KeyError("QwenOFTMIPDINOFDMActionQueryRank requires pred_action_stage1")

        query_actions = action_query_actions.detach().float()
        stage1_actions = action_output["pred_action_stage1"].float()
        if self.fdm_detach_action:
            stage1_actions = stage1_actions.detach()
        current_tokens = current_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)
        target_tokens = target_future_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)

        if target_tokens.ndim == 3:
            if current_tokens.shape != target_tokens.shape:
                raise ValueError(
                    "current/future DINO token shape mismatch: "
                    f"current={tuple(current_tokens.shape)} future={tuple(target_tokens.shape)}"
                )
            num_horizons = 1
        elif target_tokens.ndim == 4:
            expected = (target_tokens.shape[0], target_tokens.shape[2], target_tokens.shape[3])
            if current_tokens.shape != expected:
                raise ValueError(
                    "current/future DINO token shape mismatch: "
                    f"current={tuple(current_tokens.shape)} future={tuple(target_tokens.shape)}"
                )
            num_horizons = target_tokens.shape[1]
        else:
            raise ValueError(f"Unexpected future DINO shape {tuple(target_tokens.shape)}")

        if len(self.fdm_horizon_weights) < num_horizons:
            raise ValueError(f"Need {num_horizons} horizon weights, got {self.fdm_horizon_weights}")
        weights = torch.tensor(
            self.fdm_horizon_weights[:num_horizons],
            device=stage1_actions.device,
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-8)

        rank_total = zero.float()
        order_total = zero.float()
        query_distance_total = zero.float()
        stage1_distance_total = zero.float()
        metrics = {}
        for horizon in range(num_horizons):
            horizon_index = None if target_tokens.ndim == 3 else horizon
            target = target_tokens if target_tokens.ndim == 3 else target_tokens[:, horizon]
            pred_query = self.fdm_predictor(
                current_tokens,
                query_actions,
                patch_indices=patch_indices,
                horizon_indices=horizon_index,
            )
            pred_stage1 = self.fdm_predictor(
                current_tokens,
                stage1_actions,
                patch_indices=patch_indices,
                horizon_indices=horizon_index,
            )
            query_distance = fdm_distance_per_sample(pred_query, target.to(dtype=pred_query.dtype))
            stage1_distance = fdm_distance_per_sample(pred_stage1, target.to(dtype=pred_stage1.dtype))
            rank_h = F.softplus(
                (stage1_distance - query_distance.detach() + self.action_query_rank_margin)
                / self.action_query_rank_tau
            ).mean()
            order_h = (stage1_distance < query_distance).float().mean()
            weight = weights[horizon]
            rank_total = rank_total + weight * rank_h.float()
            order_total = order_total + weight * order_h.float()
            query_distance_total = query_distance_total + weight * query_distance.mean().float()
            stage1_distance_total = stage1_distance_total + weight * stage1_distance.mean().float()
            if num_horizons > 1:
                metrics[f"fdm_rank_stage1_query_h{horizon}"] = rank_h
                metrics[f"fdm_order_stage1_query_h{horizon}"] = order_h

        metrics.update(
            {
                "fdm_rank_stage1_query": rank_total,
                "fdm_order_stage1_query": order_total,
                "fdm_query_distance": query_distance_total,
                "fdm_stage1_distance_for_query_rank": stage1_distance_total,
            }
        )
        return rank_total.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype), metrics

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image") for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("QwenOFTMIPDINOFDMActionQueryRank requires sample['future_image']")
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        condition_tokens, condition_mask, current_dino_tokens, patch_indices, action_queries = (
            self._build_action_query_condition(batch_images, instructions)
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
            head_dtype = next(self.action_query_head.parameters()).dtype
            action_query_actions = self.action_query_head(action_queries.to(dtype=head_dtype))
            action_query_l1 = F.l1_loss(action_query_actions.float(), actions_target.float())

        base_fdm_loss, fdm_metrics = self._compute_fdm_loss(
            action_output,
            current_dino_tokens,
            target_future_dino_tokens,
            patch_indices,
        )
        query_rank_loss, query_rank_metrics = self._compute_action_query_rank_loss(
            action_output,
            action_query_actions,
            current_dino_tokens,
            target_future_dino_tokens,
            patch_indices,
        )
        fdm_loss = base_fdm_loss + self.action_query_rank_weight * query_rank_loss
        fdm_metrics["fdm_loss_base"] = base_fdm_loss
        fdm_metrics["fdm_loss"] = fdm_loss
        fdm_metrics.update(query_rank_metrics)

        total_loss = (
            action_output["loss"]
            + self.action_query_loss_weight * action_query_l1.to(dtype=action_output["loss"].dtype)
            + self.fdm_loss_weight * fdm_loss
        )
        output = {
            "action_loss": total_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "action_query_l1": action_query_l1.detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in fdm_metrics.items()})
        return output
