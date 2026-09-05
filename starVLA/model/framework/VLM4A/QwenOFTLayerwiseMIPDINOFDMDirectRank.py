# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""Layer-wise Qwen conditioning for OFT + MIP + DINO FDM direct ranking."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenMIPDINO import _cfg_get
from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDMDirectRank import (
    QwenOFTMIPDINOFDMDirectRankDefaultConfig,
    Qwen_OFT_MIP_DINO_FDM_DirectRank,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenOFTLayerwiseMIPDINOFDMDirectRankDefaultConfig(
    QwenOFTMIPDINOFDMDirectRankDefaultConfig
):
    name: str = "QwenOFTLayerwiseMIPDINOFDMDirectRank"
    layerwise: dict = field(
        default_factory=lambda: {
            "vlm_layer_indices": [8, 12, 16, 20, 24, 28, 32, 36],
            "direct_query_gate_init": 0.1,
        }
    )


class LayerwiseActionQueryFusion(nn.Module):
    """Fuse OFT action placeholders while retaining the final-layer residual."""

    def __init__(self, hidden_dim: int, num_layers: int, gate_init: float) -> None:
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.layer_logits = nn.Parameter(torch.linspace(-1.0, 1.0, num_layers))
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, queries: List[torch.Tensor]) -> torch.Tensor:
        if len(queries) != len(self.norms):
            raise ValueError(f"Expected {len(self.norms)} action-query layers, got {len(queries)}")
        fusion_dtype = self.layer_logits.dtype
        weights = self.layer_logits.softmax(dim=0)
        mixed = sum(
            weight * norm(query.to(dtype=fusion_dtype))
            for weight, norm, query in zip(weights, self.norms, queries)
        )
        return queries[-1].to(dtype=fusion_dtype) + self.gate.tanh() * mixed


@FRAMEWORK_REGISTRY.register("QwenOFTLayerwiseMIPDINOFDMDirectRank")
class Qwen_OFT_Layerwise_MIP_DINO_FDM_DirectRank(Qwen_OFT_MIP_DINO_FDM_DirectRank):
    """Feed successive Qwen hidden layers to successive MIP cross-attention blocks."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(QwenOFTLayerwiseMIPDINOFDMDirectRankDefaultConfig, config)
        super().__init__(config=config, **kwargs)
        self.config.framework.name = "QwenOFTLayerwiseMIPDINOFDMDirectRank"

        layerwise_cfg = self.config.framework.get("layerwise", {})
        text_cfg = getattr(self.qwen_vl_interface.model.config, "text_config", self.qwen_vl_interface.model.config)
        self.num_vlm_layers = int(text_cfg.num_hidden_layers)
        self.num_cross_attention_blocks = sum(
            not (idx % 2 == 1 and self.action_model.model.config.interleave_self_attention)
            for idx in range(len(self.action_model.model.transformer_blocks))
        )

        configured_indices = list(_cfg_get(layerwise_cfg, "vlm_layer_indices", []))
        if not configured_indices:
            configured_indices = (
                torch.linspace(1, self.num_vlm_layers, self.num_cross_attention_blocks)
                .round()
                .to(dtype=torch.long)
                .tolist()
            )
        self.vlm_layer_indices = [
            self.num_vlm_layers + index + 1 if int(index) < 0 else int(index)
            for index in configured_indices
        ]
        if len(self.vlm_layer_indices) != self.num_cross_attention_blocks:
            raise ValueError(
                "layerwise.vlm_layer_indices must match the number of MIP cross-attention blocks: "
                f"indices={self.vlm_layer_indices}, blocks={self.num_cross_attention_blocks}"
            )
        if any(index < 1 or index > self.num_vlm_layers for index in self.vlm_layer_indices):
            raise ValueError(
                f"Qwen layer indices must be in [1, {self.num_vlm_layers}], got {self.vlm_layer_indices}"
            )
        if self.vlm_layer_indices != sorted(self.vlm_layer_indices):
            raise ValueError(f"Qwen layer indices must be ordered, got {self.vlm_layer_indices}")

        hidden_dim = int(self.qwen_vl_interface.model.config.hidden_size)
        self.action_query_fusion = LayerwiseActionQueryFusion(
            hidden_dim=hidden_dim,
            num_layers=len(self.vlm_layer_indices),
            gate_init=float(_cfg_get(layerwise_cfg, "direct_query_gate_init", 0.1)),
        )

    def _encode_qwen_layerwise_with_action_queries(self, batch_images, instructions):
        instructions = self._append_oft_action_prompt(instructions)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        attention_mask = qwen_inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)

        input_ids = qwen_inputs.get("input_ids", None)
        if input_ids is None:
            raise ValueError("QwenOFTLayerwiseMIPDINOFDMDirectRank requires input_ids")
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
            hidden_states = outputs.hidden_states
            qwen_layer_tokens = [hidden_states[index] for index in self.vlm_layer_indices]

        query_layers = [self._gather_action_queries(tokens, input_ids) for tokens in qwen_layer_tokens]
        action_queries = self.action_query_fusion(query_layers)
        return qwen_layer_tokens, attention_mask, action_queries

    def _build_layerwise_ranked_action_condition(self, batch_images, instructions):
        batch_images = self._select_current_condition_images(batch_images)
        qwen_layer_tokens, qwen_mask, action_queries = self._encode_qwen_layerwise_with_action_queries(
            batch_images,
            instructions,
        )
        condition_ref = qwen_layer_tokens[-1]
        raw_dino_tokens, dino_condition_tokens, patch_indices = self._encode_dino_raw_and_condition(
            batch_images,
            condition_ref.device,
            condition_ref.dtype,
        )
        if dino_condition_tokens is None:
            return qwen_layer_tokens, qwen_mask, raw_dino_tokens, patch_indices, action_queries

        if qwen_mask is None:
            qwen_mask = torch.ones(condition_ref.shape[:2], device=condition_ref.device, dtype=torch.bool)
        dino_mask = torch.ones(
            dino_condition_tokens.shape[:2],
            device=dino_condition_tokens.device,
            dtype=torch.bool,
        )
        condition_mask = torch.cat([qwen_mask, dino_mask], dim=1)
        conditions = [torch.cat([tokens, dino_condition_tokens], dim=1) for tokens in qwen_layer_tokens]
        return conditions, condition_mask, raw_dino_tokens, patch_indices, action_queries

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image", None) for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("QwenOFTLayerwiseMIPDINOFDMDirectRank requires sample['future_image']")

        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        conditions, condition_mask, current_dino_tokens, patch_indices, action_queries = (
            self._build_layerwise_ranked_action_condition(batch_images, instructions)
        )
        condition_ref = conditions[-1]
        target_future_dino_tokens, future_patch_indices = self._encode_future_dino_tokens_for_fdm(
            future_images,
            condition_ref.device,
            condition_ref.dtype,
        )
        if not torch.equal(patch_indices, future_patch_indices):
            raise ValueError("current and future DINO patch indices must match for full-token FDM")

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions),
                device=condition_ref.device,
                dtype=condition_ref.dtype,
            )
            actions_target = actions_tensor[:, -self.action_horizon :, :]
            state_tensor = (
                torch.tensor(np.array(state), device=condition_ref.device, dtype=condition_ref.dtype)
                if state is not None
                else None
            )
            action_output = self.action_model(
                conditions,
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
            "layerwise_query_gate": self.action_query_fusion.gate.tanh().detach(),
        }
        output.update({key: value.detach() for key, value in fdm_metrics.items()})
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs: str) -> np.ndarray:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        conditions, condition_mask, _current_dino_tokens, _patch_indices, _action_queries = (
            self._build_layerwise_ranked_action_condition(batch_images, instructions)
        )
        condition_ref = conditions[-1]
        state_tensor = (
            torch.from_numpy(np.array(state)).to(condition_ref.device, dtype=condition_ref.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                conditions,
                state_tensor,
                encoder_attention_mask=condition_mask,
            )
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
