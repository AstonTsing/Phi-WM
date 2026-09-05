# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""QwenMIPDINO with a train-only DINO feature dynamics loss."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenMIPDINO import (
    QwenMIPDINODefaultConfig,
    Qwen_MIP_DINO,
    _cfg_get,
)
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.world_model.DINOFeatureDynamics import (
    DINOFeatureDynamicsPredictor,
    mip_fdm_loss,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _cfg_get_float_list(cfg, key, default):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        value = [part.strip() for part in value.strip().strip("[]").split(",") if part.strip()]
    return [float(item) for item in value]


def _cfg_get_bool(cfg, key, default):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


@dataclass
class QwenMIPDINOFDMDefaultConfig(QwenMIPDINODefaultConfig):
    """QwenMIPDINO + train-only future DINO dynamics loss."""

    name: str = "QwenMIPDINOFDM"

    fdm: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.05,
            "stage0_weight": 0.25,
            "rank_weight": 0.1,
            "rank_margin": 0.0,
            "rank_tau": 0.1,
            "detach_dino": True,
            "detach_action": False,
            "hidden_dim": 768,
            "depth": 4,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "dropout": 0.0,
            "max_patches": 2048,
            "condition_num_images": 2,
            "multi_horizon": False,
            "max_horizons": 1,
            "horizon_weights": [1.0],
        }
    )


@FRAMEWORK_REGISTRY.register("QwenMIPDINOFDM")
class Qwen_MIP_DINO_FDM(Qwen_MIP_DINO):
    """QwenVL + DINOv3 dense condition + corrected MIP action + DINO FDM loss."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(config=merge_framework_config(QwenMIPDINOFDMDefaultConfig, config), **kwargs)
        self.config.framework.name = "QwenMIPDINOFDM"
        self.fdm_cfg = self.config.framework.get("fdm", {})
        self.fdm_enabled = _cfg_get_bool(self.fdm_cfg, "enabled", True)
        self.fdm_loss_weight = float(_cfg_get(self.fdm_cfg, "loss_weight", 0.05))
        self.fdm_stage0_weight = float(_cfg_get(self.fdm_cfg, "stage0_weight", 0.25))
        self.fdm_rank_weight = float(_cfg_get(self.fdm_cfg, "rank_weight", 0.1))
        self.fdm_rank_margin = float(_cfg_get(self.fdm_cfg, "rank_margin", 0.0))
        self.fdm_rank_tau = float(_cfg_get(self.fdm_cfg, "rank_tau", 0.1))
        self.fdm_detach_dino = _cfg_get_bool(self.fdm_cfg, "detach_dino", True)
        self.fdm_detach_action = _cfg_get_bool(self.fdm_cfg, "detach_action", False)
        self.fdm_condition_num_images = int(_cfg_get(self.fdm_cfg, "condition_num_images", 2))
        self.fdm_multi_horizon = _cfg_get_bool(self.fdm_cfg, "multi_horizon", False)
        self.fdm_max_horizons = int(_cfg_get(self.fdm_cfg, "max_horizons", 1))
        self.fdm_horizon_weights = _cfg_get_float_list(self.fdm_cfg, "horizon_weights", [1.0])

        if self.fdm_enabled:
            if self.dino_model is None:
                raise ValueError("QwenMIPDINOFDM requires DINOv3 to be enabled")
            self.fdm_predictor = DINOFeatureDynamicsPredictor(
                dino_dim=int(self.dino_model.config.hidden_size),
                action_dim=int(self.config.framework.action_model.action_dim),
                action_horizon=int(self.config.framework.action_model.action_horizon),
                hidden_dim=int(_cfg_get(self.fdm_cfg, "hidden_dim", 768)),
                depth=int(_cfg_get(self.fdm_cfg, "depth", 4)),
                num_heads=int(_cfg_get(self.fdm_cfg, "num_heads", 8)),
                mlp_ratio=float(_cfg_get(self.fdm_cfg, "mlp_ratio", 4.0)),
                dropout=float(_cfg_get(self.fdm_cfg, "dropout", 0.0)),
                max_patches=int(_cfg_get(self.fdm_cfg, "max_patches", 2048)),
                max_horizons=self.fdm_max_horizons if self.fdm_multi_horizon else 1,
            )
        else:
            self.fdm_predictor = None

    def _select_last_images_per_view(self, batch_images: List[List], *, context: str) -> List[List]:
        """Keep the last timestep for each view from a view-major image sequence."""
        num_views = self.fdm_condition_num_images
        if num_views <= 0:
            return batch_images

        selected_images = []
        for images in batch_images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            images = list(images)
            if len(images) < num_views:
                raise ValueError(
                    f"QwenMIPDINOFDM expected at least {num_views} {context} image(s), got {len(images)}"
                )
            if len(images) == num_views:
                selected_images.append(images)
                continue
            if len(images) % num_views != 0:
                raise ValueError(
                    "QwenMIPDINOFDM cannot infer view-major temporal packing: "
                    f"context={context}, num_images={len(images)}, num_views={num_views}"
                )

            frames_per_view = len(images) // num_views
            selected_images.append([images[(view_idx + 1) * frames_per_view - 1] for view_idx in range(num_views)])
        return selected_images

    def _select_current_condition_images(self, batch_images: List[List]) -> List[List]:
        """Keep current-timestep views for Qwen/DINO condition.

        With pack_multiview=None, the dataloader flattens history in view-major
        order, e.g. [primary -3,-2,-1,0, wrist -3,-2,-1,0]. For Qwen we only
        want the current view from each camera: [primary 0, wrist 0].
        """
        return self._select_last_images_per_view(batch_images, context="condition")

    def _select_future_target_images(self, future_images: List[List]) -> List[List]:
        """Keep the last future timestep per view for the single-horizon FDM objective."""
        return self._select_last_images_per_view(future_images, context="future")

    def _select_future_target_image_horizons(self, future_images: List[List]) -> List[List[List]]:
        num_views = self.fdm_condition_num_images
        batch_horizons = []
        counts = []
        for images in future_images:
            images = [images] if not isinstance(images, (list, tuple)) else list(images)
            if num_views <= 0 or len(images) % num_views != 0:
                raise ValueError(
                    "QwenMIPDINOFDM cannot infer future horizon packing: "
                    f"num_images={len(images)}, num_views={num_views}"
                )
            frames_per_view = len(images) // num_views
            horizon_count = min(frames_per_view, self.fdm_max_horizons)
            counts.append(horizon_count)
            batch_horizons.append(
                [[images[view_idx * frames_per_view + h] for view_idx in range(num_views)] for h in range(horizon_count)]
            )
        if len(set(counts)) != 1:
            raise ValueError(f"Future horizon counts must match across batch, got {counts}")
        return batch_horizons

    def _encode_dino_raw_and_condition(
        self,
        batch_images: List[List],
        device: torch.device,
        condition_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not self.dino_enabled:
            return None, None, None
        pixel_values, view_counts = self._preprocess_dino_flat_views(batch_images)
        pixel_values = pixel_values.to(device=device)
        num_views = view_counts[0]

        self.dino_model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=pixel_values.is_cuda):
                output = self.dino_model(pixel_values=pixel_values)

        dino_tokens = self._select_dino_tokens(output.last_hidden_state)
        if self.dino_normalize_tokens:
            dino_tokens = torch.nn.functional.normalize(dino_tokens.float(), dim=-1).to(dtype=dino_tokens.dtype)

        batch_size = len(batch_images)
        dino_tokens = dino_tokens.reshape(batch_size, num_views * dino_tokens.shape[1], dino_tokens.shape[-1])
        raw_tokens = dino_tokens.float()
        if self.fdm_detach_dino:
            raw_tokens = raw_tokens.detach()

        patch_indices = torch.arange(raw_tokens.shape[1], device=device, dtype=torch.long)
        patch_indices = patch_indices.unsqueeze(0).expand(batch_size, -1)

        projector_dtype = next(self.dino_projector.parameters()).dtype
        condition_tokens = self.dino_projector(dino_tokens.to(device=device, dtype=projector_dtype))
        condition_tokens = condition_tokens.to(dtype=condition_dtype)
        return raw_tokens, condition_tokens, patch_indices

    def _build_fdm_action_condition(self, batch_images, instructions):
        batch_images = self._select_current_condition_images(batch_images)
        qwen_tokens, qwen_mask = self._encode_qwen(batch_images, instructions)
        raw_dino_tokens, dino_condition_tokens, patch_indices = self._encode_dino_raw_and_condition(
            batch_images,
            qwen_tokens.device,
            qwen_tokens.dtype,
        )
        if dino_condition_tokens is None:
            return qwen_tokens, qwen_mask, raw_dino_tokens, patch_indices

        condition_tokens = torch.cat([qwen_tokens, dino_condition_tokens], dim=1)
        if qwen_mask is None:
            qwen_mask = torch.ones(
                qwen_tokens.shape[:2],
                device=qwen_tokens.device,
                dtype=torch.bool,
            )
        dino_mask = torch.ones(
            dino_condition_tokens.shape[:2],
            device=dino_condition_tokens.device,
            dtype=torch.bool,
        )
        condition_mask = torch.cat([qwen_mask, dino_mask], dim=1)
        return condition_tokens, condition_mask, raw_dino_tokens, patch_indices

    def _encode_future_dino_tokens_for_fdm(
        self,
        future_images: List[List],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.fdm_multi_horizon:
            future_images = self._select_future_target_images(future_images)
            raw_tokens, _condition_tokens, patch_indices = self._encode_dino_raw_and_condition(
                future_images,
                device,
                dtype,
            )
            if raw_tokens is None:
                raise ValueError("QwenMIPDINOFDM requires DINOv3 future tokens")
            return raw_tokens, patch_indices

        future_horizons = self._select_future_target_image_horizons(future_images)
        batch_size = len(future_horizons)
        num_horizons = len(future_horizons[0])
        flat_images = [views for sample in future_horizons for views in sample]
        raw_tokens, _condition_tokens, patch_indices = self._encode_dino_raw_and_condition(flat_images, device, dtype)
        if raw_tokens is None:
            raise ValueError("QwenMIPDINOFDM requires DINOv3 future tokens")
        raw_tokens = raw_tokens.reshape(batch_size, num_horizons, raw_tokens.shape[-2], raw_tokens.shape[-1])
        patch_indices = patch_indices.reshape(batch_size, num_horizons, patch_indices.shape[-1])
        if not torch.equal(patch_indices, patch_indices[:, :1].expand_as(patch_indices)):
            raise ValueError("Future DINO patch indices must match across horizons")
        return raw_tokens, patch_indices[:, 0]

    def _compute_fdm_loss(
        self,
        action_output,
        current_dino_tokens: torch.Tensor,
        target_future_dino_tokens: torch.Tensor,
        patch_indices: torch.Tensor,
    ):
        if not self.fdm_enabled or self.fdm_loss_weight <= 0:
            zero = action_output["loss"].new_zeros(())
            return zero, {
                "fdm_loss": zero,
                "fdm_loss_stage0": zero,
                "fdm_loss_stage1": zero,
                "fdm_rank": zero,
            }
        required = ("pred_action_stage0", "pred_action_stage1")
        missing = [key for key in required if key not in action_output]
        if missing:
            raise KeyError(f"QwenMIPDINOFDM requires clean MIP action predictions, missing {missing}")

        stage0_actions = action_output["pred_action_stage0"].float()
        stage1_actions = action_output["pred_action_stage1"].float()
        if self.fdm_detach_action:
            stage0_actions = stage0_actions.detach()
            stage1_actions = stage1_actions.detach()

        current_tokens = current_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)
        target_tokens = target_future_dino_tokens.to(device=stage1_actions.device, dtype=torch.float32)

        if target_tokens.ndim == 3:
            if current_tokens.shape != target_tokens.shape:
                raise ValueError(
                    "current/future DINO token shape mismatch: "
                    f"current={tuple(current_tokens.shape)} future={tuple(target_tokens.shape)}"
                )
            pred_stage0 = self.fdm_predictor(current_tokens, stage0_actions, patch_indices=patch_indices)
            pred_stage1 = self.fdm_predictor(current_tokens, stage1_actions, patch_indices=patch_indices)
            fdm_loss, metrics = mip_fdm_loss(
                pred_stage0,
                pred_stage1,
                target_tokens.to(dtype=pred_stage1.dtype),
                stage0_weight=self.fdm_stage0_weight,
                rank_weight=self.fdm_rank_weight,
                rank_margin=self.fdm_rank_margin,
                rank_tau=self.fdm_rank_tau,
            )
            return fdm_loss.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype), metrics

        if target_tokens.ndim != 4:
            raise ValueError(f"target future DINO tokens must be [B,K,D] or [B,H,K,D], got {tuple(target_tokens.shape)}")
        if current_tokens.shape != (target_tokens.shape[0], target_tokens.shape[2], target_tokens.shape[3]):
            raise ValueError(
                "current/future DINO token shape mismatch: "
                f"current={tuple(current_tokens.shape)} future={tuple(target_tokens.shape)}"
            )

        num_horizons = target_tokens.shape[1]
        if len(self.fdm_horizon_weights) < num_horizons:
            raise ValueError(f"Need {num_horizons} horizon weights, got {self.fdm_horizon_weights}")
        weights = torch.tensor(
            self.fdm_horizon_weights[:num_horizons],
            device=stage1_actions.device,
            dtype=torch.float32,
        )
        weights = weights / weights.sum().clamp_min(1e-8)

        metrics = {}
        fdm_loss = action_output["loss"].new_zeros((), dtype=torch.float32)
        stage0_total = fdm_loss.clone()
        stage1_total = fdm_loss.clone()
        rank_total = fdm_loss.clone()
        for h in range(num_horizons):
            pred_stage0 = self.fdm_predictor(current_tokens, stage0_actions, patch_indices=patch_indices, horizon_indices=h)
            pred_stage1 = self.fdm_predictor(current_tokens, stage1_actions, patch_indices=patch_indices, horizon_indices=h)
            loss_h, metrics_h = mip_fdm_loss(
                pred_stage0,
                pred_stage1,
                target_tokens[:, h].to(dtype=pred_stage1.dtype),
                stage0_weight=self.fdm_stage0_weight,
                rank_weight=self.fdm_rank_weight,
                rank_margin=self.fdm_rank_margin,
                rank_tau=self.fdm_rank_tau,
            )
            weight = weights[h]
            fdm_loss = fdm_loss + weight * loss_h.float()
            stage0_total = stage0_total + weight * metrics_h["fdm_loss_stage0"].float()
            stage1_total = stage1_total + weight * metrics_h["fdm_loss_stage1"].float()
            rank_total = rank_total + weight * metrics_h["fdm_rank"].float()
            metrics[f"fdm_loss_h{h}"] = loss_h
            metrics[f"fdm_rank_h{h}"] = metrics_h["fdm_rank"]

        metrics.update({
            "fdm_loss": fdm_loss,
            "fdm_loss_stage0": stage0_total,
            "fdm_loss_stage1": stage1_total,
            "fdm_rank": rank_total,
        })
        return fdm_loss.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype), metrics

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image", None) for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("QwenMIPDINOFDM requires sample['future_image']; use a *_video_fdm data mix")

        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        condition_tokens, condition_mask, current_dino_tokens, patch_indices = self._build_fdm_action_condition(
            batch_images,
            instructions,
        )
        target_future_dino_tokens, future_patch_indices = self._encode_future_dino_tokens_for_fdm(
            future_images,
            condition_tokens.device,
            condition_tokens.dtype,
        )
        if not torch.equal(patch_indices, future_patch_indices):
            raise ValueError("current and future DINO patch indices must match for full-token FDM")

        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions),
                device=condition_tokens.device,
                dtype=condition_tokens.dtype,
            )
            actions_target = actions[:, -self.action_horizon :, :]

            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state),
                    device=condition_tokens.device,
                    dtype=condition_tokens.dtype,
                )

            action_output = self.action_model(
                condition_tokens,
                actions_target,
                state_tensor,
                encoder_attention_mask=condition_mask,
            )

        raw_action_loss = action_output["action_loss"]
        fdm_loss, fdm_metrics = self._compute_fdm_loss(
            action_output,
            current_dino_tokens,
            target_future_dino_tokens,
            patch_indices,
        )
        total_loss = action_output["loss"] + self.fdm_loss_weight * fdm_loss
        output = {
            "action_loss": total_loss,
            "raw_action_loss": raw_action_loss.detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in fdm_metrics.items()})
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs: str,
    ) -> np.ndarray:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        condition_tokens, condition_mask, _current_dino_tokens, _patch_indices = self._build_fdm_action_condition(
            batch_images,
            instructions,
        )
        state_tensor = (
            torch.from_numpy(np.array(state)).to(condition_tokens.device, dtype=condition_tokens.dtype)
            if state is not None
            else None
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                condition_tokens,
                state_tensor,
                encoder_attention_mask=condition_mask,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
