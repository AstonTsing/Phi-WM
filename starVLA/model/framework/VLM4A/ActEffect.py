# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0 (the "License");
"""Action-effect prediction framework.

Qwen-VL action-token features and frozen DINOv3 features condition a MIP
action head. During training, a DINO feature-dynamics model supervises the
effect of the predicted action sequence on future observations.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call
except ImportError:  # pragma: no cover - compatibility with older PyTorch
    from torch.nn.utils.stateless import functional_call

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import GR00TMIPActionHead, get_action_model
from starVLA.model.modules.dinov3_vit import DINOv3ViTModel
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model.DINOFeatureDynamics import (
    DINOFeatureDynamicsPredictor,
    fdm_distance_per_sample,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _get_config_value(config, key, default=None):
    if config is None:
        return default
    return config.get(key, default) if hasattr(config, "get") else getattr(config, key, default)


def _get_bool(config, key, default):
    value = _get_config_value(config, key, default)
    return value.lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def _select_last_view_images(batch_images, num_views, context):
    if num_views <= 0:
        return batch_images

    selected_images = []
    for images in batch_images:
        images = [images] if not isinstance(images, (list, tuple)) else list(images)
        if len(images) < num_views:
            raise ValueError(f"Expected at least {num_views} {context} images, got {len(images)}")
        if len(images) == num_views:
            selected_images.append(images)
            continue
        if len(images) % num_views:
            raise ValueError(f"Cannot infer view-major packing for {context}: {len(images)} images, {num_views} views")

        frames_per_view = len(images) // num_views
        selected_images.append([images[(view + 1) * frames_per_view - 1] for view in range(num_views)])
    return selected_images


def _encode_dino(model, batch_images, device, output_dtype):
    """Return raw DINO tokens, projected conditioning tokens, and patch indices."""
    if not model.dino_enabled:
        return None, None, None

    height, width = model.dino_input_hw
    pixels, view_counts = [], []
    for images in batch_images:
        images = [images] if not isinstance(images, (list, tuple)) else images
        view_counts.append(len(images))
        for image in images:
            image = to_pil_preserve(image).convert("RGB").resize((width, height))
            pixels.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))

    if len(set(view_counts)) != 1:
        raise ValueError(f"ActEffect expects the same number of views per sample, got {view_counts}")

    pixels = torch.stack(pixels)
    pixels = (pixels - model.dino_mean.to(pixels)) / model.dino_std.to(pixels)
    pixels = pixels.to(device)
    model.dino_model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=pixels.is_cuda):
        dino_output = model.dino_model(pixel_values=pixels)

    hidden_states = dino_output.last_hidden_state
    register_tokens = int(getattr(model.dino_model.config, "num_register_tokens", 0))
    token_groups = []
    if model.dino_include_cls:
        token_groups.append(hidden_states[:, :1])
    if model.dino_include_registers and register_tokens:
        token_groups.append(hidden_states[:, 1 : 1 + register_tokens])
    token_groups.append(hidden_states[:, 1 + register_tokens :])
    dino_tokens = torch.cat(token_groups, dim=1)

    if model.dino_normalize_tokens:
        dino_tokens = F.normalize(dino_tokens.float(), dim=-1).to(dtype=dino_tokens.dtype)

    batch_size, num_views = len(batch_images), view_counts[0]
    dino_tokens = dino_tokens.reshape(batch_size, num_views * dino_tokens.shape[1], dino_tokens.shape[-1])
    raw_dino_tokens = dino_tokens.float().detach() if model.fdm_detach_dino else dino_tokens.float()
    patch_indices = torch.arange(raw_dino_tokens.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
    projector_dtype = next(model.dino_projector.parameters()).dtype
    condition_tokens = model.dino_projector(dino_tokens.to(device=device, dtype=projector_dtype)).to(dtype=output_dtype)
    return raw_dino_tokens, condition_tokens, patch_indices


def _fdm_frozen(model, current_tokens, actions, *, patch_indices, horizon_index=None):
    """Run FDM with detached parameters while preserving gradients to actions."""
    detached_state = {name: parameter.detach() for name, parameter in model.fdm_predictor.named_parameters()}
    detached_state.update({name: buffer.detach() for name, buffer in model.fdm_predictor.named_buffers()})
    return functional_call(
        model.fdm_predictor,
        detached_state,
        (current_tokens, actions),
        {"patch_indices": patch_indices, "horizon_indices": horizon_index},
    )


def _compute_fdm_loss(model, action_output, actions_target, current_tokens, target_tokens, patch_indices):
    zero = action_output["loss"].new_zeros(())
    if not model.fdm_enabled or (
        model.fdm_dynamics_loss_weight <= 0
        and model.fdm_stage0_loss_weight <= 0
        and model.fdm_stage1_loss_weight <= 0
        and model.fdm_joint_rank_loss_weight <= 0
    ):
        return zero, {
            "fdm_loss": zero,
            "fdm_gt_loss": zero,
            "fdm_stage0_loss": zero,
            "fdm_stage1_loss": zero,
            "fdm_rank_loss": zero,
        }

    missing = [key for key in ("pred_action_stage0", "pred_action_stage1") if key not in action_output]
    if missing:
        raise KeyError(f"ActEffect requires MIP predictions, missing {missing}")

    stage0_actions = action_output["pred_action_stage0"].float()
    stage1_actions = action_output["pred_action_stage1"].float()
    actions_target = actions_target.detach().to(device=stage1_actions.device, dtype=torch.float32)
    current_tokens = current_tokens.detach().to(device=stage1_actions.device, dtype=torch.float32)
    target_tokens = target_tokens.detach().to(device=stage1_actions.device, dtype=torch.float32)

    if target_tokens.ndim == 3:
        expected_current_shape = target_tokens.shape
        num_horizons = 1
    elif target_tokens.ndim == 4:
        expected_current_shape = (target_tokens.shape[0], target_tokens.shape[2], target_tokens.shape[3])
        num_horizons = target_tokens.shape[1]
    else:
        raise ValueError(f"Future DINO tokens must be [B,K,D] or [B,H,K,D], got {tuple(target_tokens.shape)}")
    if tuple(current_tokens.shape) != tuple(expected_current_shape):
        raise ValueError(
            f"Current/future DINO token mismatch: {tuple(current_tokens.shape)} vs {tuple(target_tokens.shape)}"
        )
    if len(model.fdm_horizon_weights) < num_horizons:
        raise ValueError(f"Need {num_horizons} horizon weights, got {model.fdm_horizon_weights}")

    weights = torch.tensor(
        model.fdm_horizon_weights[:num_horizons],
        device=stage1_actions.device,
        dtype=torch.float32,
    )
    weights = weights / weights.sum().clamp_min(1e-8)
    total_loss = zero.float()
    gt_loss_total = zero.float()
    stage0_loss_total = zero.float()
    stage1_loss_total = zero.float()
    rank_loss_total = zero.float()
    distance_gt_total = zero.float()
    distance_stage0_total = zero.float()
    distance_stage1_total = zero.float()
    order_total = zero.float()
    agreement_total = zero.float()
    metrics = {}

    action_error_stage0 = (stage0_actions - actions_target).square().mean(dim=(-1, -2))
    action_error_stage1 = (stage1_actions - actions_target).square().mean(dim=(-1, -2))
    action_order = action_error_stage1 < action_error_stage0

    for horizon_index in range(num_horizons):
        predictor_horizon = None if target_tokens.ndim == 3 else horizon_index
        target = target_tokens if target_tokens.ndim == 3 else target_tokens[:, horizon_index]
        weight = weights[horizon_index]
        predictor_kwargs = {"patch_indices": patch_indices, "horizon_indices": predictor_horizon}

        # Real action supervision: this branch only updates the FDM.
        prediction_gt = model.fdm_predictor(current_tokens, actions_target, **predictor_kwargs)
        distance_gt = fdm_distance_per_sample(prediction_gt, target)
        gt_loss = distance_gt.mean()

        # Stage-0 consequence matching: detached FDM parameters, policy gradient only.
        prediction_stage0_policy = _fdm_frozen(
            model,
            current_tokens,
            stage0_actions,
            patch_indices=patch_indices,
            horizon_index=predictor_horizon,
        )
        distance_stage0 = fdm_distance_per_sample(prediction_stage0_policy, target)
        stage0_loss = distance_stage0.mean()

        # Stage-1 consequence matching: detached FDM parameters, policy gradient only.
        prediction_stage1_policy = _fdm_frozen(
            model,
            current_tokens,
            stage1_actions,
            patch_indices=patch_indices,
            horizon_index=predictor_horizon,
        )
        stage1_loss = fdm_distance_per_sample(prediction_stage1_policy, target).mean()

        # Joint rank: Stage-1 updates both policy and FDM. Stage-0 is a detached baseline.
        prediction_stage1_rank = model.fdm_predictor(current_tokens, stage1_actions, **predictor_kwargs)
        distance_stage1 = fdm_distance_per_sample(prediction_stage1_rank, target)
        rank_loss = F.softplus(
            (distance_stage1 - distance_stage0.detach() + model.fdm_rank_margin) / model.fdm_rank_tau
        ).mean()

        horizon_loss = (
            model.fdm_dynamics_loss_weight * gt_loss
            + model.fdm_stage0_loss_weight * stage0_loss
            + model.fdm_stage1_loss_weight * stage1_loss
            + model.fdm_joint_rank_loss_weight * rank_loss
        )
        total_loss = total_loss + weight * horizon_loss.float()
        gt_loss_total = gt_loss_total + weight * gt_loss.float()
        stage0_loss_total = stage0_loss_total + weight * stage0_loss.float()
        stage1_loss_total = stage1_loss_total + weight * stage1_loss.float()
        rank_loss_total = rank_loss_total + weight * rank_loss.float()
        distance_gt_total = distance_gt_total + weight * distance_gt.mean().float()
        distance_stage0_total = distance_stage0_total + weight * distance_stage0.mean().float()
        distance_stage1_total = distance_stage1_total + weight * distance_stage1.mean().float()
        fdm_order = distance_stage1 < distance_stage0
        order_total = order_total + weight * fdm_order.float().mean()
        agreement_total = agreement_total + weight * (fdm_order == action_order).float().mean()

        if num_horizons > 1:
            metrics[f"fdm_gt_loss_h{horizon_index}"] = gt_loss.detach()
            metrics[f"fdm_stage0_loss_h{horizon_index}"] = stage0_loss.detach()
            metrics[f"fdm_stage1_loss_h{horizon_index}"] = stage1_loss.detach()
            metrics[f"fdm_rank_loss_h{horizon_index}"] = rank_loss.detach()

    metrics.update({
        "fdm_loss": total_loss.detach(),
        "fdm_gt_loss": gt_loss_total.detach(),
        "fdm_stage0_loss": stage0_loss_total.detach(),
        "fdm_stage1_loss": stage1_loss_total.detach(),
        "fdm_rank_loss": rank_loss_total.detach(),
        "fdm_dist_gt": distance_gt_total.detach(),
        "fdm_dist_stage0": distance_stage0_total.detach(),
        "fdm_dist_stage1": distance_stage1_total.detach(),
        "fdm_gap_stage1_stage0": (distance_stage0_total - distance_stage1_total).detach(),
        "fdm_order_stage1_stage0": order_total.detach(),
        "raw_action_mse_stage0": action_error_stage0.mean().detach(),
        "raw_action_mse_stage1": action_error_stage1.mean().detach(),
        "action_order_stage1_stage0": action_order.float().mean().detach(),
        "action_fdm_order_agreement": agreement_total.detach(),
    })
    total_loss = total_loss.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype)
    return total_loss, metrics


# ──────────────────────────────────────────────────────────────────────
#  Default Config for ActEffect
#  - YAML values override these defaults; extra YAML keys are preserved
# ──────────────────────────────────────────────────────────────────────
@dataclass
class ActEffectDefaultConfig:
    """Default parameters for action prediction with feature-dynamics supervision."""

    # --- Registry identifier (must match @FRAMEWORK_REGISTRY.register) ---
    name: str = "ActEffect"

    # === VLM backbone (Qwen3-VL) ===
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    # === DINOv3 visual features ===
    dinov3: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "model_path": "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
            "freeze_dino": True,
            "input_size": [224, 224],
            "include_cls_token": False,
            "include_register_tokens": False,
            "normalize_tokens": False,
            "projector": {"layer_norm": True, "bias": True},
        }
    )

    # === MIP action head ===
    action_model: dict = field(
        default_factory=lambda: {
            "head_type": "gr00t_mip",
            "action_model_type": "DiT-B",
            "action_hidden_dim": 1024,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "action_dim": 7,
            "state_dim": 7,
            "action_horizon": 8,
            "future_action_window_size": 7,
            "past_action_window_size": 0,
            "mip_t": 0.9,
            "mip_action_stage0_weight": 1.0,
            "mip_action_stage1_weight": 1.0,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "num_timestep_buckets": 1000,
            "num_inference_timesteps": 4,
            "num_target_vision_tokens": 32,
            "diffusion_model_cfg": {
                "cross_attention_dim": 2048,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "num_layers": 16,
                "output_dim": 1024,
                "positional_embeddings": None,
            },
        }
    )

    # === Future DINO feature-dynamics model ===
    fdm: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "dynamics_loss_weight": 1.0,
            "stage0_loss_weight": 0.05,
            "stage1_loss_weight": 0.05,
            "joint_rank_loss_weight": 0.005,
            "rank_margin": 0.0,
            "rank_tau": 0.01,
            "detach_dino": True,
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


@FRAMEWORK_REGISTRY.register("ActEffect")
class ActEffect(baseframework):
    """Qwen-VL action prediction with DINO feature-dynamics supervision."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(ActEffectDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.qwen_hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_hidden_size
        self.action_model: GR00TMIPActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon

        self.dino_cfg = self.config.framework.get("dinov3", {})
        self.dino_enabled = _get_bool(self.dino_cfg, "enabled", True)
        dino_input_size = _get_config_value(self.dino_cfg, "input_size", [224, 224])
        self.dino_input_hw = (dino_input_size, dino_input_size) if isinstance(dino_input_size, int) else tuple(dino_input_size)
        if len(self.dino_input_hw) != 2:
            raise ValueError(f"Expected DINO input_size [height, width], got {dino_input_size}")
        self.dino_input_hw = tuple(int(size) for size in self.dino_input_hw)
        self.dino_include_cls = _get_bool(self.dino_cfg, "include_cls_token", False)
        self.dino_include_registers = _get_bool(self.dino_cfg, "include_register_tokens", False)
        self.dino_normalize_tokens = _get_bool(self.dino_cfg, "normalize_tokens", False)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        if self.dino_enabled:
            model_path = _get_config_value(
                self.dino_cfg,
                "model_path",
                "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
            )
            self.dino_model = DINOv3ViTModel.from_pretrained(str(model_path))
            if _get_bool(self.dino_cfg, "freeze_dino", True):
                self.dino_model.eval()
                for parameter in self.dino_model.parameters():
                    parameter.requires_grad = False

            projector_config = _get_config_value(self.dino_cfg, "projector", {})
            dino_hidden_size = int(self.dino_model.config.hidden_size)
            projector_bias = _get_bool(projector_config, "bias", True)
            self.dino_projector = (
                nn.Sequential(nn.LayerNorm(dino_hidden_size), nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=projector_bias))
                if _get_bool(projector_config, "layer_norm", True)
                else nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=projector_bias)
            )
        else:
            self.dino_model = None
            self.dino_projector = None

        self.fdm_cfg = self.config.framework.get("fdm", {})
        self.fdm_enabled = _get_bool(self.fdm_cfg, "enabled", True)
        self.fdm_dynamics_loss_weight = float(_get_config_value(self.fdm_cfg, "dynamics_loss_weight", 1.0))
        self.fdm_stage0_loss_weight = float(_get_config_value(self.fdm_cfg, "stage0_loss_weight", 0.05))
        self.fdm_stage1_loss_weight = float(_get_config_value(self.fdm_cfg, "stage1_loss_weight", 0.05))
        self.fdm_joint_rank_loss_weight = float(_get_config_value(self.fdm_cfg, "joint_rank_loss_weight", 0.005))
        self.fdm_rank_margin = float(_get_config_value(self.fdm_cfg, "rank_margin", 0.0))
        self.fdm_rank_tau = float(_get_config_value(self.fdm_cfg, "rank_tau", 0.01))
        if self.fdm_rank_tau <= 0:
            raise ValueError(f"fdm.rank_tau must be positive, got {self.fdm_rank_tau}")
        self.fdm_detach_dino = _get_bool(self.fdm_cfg, "detach_dino", True)
        self.fdm_condition_num_images = int(_get_config_value(self.fdm_cfg, "condition_num_images", 2))
        self.fdm_multi_horizon = _get_bool(self.fdm_cfg, "multi_horizon", False)
        self.fdm_max_horizons = int(_get_config_value(self.fdm_cfg, "max_horizons", 1))
        horizon_weights = _get_config_value(self.fdm_cfg, "horizon_weights", [1.0])
        if isinstance(horizon_weights, str):
            horizon_weights = [value.strip() for value in horizon_weights.strip().strip("[]").split(",") if value.strip()]
        self.fdm_horizon_weights = [float(weight) for weight in horizon_weights]

        if self.fdm_enabled:
            if self.dino_model is None:
                raise ValueError("ActEffect requires DINOv3 when fdm.enabled=true")
            self.fdm_predictor = DINOFeatureDynamicsPredictor(
                dino_dim=int(self.dino_model.config.hidden_size),
                action_dim=int(self.config.framework.action_model.action_dim),
                action_horizon=self.action_horizon,
                hidden_dim=int(_get_config_value(self.fdm_cfg, "hidden_dim", 768)),
                depth=int(_get_config_value(self.fdm_cfg, "depth", 4)),
                num_heads=int(_get_config_value(self.fdm_cfg, "num_heads", 8)),
                mlp_ratio=float(_get_config_value(self.fdm_cfg, "mlp_ratio", 4.0)),
                dropout=float(_get_config_value(self.fdm_cfg, "dropout", 0.0)),
                max_patches=int(_get_config_value(self.fdm_cfg, "max_patches", 2048)),
                max_horizons=self.fdm_max_horizons if self.fdm_multi_horizon else 1,
            )
        else:
            self.fdm_predictor = None

        self.action_token = "<robot_action_0>"
        token_ids = self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                "ActEffect expects the action placeholder to be a single token. "
                f"Got token ids {token_ids}. Use the Action-token Qwen checkpoint or add this token first."
            )
        self.action_token_id = int(token_ids[0])

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image") for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("ActEffect requires sample['future_image']; use a *_video_fdm data mix")

        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        states = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        condition_images = _select_last_view_images(batch_images, self.fdm_condition_num_images, "condition")
        action_tokens = self.action_token * self.action_horizon
        prompt_suffix = f" Please predict the next {self.action_horizon} robot actions: <action>{action_tokens}<action>."
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=condition_images,
            instructions=[instruction + prompt_suffix for instruction in instructions],
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("ActEffect requires input_ids for action-token validation")
        action_token_counts = (input_ids == self.action_token_id).sum(dim=1)
        if (action_token_counts < self.action_horizon).any():
            raise RuntimeError(
                f"Insufficient OFT action tokens: counts={action_token_counts.tolist()}, required={self.action_horizon}"
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwen_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        qwen_tokens = qwen_outputs.hidden_states[-1]
        current_tokens, dino_condition, patch_indices = _encode_dino(
            self,
            condition_images,
            qwen_tokens.device,
            qwen_tokens.dtype,
        )
        if dino_condition is not None:
            if attention_mask is None:
                attention_mask = torch.ones(qwen_tokens.shape[:2], device=qwen_tokens.device, dtype=torch.bool)
            dino_mask = torch.ones(dino_condition.shape[:2], device=dino_condition.device, dtype=torch.bool)
            condition = torch.cat([qwen_tokens, dino_condition], dim=1)
            attention_mask = torch.cat([attention_mask, dino_mask], dim=1)
        else:
            condition = qwen_tokens

        if self.fdm_multi_horizon:
            all_horizons, horizon_counts = [], []
            for images in future_images:
                images = [images] if not isinstance(images, (list, tuple)) else list(images)
                if self.fdm_condition_num_images <= 0 or len(images) % self.fdm_condition_num_images:
                    raise ValueError("Cannot infer future horizon packing")
                frames_per_view = len(images) // self.fdm_condition_num_images
                num_horizons = min(frames_per_view, self.fdm_max_horizons)
                horizon_counts.append(num_horizons)
                all_horizons.append([
                    [images[view * frames_per_view + horizon] for view in range(self.fdm_condition_num_images)]
                    for horizon in range(num_horizons)
                ])
            if len(set(horizon_counts)) != 1:
                raise ValueError(f"Future horizon counts must match across batch, got {horizon_counts}")
            batch_size, num_horizons = len(all_horizons), len(all_horizons[0])
            future_tokens, _future_condition, future_patch_indices = _encode_dino(
                self,
                [views for sample in all_horizons for views in sample],
                condition.device,
                condition.dtype,
            )
            if future_tokens is None:
                raise ValueError("ActEffect requires DINOv3 future tokens")
            future_tokens = future_tokens.reshape(batch_size, num_horizons, future_tokens.shape[-2], future_tokens.shape[-1])
            future_patch_indices = future_patch_indices.reshape(batch_size, num_horizons, future_patch_indices.shape[-1])
            if not torch.equal(future_patch_indices, future_patch_indices[:, :1].expand_as(future_patch_indices)):
                raise ValueError("Future DINO patch indices must match across horizons")
            future_patch_indices = future_patch_indices[:, 0]
        else:
            future_tokens, _future_condition, future_patch_indices = _encode_dino(
                self,
                _select_last_view_images(future_images, self.fdm_condition_num_images, "future"),
                condition.device,
                condition.dtype,
            )
            if future_tokens is None:
                raise ValueError("ActEffect requires DINOv3 future tokens")

        if not torch.equal(patch_indices, future_patch_indices):
            raise ValueError("Current and future DINO patch indices must match")

        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=condition.device,
            dtype=condition.dtype,
        )
        actions_target = actions[:, -self.action_horizon :]
        states = torch.tensor(np.array(states), device=condition.device, dtype=condition.dtype) if states is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            action_output = self.action_model(
                condition,
                actions_target,
                states,
                encoder_attention_mask=attention_mask,
            )

        fdm_loss, metrics = _compute_fdm_loss(
            self,
            action_output,
            actions_target,
            current_tokens,
            future_tokens,
            patch_indices,
        )
        output = {
            "action_loss": action_output["loss"] + fdm_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in metrics.items()})
        return output

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        **kwargs,
    ) -> dict:
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if image_size:
            batch_images = resize_images(batch_images, target_size=image_size)
        batch_images = _select_last_view_images(batch_images, self.fdm_condition_num_images, "condition")

        action_tokens = self.action_token * self.action_horizon
        prompt_suffix = f" Please predict the next {self.action_horizon} robot actions: <action>{action_tokens}<action>."
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=[example["lang"] + prompt_suffix for example in examples],
        )
        attention_mask = qwen_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool)
        input_ids = qwen_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("ActEffect requires input_ids for action-token validation")
        action_token_counts = (input_ids == self.action_token_id).sum(dim=1)
        if (action_token_counts < self.action_horizon).any():
            raise RuntimeError(
                f"Insufficient OFT action tokens: counts={action_token_counts.tolist()}, required={self.action_horizon}"
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwen_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
        qwen_tokens = qwen_outputs.hidden_states[-1]
        _current_tokens, dino_condition, _patch_indices = _encode_dino(
            self,
            batch_images,
            qwen_tokens.device,
            qwen_tokens.dtype,
        )
        if dino_condition is not None:
            if attention_mask is None:
                attention_mask = torch.ones(qwen_tokens.shape[:2], device=qwen_tokens.device, dtype=torch.bool)
            dino_mask = torch.ones(dino_condition.shape[:2], device=dino_condition.device, dtype=torch.bool)
            condition = torch.cat([qwen_tokens, dino_condition], dim=1)
            attention_mask = torch.cat([attention_mask, dino_mask], dim=1)
        else:
            condition = qwen_tokens

        use_state = getattr(self.action_model, "state_encoder", None) is not None
        states = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        states = torch.from_numpy(np.array(states)).to(condition.device, dtype=condition.dtype) if states is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(condition, states, encoder_attention_mask=attention_mask)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
