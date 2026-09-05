# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0 (the "License");
"""Standalone OFT-prompted Qwen, MIP, DINO, and LeWM-style FDM framework.

The VLM receives explicit action placeholders, and its full hidden sequence
plus frozen DINOv3 patch tokens condition a two-stage MIP action head. A
train-only DINO feature-dynamics model supplies future-feature ranking loss.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_ActionHeader import GR00TMIPActionHead, get_action_model
from starVLA.model.modules.dinov3_vit import DINOv3ViTModel
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.world_model.DINOFeatureDynamicsLeWM import DINOFeatureDynamicsPredictor, mip_fdm_loss
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    return cfg.get(key, default) if hasattr(cfg, "get") else getattr(cfg, key, default)


def _cfg_bool(cfg, key, default):
    value = _cfg_get(cfg, key, default)
    return value.lower() in {"1", "true", "yes", "on"} if isinstance(value, str) else bool(value)


def _cfg_float_list(cfg, key, default):
    value = _cfg_get(cfg, key, default)
    if isinstance(value, str):
        value = [item.strip() for item in value.strip().strip("[]").split(",") if item.strip()]
    return [float(item) for item in value]


def _to_hw(value):
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected [height, width], got {value}")
    return int(value[0]), int(value[1])


@dataclass
class ActEffectLewmDefaultConfig:
    """Complete default configuration; this framework has no VLA-framework parent."""

    name: str = "ActEffectLewm"
    qwenvl: dict = field(default_factory=lambda: {
        "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        "attn_implementation": "flash_attention_2", "vl_hidden_dim": 2048,
    })
    dinov3: dict = field(default_factory=lambda: {
        "enabled": True,
        "model_path": "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
        "freeze_dino": True, "input_size": [224, 224],
        "include_cls_token": False, "include_register_tokens": False,
        "normalize_tokens": False, "projector": {"layer_norm": True, "bias": True},
    })
    action_model: dict = field(default_factory=lambda: {
        "head_type": "gr00t_mip", "action_model_type": "DiT-B", "action_hidden_dim": 1024,
        "hidden_size": 1024, "add_pos_embed": True, "max_seq_len": 1024,
        "action_dim": 7, "state_dim": 7, "action_horizon": 8,
        "future_action_window_size": 7, "past_action_window_size": 0, "mip_t": 0.9,
        "mip_action_stage0_weight": 1.0, "mip_action_stage1_weight": 1.0,
        "noise_beta_alpha": 1.5, "noise_beta_beta": 1.0, "noise_s": 0.999,
        "num_timestep_buckets": 1000, "num_inference_timesteps": 4,
        "num_target_vision_tokens": 32,
        "diffusion_model_cfg": {
            "cross_attention_dim": 2048, "dropout": 0.2, "final_dropout": True,
            "interleave_self_attention": True, "norm_type": "ada_norm", "num_layers": 16,
            "output_dim": 1024, "positional_embeddings": None,
        },
    })
    fdm: dict = field(default_factory=lambda: {
        "enabled": True, "loss_weight": 0.05, "stage0_weight": 0.0,
        "rank_weight": 0.1, "rank_margin": 0.0, "rank_tau": 0.1,
        "detach_dino": True, "detach_action": False, "hidden_dim": 768,
        "depth": 4, "num_heads": 8, "mlp_ratio": 4.0, "dropout": 0.0,
        "max_patches": 2048, "condition_num_images": 2, "multi_horizon": False,
        "max_horizons": 1, "horizon_weights": [1.0],
    })


@FRAMEWORK_REGISTRY.register("ActEffectLewm")
class ActEffectLewm(baseframework):
    """OFT QwenVL + frozen DINOv3 + MIP action head + DINO dynamics loss."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(ActEffectLewmDefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.qwen_hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_hidden_size
        self.action_model: GR00TMIPActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self.chunk_len = self.action_horizon

        self.dino_cfg = self.config.framework.get("dinov3", {})
        self.dino_enabled = _cfg_bool(self.dino_cfg, "enabled", True)
        self.dino_input_hw = _to_hw(_cfg_get(self.dino_cfg, "input_size", [224, 224]))
        self.dino_include_cls = _cfg_bool(self.dino_cfg, "include_cls_token", False)
        self.dino_include_registers = _cfg_bool(self.dino_cfg, "include_register_tokens", False)
        self.dino_normalize_tokens = _cfg_bool(self.dino_cfg, "normalize_tokens", False)
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._init_dino()

        self.fdm_cfg = self.config.framework.get("fdm", {})
        self.fdm_enabled = _cfg_bool(self.fdm_cfg, "enabled", True)
        self.fdm_loss_weight = float(_cfg_get(self.fdm_cfg, "loss_weight", 0.05))
        self.fdm_stage0_weight = float(_cfg_get(self.fdm_cfg, "stage0_weight", 0.25))
        self.fdm_rank_weight = float(_cfg_get(self.fdm_cfg, "rank_weight", 0.1))
        self.fdm_rank_margin = float(_cfg_get(self.fdm_cfg, "rank_margin", 0.0))
        self.fdm_rank_tau = float(_cfg_get(self.fdm_cfg, "rank_tau", 0.1))
        self.fdm_detach_dino = _cfg_bool(self.fdm_cfg, "detach_dino", True)
        self.fdm_detach_action = _cfg_bool(self.fdm_cfg, "detach_action", False)
        self.fdm_condition_num_images = int(_cfg_get(self.fdm_cfg, "condition_num_images", 2))
        self.fdm_multi_horizon = _cfg_bool(self.fdm_cfg, "multi_horizon", False)
        self.fdm_max_horizons = int(_cfg_get(self.fdm_cfg, "max_horizons", 1))
        self.fdm_horizon_weights = _cfg_float_list(self.fdm_cfg, "horizon_weights", [1.0])
        if self.fdm_enabled:
            if self.dino_model is None:
                raise ValueError("ActEffectLewm requires DINOv3 to be enabled")
            self.fdm_predictor = DINOFeatureDynamicsPredictor(
                dino_dim=int(self.dino_model.config.hidden_size),
                action_dim=int(self.config.framework.action_model.action_dim),
                action_horizon=self.action_horizon,
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

        self.action_token = "<robot_action_0>"
        token_ids = self.qwen_vl_interface.processor.tokenizer(self.action_token, add_special_tokens=False)["input_ids"]
        if len(token_ids) != 1:
            raise ValueError(
                "ActEffectLewm expects the action placeholder to be a single token. "
                f"Got token ids {token_ids}. Use the Action-token Qwen checkpoint or add this token first."
            )
        self.action_token_id = int(token_ids[0])

    def _init_dino(self) -> None:
        if not self.dino_enabled:
            self.dino_model = self.dino_projector = None
            return
        model_path = _cfg_get(
            self.dino_cfg, "model_path",
            "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
        )
        self.dino_model = DINOv3ViTModel.from_pretrained(str(model_path))
        if _cfg_bool(self.dino_cfg, "freeze_dino", True):
            self.dino_model.eval()
            for parameter in self.dino_model.parameters():
                parameter.requires_grad = False
        dino_hidden_size = int(self.dino_model.config.hidden_size)
        projector_cfg = _cfg_get(self.dino_cfg, "projector", {})
        bias = _cfg_bool(projector_cfg, "bias", True)
        self.dino_projector = (
            nn.Sequential(nn.LayerNorm(dino_hidden_size), nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=bias))
            if _cfg_bool(projector_cfg, "layer_norm", True)
            else nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=bias)
        )

    def _preprocess_dino_flat_views(self, batch_images: List[List]) -> Tuple[torch.Tensor, List[int]]:
        height, width = self.dino_input_hw
        tensors, view_counts = [], []
        for images in batch_images:
            images = [images] if not isinstance(images, (list, tuple)) else images
            view_counts.append(len(images))
            for image in images:
                image = to_pil_preserve(image).convert("RGB").resize((width, height))
                tensors.append(torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1))
        if len(set(view_counts)) != 1:
            raise ValueError(f"ActEffectLewm expects same views per sample, got {view_counts}")
        pixels = torch.stack(tensors)
        return (pixels - self.dino_mean.to(pixels)) / self.dino_std.to(pixels), view_counts

    def _select_dino_tokens(self, hidden: torch.Tensor) -> torch.Tensor:
        registers = int(getattr(self.dino_model.config, "num_register_tokens", 0))
        pieces = []
        if self.dino_include_cls:
            pieces.append(hidden[:, :1])
        if self.dino_include_registers and registers:
            pieces.append(hidden[:, 1 : 1 + registers])
        pieces.append(hidden[:, 1 + registers :])
        return torch.cat(pieces, dim=1)

    def _select_last_images_per_view(self, batch_images: List[List], *, context: str) -> List[List]:
        num_views = self.fdm_condition_num_images
        if num_views <= 0:
            return batch_images
        selected = []
        for images in batch_images:
            images = [images] if not isinstance(images, (list, tuple)) else list(images)
            if len(images) < num_views:
                raise ValueError(f"Expected at least {num_views} {context} images, got {len(images)}")
            if len(images) == num_views:
                selected.append(images)
                continue
            if len(images) % num_views:
                raise ValueError(f"Cannot infer view-major packing for {context}: {len(images)} images, {num_views} views")
            frames = len(images) // num_views
            selected.append([images[(view + 1) * frames - 1] for view in range(num_views)])
        return selected

    def _future_image_horizons(self, future_images: List[List]) -> List[List[List]]:
        all_horizons, counts = [], []
        for images in future_images:
            images = [images] if not isinstance(images, (list, tuple)) else list(images)
            if self.fdm_condition_num_images <= 0 or len(images) % self.fdm_condition_num_images:
                raise ValueError("Cannot infer future horizon packing")
            frames = len(images) // self.fdm_condition_num_images
            count = min(frames, self.fdm_max_horizons)
            counts.append(count)
            all_horizons.append([
                [images[view * frames + horizon] for view in range(self.fdm_condition_num_images)]
                for horizon in range(count)
            ])
        if len(set(counts)) != 1:
            raise ValueError(f"Future horizon counts must match across batch, got {counts}")
        return all_horizons

    def _select_current_condition_images(self, batch_images: List[List]) -> List[List]:
        """Compatibility entry point used by the DirectRank subclass."""
        return self._select_last_images_per_view(batch_images, context="condition")

    def _select_future_target_images(self, future_images: List[List]) -> List[List]:
        return self._select_last_images_per_view(future_images, context="future")

    def _append_oft_action_prompt(self, instructions: List[str]) -> List[str]:
        placeholders = self.action_token * self.action_horizon
        suffix = f" Please predict the next {self.action_horizon} robot actions: <action>{placeholders}<action>."
        return [instruction + suffix for instruction in instructions]

    def _encode_qwen(self, batch_images, instructions):
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=self._append_oft_action_prompt(instructions)
        )
        mask = inputs.get("attention_mask")
        if mask is not None:
            mask = mask.to(dtype=torch.bool)
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("ActEffectLewm requires input_ids for action-token validation")
        counts = (input_ids == self.action_token_id).sum(dim=1)
        if (counts < self.action_horizon).any():
            raise RuntimeError(f"Insufficient OFT action tokens: counts={counts.tolist()}, required={self.action_horizon}")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface(
                **inputs, output_attentions=False, output_hidden_states=True, return_dict=True
            )
        return outputs.hidden_states[-1], mask

    def _encode_dino_raw_and_condition(self, batch_images, device, condition_dtype):
        if not self.dino_enabled:
            return None, None, None
        pixels, view_counts = self._preprocess_dino_flat_views(batch_images)
        pixels = pixels.to(device)
        self.dino_model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=pixels.is_cuda):
            output = self.dino_model(pixel_values=pixels)
        tokens = self._select_dino_tokens(output.last_hidden_state)
        if self.dino_normalize_tokens:
            tokens = F.normalize(tokens.float(), dim=-1).to(dtype=tokens.dtype)
        batch_size, num_views = len(batch_images), view_counts[0]
        tokens = tokens.reshape(batch_size, num_views * tokens.shape[1], tokens.shape[-1])
        raw_tokens = tokens.float().detach() if self.fdm_detach_dino else tokens.float()
        patch_indices = torch.arange(raw_tokens.shape[1], device=device).unsqueeze(0).expand(batch_size, -1)
        projector_dtype = next(self.dino_projector.parameters()).dtype
        condition = self.dino_projector(tokens.to(device=device, dtype=projector_dtype)).to(dtype=condition_dtype)
        return raw_tokens, condition, patch_indices

    def _build_fdm_action_condition(self, batch_images, instructions):
        batch_images = self._select_last_images_per_view(batch_images, context="condition")
        qwen_tokens, qwen_mask = self._encode_qwen(batch_images, instructions)
        raw_dino, dino_tokens, patch_indices = self._encode_dino_raw_and_condition(
            batch_images, qwen_tokens.device, qwen_tokens.dtype
        )
        if dino_tokens is None:
            return qwen_tokens, qwen_mask, raw_dino, patch_indices
        if qwen_mask is None:
            qwen_mask = torch.ones(qwen_tokens.shape[:2], device=qwen_tokens.device, dtype=torch.bool)
        dino_mask = torch.ones(dino_tokens.shape[:2], device=dino_tokens.device, dtype=torch.bool)
        return torch.cat([qwen_tokens, dino_tokens], 1), torch.cat([qwen_mask, dino_mask], 1), raw_dino, patch_indices

    def _encode_future_dino_tokens_for_fdm(self, future_images, device, dtype):
        if not self.fdm_multi_horizon:
            raw, _condition, patches = self._encode_dino_raw_and_condition(
                self._select_last_images_per_view(future_images, context="future"), device, dtype
            )
            if raw is None:
                raise ValueError("ActEffectLewm requires DINOv3 future tokens")
            return raw, patches
        horizons = self._future_image_horizons(future_images)
        batch_size, num_horizons = len(horizons), len(horizons[0])
        raw, _condition, patches = self._encode_dino_raw_and_condition(
            [views for sample in horizons for views in sample], device, dtype
        )
        if raw is None:
            raise ValueError("ActEffectLewm requires DINOv3 future tokens")
        raw = raw.reshape(batch_size, num_horizons, raw.shape[-2], raw.shape[-1])
        patches = patches.reshape(batch_size, num_horizons, patches.shape[-1])
        if not torch.equal(patches, patches[:, :1].expand_as(patches)):
            raise ValueError("Future DINO patch indices must match across horizons")
        return raw, patches[:, 0]

    def _compute_fdm_loss(self, action_output, current_tokens, target_tokens, patch_indices):
        if not self.fdm_enabled or self.fdm_loss_weight <= 0:
            zero = action_output["loss"].new_zeros(())
            return zero, {"fdm_loss": zero, "fdm_loss_stage0": zero, "fdm_loss_stage1": zero, "fdm_rank": zero}
        missing = [key for key in ("pred_action_stage0", "pred_action_stage1") if key not in action_output]
        if missing:
            raise KeyError(f"ActEffectLewm requires MIP predictions, missing {missing}")
        stage0, stage1 = action_output["pred_action_stage0"].float(), action_output["pred_action_stage1"].float()
        if self.fdm_detach_action:
            stage0, stage1 = stage0.detach(), stage1.detach()
        current = current_tokens.to(device=stage1.device, dtype=torch.float32)
        target = target_tokens.to(device=stage1.device, dtype=torch.float32)

        def loss_at_horizon(horizon=None):
            predictor_kwargs = {"patch_indices": patch_indices}
            if horizon is not None:
                predictor_kwargs["horizon_indices"] = horizon
            pred0 = self.fdm_predictor(current, stage0, **predictor_kwargs)
            pred1 = self.fdm_predictor(current, stage1, **predictor_kwargs)
            target_h = target if horizon is None else target[:, horizon]
            return mip_fdm_loss(
                pred0, pred1, target_h.to(dtype=pred1.dtype),
                stage0_weight=self.fdm_stage0_weight, rank_weight=self.fdm_rank_weight,
                rank_margin=self.fdm_rank_margin, rank_tau=self.fdm_rank_tau,
            )

        if target.ndim == 3:
            if current.shape != target.shape:
                raise ValueError(f"current/future DINO mismatch: {tuple(current.shape)} vs {tuple(target.shape)}")
            loss, metrics = loss_at_horizon()
            return loss.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype), metrics
        if target.ndim != 4 or current.shape != (target.shape[0], target.shape[2], target.shape[3]):
            raise ValueError("current/future multi-horizon DINO token shapes do not match")
        num_horizons = target.shape[1]
        if len(self.fdm_horizon_weights) < num_horizons:
            raise ValueError(f"Need {num_horizons} horizon weights, got {self.fdm_horizon_weights}")
        weights = torch.tensor(self.fdm_horizon_weights[:num_horizons], device=stage1.device, dtype=torch.float32)
        weights = weights / weights.sum().clamp_min(1e-8)
        total = action_output["loss"].new_zeros((), dtype=torch.float32)
        stage0_total, stage1_total, rank_total, metrics = total.clone(), total.clone(), total.clone(), {}
        for horizon in range(num_horizons):
            loss, horizon_metrics = loss_at_horizon(horizon)
            total = total + weights[horizon] * loss.float()
            stage0_total = stage0_total + weights[horizon] * horizon_metrics["fdm_loss_stage0"].float()
            stage1_total = stage1_total + weights[horizon] * horizon_metrics["fdm_loss_stage1"].float()
            rank_total = rank_total + weights[horizon] * horizon_metrics["fdm_rank"].float()
            metrics[f"fdm_loss_h{horizon}"] = loss
            metrics[f"fdm_rank_h{horizon}"] = horizon_metrics["fdm_rank"]
        metrics.update({"fdm_loss": total, "fdm_loss_stage0": stage0_total,
                        "fdm_loss_stage1": stage1_total, "fdm_rank": rank_total})
        return total.to(device=action_output["loss"].device, dtype=action_output["loss"].dtype), metrics

    def forward(self, examples: List[dict] = None, **kwargs) -> Tuple:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image") for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("ActEffectLewm requires sample['future_image']; use a *_video_fdm data mix")
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        condition, mask, current, patches = self._build_fdm_action_condition(batch_images, instructions)
        future, future_patches = self._encode_future_dino_tokens_for_fdm(future_images, condition.device, condition.dtype)
        if not torch.equal(patches, future_patches):
            raise ValueError("Current and future DINO patch indices must match")
        actions = torch.tensor(np.array([example["action"] for example in examples]), device=condition.device, dtype=condition.dtype)
        state_tensor = torch.tensor(np.array(state), device=condition.device, dtype=condition.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            action_output = self.action_model(condition, actions[:, -self.action_horizon :], state_tensor, encoder_attention_mask=mask)
        fdm_loss, metrics = self._compute_fdm_loss(action_output, current, future, patches)
        output = {
            "action_loss": action_output["loss"] + self.fdm_loss_weight * fdm_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }
        output.update({key: value.detach() for key, value in metrics.items()})
        return output

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if size:
            batch_images = resize_images(batch_images, target_size=size)
        instructions = [example["lang"] for example in examples]
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        condition, mask, _current, _patches = self._build_fdm_action_condition(batch_images, instructions)
        state_tensor = torch.from_numpy(np.array(state)).to(condition.device, dtype=condition.dtype) if state is not None else None
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(condition, state_tensor, encoder_attention_mask=mask)
        return {"normalized_actions": pred_actions.detach().float().cpu().numpy()}
