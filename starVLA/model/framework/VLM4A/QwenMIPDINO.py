# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Qwen-MIP-DINO Framework

QwenMIP with an additional frozen DINOv3 dense visual token stream. DINOv3
patch tokens from all input views are projected to Qwen hidden size and
concatenated to QwenVL tokens before the GR00T MIP action head.
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
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images


logger = initialize_overwatch(__name__)


def _to_hw(value):
    if isinstance(value, int):
        return int(value), int(value)
    if len(value) != 2:
        raise ValueError(f"Expected [height, width], got {value}")
    return int(value[0]), int(value[1])


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


@dataclass
class QwenMIPDINODefaultConfig:
    """QwenMIPDINO framework default parameters."""

    name: str = "QwenMIPDINO"

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
            "attn_implementation": "flash_attention_2",
            "vl_hidden_dim": 2048,
        }
    )

    dinov3: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "model_path": "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
            "freeze_dino": True,
            "input_size": [224, 224],
            "include_cls_token": False,
            "include_register_tokens": False,
            "normalize_tokens": False,
            "projector": {
                "layer_norm": True,
                "bias": True,
            },
        }
    )

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


@FRAMEWORK_REGISTRY.register("QwenMIPDINO")
class Qwen_MIP_DINO(baseframework):
    """QwenVL + frozen DINOv3 dense tokens + GR00T MIP action head."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = merge_framework_config(QwenMIPDINODefaultConfig, config)
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        self.qwen_hidden_size = int(self.qwen_vl_interface.model.config.hidden_size)
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.qwen_hidden_size
        self.action_model: GR00TMIPActionHead = get_action_model(config=self.config)
        self.action_horizon = int(self.config.framework.action_model.action_horizon)

        self.dino_cfg = self.config.framework.get("dinov3", {})
        self.dino_enabled = bool(_cfg_get(self.dino_cfg, "enabled", True))
        self.dino_input_hw = _to_hw(_cfg_get(self.dino_cfg, "input_size", [224, 224]))
        self.dino_include_cls = bool(_cfg_get(self.dino_cfg, "include_cls_token", False))
        self.dino_include_registers = bool(_cfg_get(self.dino_cfg, "include_register_tokens", False))
        self.dino_normalize_tokens = bool(_cfg_get(self.dino_cfg, "normalize_tokens", False))
        self.dino_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.dino_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self._init_dino()

    def _init_dino(self) -> None:
        if not self.dino_enabled:
            self.dino_model = None
            self.dino_projector = None
            return

        model_path = _cfg_get(
            self.dino_cfg,
            "model_path",
            "/root/tianyi/LDA-1B/playground/Pretrained_models/dinov3-vits16-pretrain-lvd1689m",
        )
        self.dino_model = DINOv3ViTModel.from_pretrained(str(model_path))
        if bool(_cfg_get(self.dino_cfg, "freeze_dino", True)):
            self.dino_model.eval()
            for param in self.dino_model.parameters():
                param.requires_grad = False

        dino_hidden_size = int(self.dino_model.config.hidden_size)
        projector_cfg = _cfg_get(self.dino_cfg, "projector", {})
        use_layer_norm = bool(_cfg_get(projector_cfg, "layer_norm", True))
        bias = bool(_cfg_get(projector_cfg, "bias", True))
        if use_layer_norm:
            self.dino_projector = nn.Sequential(
                nn.LayerNorm(dino_hidden_size),
                nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=bias),
            )
        else:
            self.dino_projector = nn.Linear(dino_hidden_size, self.qwen_hidden_size, bias=bias)

    def _preprocess_dino_flat_views(self, batch_images: List[List]) -> Tuple[torch.Tensor, List[int]]:
        height, width = self.dino_input_hw
        tensors = []
        view_counts = []
        for images in batch_images:
            if not isinstance(images, (list, tuple)):
                images = [images]
            view_counts.append(len(images))
            for image in images:
                image = to_pil_preserve(image).convert("RGB").resize((width, height))
                array = np.asarray(image, dtype=np.float32) / 255.0
                tensor = torch.from_numpy(array).permute(2, 0, 1)
                tensors.append(tensor)
        if len(set(view_counts)) != 1:
            raise ValueError(f"QwenMIPDINO expects the same number of views per sample, got {view_counts}")
        pixel_values = torch.stack(tensors, dim=0)
        mean = self.dino_mean.to(pixel_values)
        std = self.dino_std.to(pixel_values)
        return (pixel_values - mean) / std, view_counts

    def _select_dino_tokens(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        num_register_tokens = int(getattr(self.dino_model.config, "num_register_tokens", 0))
        pieces = []
        if self.dino_include_cls:
            pieces.append(last_hidden_state[:, :1, :])
        if self.dino_include_registers and num_register_tokens > 0:
            pieces.append(last_hidden_state[:, 1 : 1 + num_register_tokens, :])
        patch_start = 1 + num_register_tokens
        pieces.append(last_hidden_state[:, patch_start:, :])
        return torch.cat(pieces, dim=1)

    def _encode_dino_all_views(self, batch_images: List[List], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.dino_enabled:
            return None
        pixel_values, view_counts = self._preprocess_dino_flat_views(batch_images)
        pixel_values = pixel_values.to(device=device)
        num_views = view_counts[0]

        self.dino_model.eval()
        with torch.no_grad():
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=pixel_values.is_cuda):
                output = self.dino_model(pixel_values=pixel_values)

        dino_tokens = self._select_dino_tokens(output.last_hidden_state)
        if self.dino_normalize_tokens:
            dino_tokens = F.normalize(dino_tokens.float(), dim=-1).to(dtype=dino_tokens.dtype)

        batch_size = len(batch_images)
        dino_tokens = dino_tokens.reshape(batch_size, num_views * dino_tokens.shape[1], dino_tokens.shape[-1])
        projector_dtype = next(self.dino_projector.parameters()).dtype
        dino_tokens = self.dino_projector(dino_tokens.to(device=device, dtype=projector_dtype))
        return dino_tokens.to(dtype=dtype)

    def _encode_qwen(
        self,
        batch_images,
        instructions,
    ):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        backbone_attention_mask = qwen_inputs.get("attention_mask", None)
        if backbone_attention_mask is not None:
            backbone_attention_mask = backbone_attention_mask.to(dtype=torch.bool)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        return last_hidden, backbone_attention_mask

    def _build_action_condition(self, batch_images, instructions):
        qwen_tokens, qwen_mask = self._encode_qwen(batch_images, instructions)
        dino_tokens = self._encode_dino_all_views(batch_images, qwen_tokens.device, qwen_tokens.dtype)
        if dino_tokens is None:
            return qwen_tokens, qwen_mask

        condition_tokens = torch.cat([qwen_tokens, dino_tokens], dim=1)
        if qwen_mask is None:
            qwen_mask = torch.ones(
                qwen_tokens.shape[:2],
                device=qwen_tokens.device,
                dtype=torch.bool,
            )
        dino_mask = torch.ones(
            dino_tokens.shape[:2],
            device=dino_tokens.device,
            dtype=torch.bool,
        )
        condition_mask = torch.cat([qwen_mask, dino_mask], dim=1)
        return condition_tokens, condition_mask

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        condition_tokens, condition_mask = self._build_action_condition(batch_images, instructions)

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

        action_loss = action_output["loss"]
        return {
            "action_loss": action_loss,
            "raw_action_loss": action_output["action_loss"].detach(),
            "mip_action_loss0": action_output["mip_action_loss0"].detach(),
            "mip_action_loss1": action_output["mip_action_loss1"].detach(),
        }

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
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        condition_tokens, condition_mask = self._build_action_condition(batch_images, instructions)
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
