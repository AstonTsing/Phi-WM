# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Cosmos-Predict2 World Model Interface.

Wraps NVIDIA Cosmos-Predict2 (diffusion-based Video2World model) as a
world-model backend for starVLA action prediction frameworks.

Architecture:
  - T5EncoderModel: text instruction → text embeddings [B, L_text, 1024]
  - AutoencoderKLWan (VAE): observation images → video latents [B, C, T, H, W]
  - CosmosTransformer3DModel (DiT): 28-layer transformer, hidden_dim=4096
    Takes noised latents + text embeddings → denoised latents
    We extract intermediate hidden states for action-conditioning.

Key difference from VLM wrappers:
  - No chat template / processor — uses T5 for text, VAE for vision
  - Hidden states come from DiT blocks, not autoregressive LM
  - The `build_inputs` interface provides a clean world-model API
    that does not depend on VLM-specific naming conventions.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


class _CosmoPredict2_Interface(nn.Module):
    """
    World model wrapper for Cosmos-Predict2 (diffusers-based).

    Exposes a compatible interface with VLM wrappers so that framework
    code can swap VLM ↔ WM transparently. The key methods are:
      - forward(**kwargs) → model outputs with hidden_states
      - build_inputs(images, instructions) → dict of tensors
      - generate(**kwargs) → video generation (optional)

    Representation extraction strategy:
      We run a single DiT forward pass at noise level σ≈0 and register
      forward hooks to capture intermediate block outputs. These are
      concatenated/pooled to produce a [B, N_tokens, hidden_dim] tensor
      that the action head can consume — analogous to VLM hidden_states.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__()

        wm_cfg = config.framework.get("world_model", {})
        model_name = wm_cfg.get(
            "base_wm",
            config.framework.get("qwenvl", {}).get("base_vlm", "nvidia/Cosmos-Predict2-2B-Video2World"),
        )
        self.config = config

        # Import diffusers components
        from diffusers import (
            AutoencoderKLWan,
            CosmosTransformer3DModel,
            FlowMatchEulerDiscreteScheduler,
        )
        from transformers import T5EncoderModel, T5TokenizerFast

        logger.info(f"Loading Cosmos-Predict2 from {model_name}")

        # Load components individually (Pipeline is not nn.Module; split loading enables per-component freeze/finetune)
        self.tokenizer = T5TokenizerFast.from_pretrained(
            model_name, subfolder="tokenizer"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            model_name, subfolder="text_encoder", torch_dtype=torch.bfloat16
        )
        self.transformer = CosmosTransformer3DModel.from_pretrained(
            model_name, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        enable_grad_ckpt = bool(getattr(getattr(config, "trainer", None), "enable_gradient_checkpointing", False))
        if enable_grad_ckpt:
            if hasattr(self.transformer, "enable_gradient_checkpointing"):
                self.transformer.enable_gradient_checkpointing()
            else:
                self.transformer.gradient_checkpointing = True
            logger.info("Enabled Cosmos transformer gradient checkpointing")
        self.vae = AutoencoderKLWan.from_pretrained(
            model_name, subfolder="vae", torch_dtype=torch.bfloat16
        )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_name, subfolder="scheduler"
        )

        # Use diffusers' VideoProcessor for image/video preprocessing (resize, normalize, etc.)
        from diffusers.video_processor import VideoProcessor
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample)
        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample)
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

        obs_size = wm_cfg.get("vae_resolution", None)
        if obs_size is None:
            obs_size = getattr(getattr(getattr(config, "datasets", None), "vla_data", None), "obs_image_size", None)
        if obs_size is None:
            self._vae_height, self._vae_width = 192, 320
        elif isinstance(obs_size, int):
            self._vae_height = self._vae_width = int(obs_size)
        else:  # list / tuple / OmegaConf ListConfig — all support indexing
            self._vae_height, self._vae_width = int(obs_size[0]), int(obs_size[1])
        logger.info(f"Cosmos VAE encode resolution: {self._vae_height}x{self._vae_width}")

        # Freeze VAE and text encoder by default
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        # Expose config compatible with framework expectations
        # DiT: 16 heads × 128 dim = 2048
        self._hidden_size = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim

        # Create a config-like object for the framework to read hidden_size
        class _FakeConfig:
            pass

        self._model_config = _FakeConfig()
        self._model_config.hidden_size = self._hidden_size

        # Hook storage for intermediate features
        self._intermediate_features = []
        self._capture_intermediate_features = False
        self._hooks = []

        # Which transformer blocks to extract features from (-1 = last)
        extract_layers = wm_cfg.get("extract_layers", [-1])
        self._extract_layers = extract_layers
        self._install_asymmetric_self_attention_support()
        self._register_hooks()

    @property
    def model(self):
        """Compatibility shim: framework code accesses self.qwen_vl_interface.model.config.hidden_size"""
        class _ModelShim:
            pass
        shim = _ModelShim()
        shim.config = self._model_config
        return shim

    def _register_hooks(self):
        """Register forward hooks on selected transformer blocks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

        num_blocks = len(self.transformer.transformer_blocks)
        for layer_idx in self._extract_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_blocks + layer_idx
            if 0 <= actual_idx < num_blocks:
                block = self.transformer.transformer_blocks[actual_idx]
                hook = block.register_forward_hook(self._capture_hook)
                self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        """Capture intermediate transformer block output only during the main forward pass."""
        if not self._capture_intermediate_features:
            return
        # DiT block output is a tuple; first element is hidden_states
        if isinstance(output, tuple):
            self._intermediate_features.append(output[0])
        else:
            self._intermediate_features.append(output)

    def _install_asymmetric_self_attention_support(self) -> None:
        """Allow train-time condition tokens to ignore future tokens.

        Diffusers' Cosmos block supports masks in the attention processor, but
        the stock block does not pass a mask into self-attention.  Storing the
        mask on each block keeps it visible during gradient-checkpoint
        recomputation.
        """
        for block in self.transformer.transformer_blocks:
            if getattr(block, "_starvla_self_mask_patched", False):
                continue

            def forward(
                block_self,
                hidden_states,
                encoder_hidden_states,
                embedded_timestep,
                temb=None,
                image_rotary_emb=None,
                extra_pos_emb=None,
                attention_mask=None,
            ):
                if extra_pos_emb is not None:
                    hidden_states = hidden_states + extra_pos_emb

                norm_hidden_states, gate = block_self.norm1(hidden_states, embedded_timestep, temb)
                self_attention_mask = getattr(block_self, "_starvla_self_attention_mask", None)
                attn_output = block_self.attn1(
                    norm_hidden_states,
                    attention_mask=self_attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )
                hidden_states = hidden_states + gate * attn_output

                norm_hidden_states, gate = block_self.norm2(hidden_states, embedded_timestep, temb)
                attn_output = block_self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
                hidden_states = hidden_states + gate * attn_output

                norm_hidden_states, gate = block_self.norm3(hidden_states, embedded_timestep, temb)
                ff_output = block_self.ff(norm_hidden_states)
                hidden_states = hidden_states + gate * ff_output
                return hidden_states

            block.forward = forward.__get__(block, block.__class__)
            block._starvla_self_attention_mask = None
            block._starvla_self_mask_patched = True

    def _set_cosmos_self_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        for block in self.transformer.transformer_blocks:
            if getattr(block, "_starvla_self_mask_patched", False):
                block._starvla_self_attention_mask = attention_mask

    def _build_condition_to_future_attention_mask(
        self,
        condition_latent_frames: int,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, int]:
        p_t, p_h, p_w = self.transformer.config.patch_size
        if condition_latent_frames % p_t != 0 or num_latent_frames % p_t != 0:
            raise ValueError(
                "Latent frame counts must be divisible by temporal patch size: "
                f"condition={condition_latent_frames}, total={num_latent_frames}, patch={p_t}"
            )

        post_t = num_latent_frames // p_t
        post_h = latent_height // p_h
        post_w = latent_width // p_w
        condition_token_frames = condition_latent_frames // p_t
        condition_tokens = condition_token_frames * post_h * post_w
        total_tokens = post_t * post_h * post_w
        if condition_tokens <= 0 or condition_tokens >= total_tokens:
            raise ValueError(
                f"Expected both condition and future tokens, got condition={condition_tokens}, total={total_tokens}"
            )

        mask = torch.ones(1, 1, total_tokens, total_tokens, device=device, dtype=torch.bool)
        mask[:, :, :condition_tokens, condition_tokens:] = False
        return mask, condition_tokens

    def _encode_text(self, instructions, max_length=512):
        """Encode text instructions using T5."""
        device = next(self.text_encoder.parameters()).device
        text_inputs = self.tokenizer(
            instructions,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            ).last_hidden_state  # [B, L, 1024]

        return text_embeds, text_inputs.attention_mask

    def _encode_images(self, images, num_frames=None):
        """Encode observation images through VAE to get latent tokens.

        Follows the Cosmos pipeline approach: pad images to uniform frame count
        with last-frame repetition, then VAE-encode the whole video.

        Args:
            images: List of List of PIL Images [B, [imgs...]]
            num_frames: If given, pad/truncate to this exact count.
                If None (default), pad to the max frame count in the batch.
                VAE temporal factor is 4, so T_latent = (num_frames-1)//4+1.

        Returns:
            latents: [B, C, T_latent, H/8, W/8] video latent tensor
            cond_frame_counts: list[int], real frame count per sample (before padding)
        """
        device = next(self.vae.parameters()).device
        dtype = self.vae.dtype
        # 480×832 is the pretrained resolution; spatial dims must be multiples of 16.
        # RoPE encoding: resolution changes are safe without reloading weights.
        # 320×576 (native) vs 224×384 (faster): tokens per latent frame 720 → 336,
        # attention cost ~0.22× of native. Both are within transformer max_size=[128,240,240].
        # height, width = 320, 576
        # height, width = 256, 448
        # height, width = 224, 384
        # height, width = 192, 320
        height, width = self._vae_height, self._vae_width

        # First pass: preprocess each sample, record real frame counts
        preprocessed = []
        cond_frame_counts = []
        for sample_imgs in images:
            if not isinstance(sample_imgs, (list, tuple)):
                sample_imgs = [sample_imgs]

            video_tensor = self.video_processor.preprocess_video(sample_imgs, height=height, width=width)
            video_tensor = video_tensor.to(device=device, dtype=dtype)  # [1, C, n_imgs, H, W]
            preprocessed.append(video_tensor)
            cond_frame_counts.append(video_tensor.shape[2])

        # Determine target frame count: use num_frames if specified, otherwise batch max
        # Ensure at least 1 frame (VAE needs T >= 1)
        if num_frames is None:
            target_frames = max(cond_frame_counts)
        else:
            target_frames = num_frames

        # Second pass: truncate or pad each sample to target_frames
        batch_videos = []
        for i, video_tensor in enumerate(preprocessed):
            n = video_tensor.shape[2]
            if n > target_frames:
                video_tensor = video_tensor[:, :, :target_frames]
                cond_frame_counts[i] = target_frames
            elif n < target_frames:
                # Pad with last-frame repetition (matches official pipeline)
                last_frame = video_tensor[:, :, -1:]
                padding = last_frame.repeat(1, 1, target_frames - n, 1, 1)
                video_tensor = torch.cat([video_tensor, padding], dim=2)
            batch_videos.append(video_tensor.squeeze(0))  # [C, target_frames, H, W]

        # Stack to [B, C, target_frames, H, W]
        video = torch.stack(batch_videos, dim=0)

        with torch.no_grad():
            # T_latent = temporal latent frames = (num_frames-1)//4+1  (VAE temporal downsample factor=4)
            # e.g. 5 frames → 2 latent frames, 9 frames → 3 latent frames
            latents = self.vae.encode(video).latent_dist.sample()  # [B, 16, T_latent, H/8, W/8]

        # Normalize latents (matches official pipeline: prepare_latents) # TODO check if this normalization is actually needed
        if self.vae.config.latents_mean is not None:
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, dtype=latents.dtype)
            ) 
            sigma_data = self.scheduler.config.sigma_data
            latents = (latents - latents_mean) / latents_std * sigma_data

        return latents, cond_frame_counts # latents: [B, C, T_latent, H/8, W/8], cond_frame_counts: list[int]

    def build_inputs(self, images, instructions, **kwargs):
        """Build inputs for the DiT world model.

        Instead of chat templates (VLM), we:
        1. Encode text with T5
        2. Encode images with VAE
        3. Package for the DiT forward pass

        Returns:
            dict with keys matching what forward() expects
        """
        assert len(images) == len(instructions)

        # Ensure encoders are on the right device
        device = next(self.transformer.parameters()).device
        # self.text_encoder.to(device)
        # self.vae.to(device)

        text_embeds, text_mask = self._encode_text(instructions)
        latents, cond_frame_counts = self._encode_images(images)

        # Offload T5 and VAE to CPU to free VRAM for the transformer
        # self.text_encoder.to("cpu")
        # self.vae.to("cpu")
        # torch.cuda.empty_cache()

        # For feature extraction, use timestep=0 (clean / minimal noise)
        batch_size = latents.shape[0]
        device = latents.device
        _, _, t_lat, h_lat, w_lat = latents.shape # B, C, T_latent, H/8, W/8
        timestep = torch.zeros(batch_size, device=device, dtype=torch.long)
        # condition_mask: tells DiT which latent frames are reliable conditions (=1) vs to-be-generated (=0).
        # In Video2World, this separates input frames from predicted future frames.
        # Here (action prediction, not generation), we set timestep=0 + condition_mask on real frames
        # so DiT runs a near-clean forward pass for feature extraction, not actual denoising.
        # in_channels = 16 (latents) + 1 (condition_mask) = 17
        # Shape: [B, 1, T_latent, H_latent, W_latent]
        condition_mask = latents.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        for i, n_cond in enumerate(cond_frame_counts):
            # Map pixel-frame count to latent-frame count
            n_cond_latent = (n_cond - 1) // self.vae_scale_factor_temporal + 1
            condition_mask[i, :, :n_cond_latent] = 1.0

        # padding_mask: concat_padding_mask=True adds 1 more channel → 18 total
        # Shape: [1, 1, H_orig, W_orig] — all zeros = no padding
        # Will be resized to latent spatial dims by the transformer
        padding_mask = latents.new_zeros(1, 1, h_lat, w_lat)

        return {
            "hidden_states": latents,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "_is_wm_input": True,
        }

    def build_video_loss_inputs(
        self,
        condition_images: List[List],
        future_images: List,
        instructions: List[str],
        num_frames: Optional[int] = None,
        sigma_conditioning: float = 1e-4,
    ):
        """Build one joint Cosmos pass for action features plus future-frame loss.

        The transformer sees clean condition latents and noised future latents.
        Future timesteps use t=sigma/(sigma+1), matching Cosmos-Predict2
        Video2World conditioning. The raw velocity target is noise - z_future.
        """
        if len(condition_images) != len(future_images) or len(condition_images) != len(instructions):
            raise ValueError(
                "condition_images, future_images, and instructions must have the same batch length"
            )
        if any(image is None for image in future_images):
            raise KeyError("video loss training requires future_images for every sample")

        normalized_condition_images = []
        condition_frame_counts = []
        for sample_images in condition_images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            sample_images = list(sample_images)
            if not sample_images:
                raise ValueError("Each sample needs at least one condition image")
            normalized_condition_images.append(sample_images)
            condition_frame_counts.append(len(sample_images))

        condition_latent_counts = [
            (count - 1) // self.vae_scale_factor_temporal + 1 for count in condition_frame_counts
        ]
        if len(set(condition_latent_counts)) != 1:
            raise ValueError(
                "video loss currently expects a uniform number of condition latent frames per batch; "
                f"got {condition_latent_counts}"
            )

        video_images = [
            list(sample_images) + [future_image]
            for sample_images, future_image in zip(normalized_condition_images, future_images)
        ]
        text_embeds, text_mask = self._encode_text(instructions)
        clean_latents, _ = self._encode_images(video_images, num_frames=num_frames)

        condition_latent_frames = condition_latent_counts[0]
        if clean_latents.shape[2] <= condition_latent_frames:
            raise ValueError(
                "Encoded video does not contain a separate future latent frame: "
                f"latents={tuple(clean_latents.shape)}, condition_latent_frames={condition_latent_frames}"
            )

        z_cond = clean_latents[:, :, :condition_latent_frames]
        z_future = clean_latents[:, :, condition_latent_frames:]
        noise = torch.randn_like(z_future)
        batch_size = clean_latents.shape[0]
        sigma = torch.exp(torch.randn(batch_size, device=clean_latents.device, dtype=torch.float32)).to(
            dtype=clean_latents.dtype
        )
        t = (sigma / (sigma + 1.0)).view(batch_size, 1, 1, 1, 1)
        z_noisy_future = (1.0 - t) * z_future + t * noise
        target_velocity = noise - z_future
        hidden_states = torch.cat([z_cond, z_noisy_future], dim=2)

        _, _, t_lat, h_lat, w_lat = hidden_states.shape
        condition_mask = hidden_states.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        condition_mask[:, :, :condition_latent_frames] = 1.0
        padding_mask = hidden_states.new_zeros(1, 1, h_lat, w_lat)

        timestep = hidden_states.new_full(
            (batch_size, 1, t_lat, 1, 1),
            float(sigma_conditioning) / (float(sigma_conditioning) + 1.0),
        )
        timestep[:, :, condition_latent_frames:] = t.to(dtype=hidden_states.dtype)
        self_attention_mask, condition_tokens = self._build_condition_to_future_attention_mask(
            condition_latent_frames=condition_latent_frames,
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            device=hidden_states.device,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "video_loss_target": target_velocity,
            "video_loss_condition_latent_frames": condition_latent_frames,
            "video_loss_condition_tokens": condition_tokens,
            "video_loss_self_attention_mask": self_attention_mask,
            "_is_wm_video_loss_input": True,
        }


    def build_rollout_video_loss_inputs(
        self,
        condition_images: List[List],
        future_images: List,
        instructions: List[str],
        num_frames: Optional[int] = None,
        sigma_conditioning: float = 1e-4,
    ):
        """Build video-loss inputs for one condition frame plus future rollout frames.

        This keeps the original build_video_loss_inputs() contract untouched while
        allowing rollout experiments to pass future_images as a per-sample list.
        """
        if len(condition_images) != len(future_images) or len(condition_images) != len(instructions):
            raise ValueError(
                "condition_images, future_images, and instructions must have the same batch length"
            )
        if any(image is None for image in future_images):
            raise KeyError("rollout video loss training requires future_images for every sample")

        normalized_condition_images = []
        condition_frame_counts = []
        for sample_images in condition_images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            sample_images = list(sample_images)
            if not sample_images:
                raise ValueError("Each sample needs at least one condition image")
            normalized_condition_images.append(sample_images)
            condition_frame_counts.append(len(sample_images))

        normalized_future_images = []
        for sample_future_images in future_images:
            if not isinstance(sample_future_images, (list, tuple)):
                sample_future_images = [sample_future_images]
            sample_future_images = list(sample_future_images)
            if not sample_future_images:
                raise ValueError("Each sample needs at least one future image")
            normalized_future_images.append(sample_future_images)

        condition_latent_counts = [
            (count - 1) // self.vae_scale_factor_temporal + 1 for count in condition_frame_counts
        ]
        if len(set(condition_latent_counts)) != 1:
            raise ValueError(
                "rollout video loss currently expects a uniform number of condition latent frames per batch; "
                f"got {condition_latent_counts}"
            )

        video_images = [
            list(sample_images) + list(sample_future_images)
            for sample_images, sample_future_images in zip(normalized_condition_images, normalized_future_images)
        ]
        text_embeds, text_mask = self._encode_text(instructions)
        clean_latents, _ = self._encode_images(video_images, num_frames=num_frames)

        condition_latent_frames = condition_latent_counts[0]
        if clean_latents.shape[2] <= condition_latent_frames:
            raise ValueError(
                "Encoded rollout video does not contain separate future latent frames: "
                f"latents={tuple(clean_latents.shape)}, condition_latent_frames={condition_latent_frames}"
            )

        z_cond = clean_latents[:, :, :condition_latent_frames]
        z_future = clean_latents[:, :, condition_latent_frames:]
        noise = torch.randn_like(z_future)
        batch_size = clean_latents.shape[0]
        sigma = torch.exp(torch.randn(batch_size, device=clean_latents.device, dtype=torch.float32)).to(
            dtype=clean_latents.dtype
        )
        t = (sigma / (sigma + 1.0)).view(batch_size, 1, 1, 1, 1)
        z_noisy_future = (1.0 - t) * z_future + t * noise
        target_velocity = noise - z_future
        hidden_states = torch.cat([z_cond, z_noisy_future], dim=2)

        _, _, t_lat, h_lat, w_lat = hidden_states.shape
        condition_mask = hidden_states.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        condition_mask[:, :, :condition_latent_frames] = 1.0
        padding_mask = hidden_states.new_zeros(1, 1, h_lat, w_lat)

        timestep = hidden_states.new_full(
            (batch_size, 1, t_lat, 1, 1),
            float(sigma_conditioning) / (float(sigma_conditioning) + 1.0),
        )
        timestep[:, :, condition_latent_frames:] = t.to(dtype=hidden_states.dtype)
        self_attention_mask, condition_tokens = self._build_condition_to_future_attention_mask(
            condition_latent_frames=condition_latent_frames,
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            device=hidden_states.device,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "video_loss_target": target_velocity,
            "video_loss_condition_latent_frames": condition_latent_frames,
            "video_loss_condition_tokens": condition_tokens,
            "video_loss_self_attention_mask": self_attention_mask,
            "_is_wm_video_loss_input": True,
        }

    def build_noisy_future_inputs(
        self,
        condition_images: List[List],
        instructions: List[str],
        future_timestep: float = 0.999,
        sigma_conditioning: float = 1e-4,
        future_latent_frames: int = 1,
    ):
        """Build a train-like inference pass with a masked noisy future slot.

        This mirrors the action-feature side of build_video_loss_inputs() but
        does not require a real future image or compute a video loss. The
        transformer sees clean condition latents followed by random future
        latents; condition tokens are masked from attending to future tokens,
        and forward() returns only condition-token hidden states.
        """
        if len(condition_images) != len(instructions):
            raise ValueError("condition_images and instructions must have the same batch length")
        if future_latent_frames <= 0:
            raise ValueError(f"future_latent_frames must be positive, got {future_latent_frames}")
        if not 0.0 <= float(future_timestep) <= 1.0:
            raise ValueError(f"future_timestep must be in [0, 1], got {future_timestep}")

        normalized_condition_images = []
        condition_frame_counts = []
        for sample_images in condition_images:
            if not isinstance(sample_images, (list, tuple)):
                sample_images = [sample_images]
            sample_images = list(sample_images)
            if not sample_images:
                raise ValueError("Each sample needs at least one condition image")
            normalized_condition_images.append(sample_images)
            condition_frame_counts.append(len(sample_images))

        condition_latent_counts = [
            (count - 1) // self.vae_scale_factor_temporal + 1 for count in condition_frame_counts
        ]
        if len(set(condition_latent_counts)) != 1:
            raise ValueError(
                "noisy-future inference expects a uniform number of condition latent frames per batch; "
                f"got {condition_latent_counts}"
            )

        text_embeds, text_mask = self._encode_text(instructions)
        z_cond, _ = self._encode_images(normalized_condition_images)
        condition_latent_frames = condition_latent_counts[0]
        z_cond = z_cond[:, :, :condition_latent_frames]

        noise_future = torch.randn(
            z_cond.shape[0],
            z_cond.shape[1],
            future_latent_frames,
            z_cond.shape[3],
            z_cond.shape[4],
            device=z_cond.device,
            dtype=z_cond.dtype,
        )
        hidden_states = torch.cat([z_cond, noise_future], dim=2)

        batch_size, _, t_lat, h_lat, w_lat = hidden_states.shape
        condition_mask = hidden_states.new_zeros(batch_size, 1, t_lat, h_lat, w_lat)
        condition_mask[:, :, :condition_latent_frames] = 1.0
        padding_mask = hidden_states.new_zeros(1, 1, h_lat, w_lat)

        cond_t = float(sigma_conditioning) / (float(sigma_conditioning) + 1.0)
        timestep = hidden_states.new_full((batch_size, 1, t_lat, 1, 1), cond_t)
        timestep[:, :, condition_latent_frames:] = float(future_timestep)

        self_attention_mask, condition_tokens = self._build_condition_to_future_attention_mask(
            condition_latent_frames=condition_latent_frames,
            num_latent_frames=t_lat,
            latent_height=h_lat,
            latent_width=w_lat,
            device=hidden_states.device,
        )

        return {
            "hidden_states": hidden_states,
            "timestep": timestep,
            "encoder_hidden_states": text_embeds,
            "attention_mask": text_mask,
            "condition_mask": condition_mask,
            "padding_mask": padding_mask,
            "noisy_future_condition_tokens": condition_tokens,
            "noisy_future_self_attention_mask": self_attention_mask,
            "_is_wm_noisy_future_input": True,
        }

    def forward(self, **kwargs):
        """Forward pass through the DiT transformer.

        Runs a single-step forward to extract rich spatiotemporal features.
        Returns an output object with .hidden_states for compatibility.
        """
        is_wm = kwargs.pop("_is_wm_input", False)
        is_video_loss = kwargs.pop("_is_wm_video_loss_input", False)
        is_noisy_future = kwargs.pop("_is_wm_noisy_future_input", False)
        output_hidden_states = kwargs.pop("output_hidden_states", False)
        return_dict = kwargs.pop("return_dict", True)
        kwargs.pop("output_attentions", None)
        video_loss_target = kwargs.pop("video_loss_target", None)
        video_loss_condition_latent_frames = kwargs.pop("video_loss_condition_latent_frames", None)
        video_loss_condition_tokens = kwargs.pop("video_loss_condition_tokens", None)
        video_loss_self_attention_mask = kwargs.pop("video_loss_self_attention_mask", None)
        noisy_future_condition_tokens = kwargs.pop("noisy_future_condition_tokens", None)
        noisy_future_self_attention_mask = kwargs.pop("noisy_future_self_attention_mask", None)

        # Clear feature buffer
        self._intermediate_features.clear()

        self._capture_intermediate_features = True
        try:
            if is_video_loss:
                self._set_cosmos_self_attention_mask(video_loss_self_attention_mask)
            elif is_noisy_future:
                self._set_cosmos_self_attention_mask(noisy_future_self_attention_mask)
            else:
                self._set_cosmos_self_attention_mask(None)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dit_output = self.transformer(
                    hidden_states=kwargs["hidden_states"],
                    timestep=kwargs["timestep"],
                    encoder_hidden_states=kwargs["encoder_hidden_states"],
                    condition_mask=kwargs.get("condition_mask", None),
                    padding_mask=kwargs.get("padding_mask", None),
                )
        finally:
            self._capture_intermediate_features = False

        # Build hidden_states tuple from captured intermediate features
        # Also reshape from [B, C, T, H, W] to [B, N_tokens, hidden_dim]
        extracted = []
        for feat in self._intermediate_features:
            if feat.dim() == 5:
                # [B, C, T, H, W] -> [B, T*H*W, C]
                B, C, T, H, W = feat.shape
                feat = feat.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(feat)

        # If no hooks fired (shouldn't happen), use transformer output
        if not extracted:
            out = dit_output.sample if hasattr(dit_output, "sample") else dit_output
            if out.dim() == 5:
                B, C, T, H, W = out.shape
                out = out.permute(0, 2, 3, 4, 1).reshape(B, T * H * W, C)
            extracted.append(out)

        # Build compatible output object
        class _WMOutput:
            def __init__(self, hidden_states_tuple, loss=None, video_loss=None):
                self.hidden_states = hidden_states_tuple
                self.loss = loss
                self.video_loss = video_loss

        if is_video_loss:
            if video_loss_target is None or video_loss_condition_latent_frames is None:
                raise ValueError("Video loss forward requires target and condition frame metadata")
            if video_loss_condition_tokens is None:
                raise ValueError("Video loss forward requires condition token metadata")
            extracted = [feat[:, :video_loss_condition_tokens, :] for feat in extracted]
            pred_velocity = dit_output.sample[:, :, video_loss_condition_latent_frames:]
            video_loss = F.mse_loss(pred_velocity.float(), video_loss_target.float())
            return _WMOutput(hidden_states_tuple=tuple(extracted), loss=video_loss, video_loss=video_loss)

        if is_noisy_future:
            if noisy_future_condition_tokens is None:
                raise ValueError("Noisy-future forward requires condition token metadata")
            extracted = [feat[:, :noisy_future_condition_tokens, :] for feat in extracted]
            return _WMOutput(hidden_states_tuple=tuple(extracted))

        return _WMOutput(hidden_states_tuple=tuple(extracted))

    def generate(self, **kwargs):
        """Video generation (for world-model imagination / planning).

        This builds the full Cosmos2VideoToWorldPipeline on-the-fly.
        Not used during standard VLA training, but useful for visualization
        and planning-based approaches.
        """
        from diffusers import Cosmos2VideoToWorldPipeline

        pipe = Cosmos2VideoToWorldPipeline(
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            vae=self.vae,
            scheduler=self.scheduler,
            safety_checker=None,
        )
        return pipe(**kwargs)
