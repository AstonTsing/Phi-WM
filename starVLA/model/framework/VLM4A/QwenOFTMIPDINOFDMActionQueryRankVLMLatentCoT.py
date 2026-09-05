# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 2.0 (the "License");
"""Qwen-internal future-visual latent CoT before ActionQueryRank MIP-FDM."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDMActionQueryRank import (
    QwenOFTMIPDINOFDMActionQueryRankDefaultConfig,
    Qwen_OFT_MIP_DINO_FDM_ActionQueryRank,
    _cfg_get,
)
from starVLA.model.framework.VLM4A.QwenOFTMIPDINOFDM import _cfg_bool
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images


@dataclass
class QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoTDefaultConfig(
    QwenOFTMIPDINOFDMActionQueryRankDefaultConfig
):
    name: str = "QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoT"
    latent_cot: dict = field(
        default_factory=lambda: {
            "enabled": True,
            "loss_weight": 0.1,
            "num_latents": 16,
            "future_image_indices": [0, 1],
            "action_gradient_to_latent": True,
            "inference_use_cache": True,
        }
    )


@FRAMEWORK_REGISTRY.register("QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoT")
class Qwen_OFT_MIP_DINO_FDM_ActionQueryRank_VLM_LatentCoT(
    Qwen_OFT_MIP_DINO_FDM_ActionQueryRank
):
    SPECIAL_TOKENS = {
        "language_start": "<|language_start|>",
        "language_end": "<|language_end|>",
        "latent_start": "<|future_visual_start|>",
        "latent_pad": "<|future_visual_pad|>",
        "latent_end": "<|future_visual_end|>",
        "action_start": "<|action_start|>",
        "action_end": "<|action_end|>",
    }

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        config = merge_framework_config(
            QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoTDefaultConfig, config
        )
        super().__init__(config=config, **kwargs)
        self.config.framework.name = "QwenOFTMIPDINOFDMActionQueryRankVLMLatentCoT"
        self.latent_cot_cfg = self.config.framework.get("latent_cot", {})
        if not _cfg_bool(self.latent_cot_cfg, "enabled", True):
            raise ValueError("VLMLatentCoT requires latent_cot.enabled=true")
        self.latent_cot_loss_weight = float(_cfg_get(self.latent_cot_cfg, "loss_weight", 0.1))
        self.latent_cot_num_latents = int(_cfg_get(self.latent_cot_cfg, "num_latents", 16))
        self.latent_future_image_indices = [
            int(index) for index in _cfg_get(self.latent_cot_cfg, "future_image_indices", [0, 1])
        ]
        self.action_gradient_to_latent = _cfg_bool(
            self.latent_cot_cfg, "action_gradient_to_latent", True
        )
        self.inference_use_cache = _cfg_bool(self.latent_cot_cfg, "inference_use_cache", True)
        if not self.latent_future_image_indices:
            raise ValueError("latent_cot.future_image_indices cannot be empty")
        if self.latent_cot_num_latents % len(self.latent_future_image_indices):
            raise ValueError("num_latents must be divisible by the number of selected future images")
        self.latent_tokens_per_image = self.latent_cot_num_latents // len(self.latent_future_image_indices)
        self._register_latent_special_tokens()
        if _cfg_bool(self.config.trainer, "enable_gradient_checkpointing", False):
            self._qwen.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    @property
    def _qwen(self):
        return self.qwen_vl_interface.model

    def _register_latent_special_tokens(self) -> None:
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        old_vocab = set(tokenizer.get_vocab())
        tokenizer.add_special_tokens({"additional_special_tokens": list(self.SPECIAL_TOKENS.values())})
        if len(tokenizer) > self._qwen.get_input_embeddings().num_embeddings:
            self._qwen.resize_token_embeddings(len(tokenizer))

        config = self._qwen.config
        source_ids = {
            "language_start": int(config.vision_start_token_id),
            "language_end": int(config.vision_end_token_id),
            "latent_start": int(config.vision_start_token_id),
            "latent_pad": int(config.image_token_id),
            "latent_end": int(config.vision_end_token_id),
            "action_start": int(config.vision_start_token_id),
            "action_end": int(config.vision_end_token_id),
        }
        embeddings = self._qwen.get_input_embeddings().weight
        with torch.no_grad():
            for name, token in self.SPECIAL_TOKENS.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token not in old_vocab:
                    embeddings[token_id].copy_(embeddings[source_ids[name]])
                ids = tokenizer(token, add_special_tokens=False)["input_ids"]
                if ids != [token_id]:
                    raise RuntimeError(f"Special token {token} is not atomic: {ids}")
                setattr(self, f"{name}_token_id", int(token_id))

    def _format_instruction(self, instruction: str) -> str:
        prompt = self.config.datasets.vla_data.get("CoT_prompt", None)
        return str(prompt).replace("{instruction}", instruction) if prompt else instruction

    def _build_latent_qwen_inputs(self, images, instructions):
        latent_block = (
            self.SPECIAL_TOKENS["latent_start"]
            + self.SPECIAL_TOKENS["latent_pad"] * self.latent_cot_num_latents
            + self.SPECIAL_TOKENS["latent_end"]
        )
        action_block = (
            self.SPECIAL_TOKENS["action_start"]
            + self.action_token * self.action_horizon
            + self.SPECIAL_TOKENS["action_end"]
        )
        messages = []
        for sample_images, instruction in zip(images, instructions):
            sample_images = sample_images if isinstance(sample_images, (list, tuple)) else [sample_images]
            content = [{"type": "image", "image": image} for image in sample_images]
            language = (
                self.SPECIAL_TOKENS["language_start"]
                + self._format_instruction(instruction)
                + self.SPECIAL_TOKENS["language_end"]
            )
            content.append({"type": "text", "text": language})
            messages.append(
                [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": [{"type": "text", "text": latent_block + action_block}]},
                ]
            )
        inputs = self.qwen_vl_interface.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        return inputs.to(self._qwen.device)

    def _prepare_qwen_embeddings(self, inputs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"].to(dtype=torch.bool)
        embeddings = self._qwen.get_input_embeddings()(input_ids)
        image_grid = inputs.get("image_grid_thw")
        pixel_values = inputs.get("pixel_values")
        visual_mask = None
        deepstack = None
        if pixel_values is not None:
            image_features, deepstack = self._qwen.get_image_features(pixel_values, image_grid)
            image_features = torch.cat(image_features, dim=0).to(embeddings)
            visual_mask = input_ids == int(self._qwen.config.image_token_id)
            if visual_mask.sum().item() != image_features.shape[0]:
                raise ValueError(
                    f"Qwen image token mismatch: mask={visual_mask.sum().item()} features={image_features.shape[0]}"
                )
            embeddings = embeddings.masked_scatter(
                visual_mask.unsqueeze(-1).expand_as(embeddings), image_features
            )
        position_ids, _rope_delta = self._qwen.model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid,
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=attention_mask,
        )
        return input_ids, attention_mask, embeddings, position_ids, visual_mask, deepstack

    def _run_qwen_text(
        self,
        embeddings,
        attention_mask,
        position_ids,
        *,
        visual_mask=None,
        deepstack=None,
        past_key_values=None,
        cache_position=None,
        use_cache=False,
    ):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=embeddings.is_cuda):
            return self._qwen.language_model(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_mask,
                deepstack_visual_embeds=deepstack,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=use_cache,
                return_dict=True,
            )

    @staticmethod
    def _ordered_positions(mask: torch.Tensor, count: int, name: str) -> torch.Tensor:
        counts = mask.sum(dim=1)
        if not torch.equal(counts, torch.full_like(counts, count)):
            raise RuntimeError(f"Expected {count} {name} tokens per sample, got {counts.tolist()}")
        positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0).expand_as(mask)
        positions = torch.where(mask, positions, torch.full_like(positions, mask.shape[1]))
        return positions.topk(k=count, dim=-1, largest=False).values.sort(dim=-1).values

    @staticmethod
    def _gather_hidden(hidden: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return hidden.gather(1, positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))

    @staticmethod
    def _pool_shape(token_count: int, height: int, width: int):
        candidates = [(h, token_count // h) for h in range(1, token_count + 1) if token_count % h == 0]
        target_ratio = float(width) / max(float(height), 1.0)
        return min(candidates, key=lambda shape: abs(float(shape[1]) / shape[0] - target_ratio))

    def _pool_future_image_feature(self, feature, grid, token_count):
        merge = int(self._qwen.config.vision_config.spatial_merge_size)
        time, height, width = [int(value) for value in grid.tolist()]
        height, width = height // merge, width // merge
        if feature.shape[0] != time * height * width:
            raise ValueError(f"Future Qwen feature/grid mismatch: {tuple(feature.shape)} vs {(time, height, width)}")
        feature = feature.reshape(time, height, width, -1).mean(dim=0).permute(2, 0, 1).unsqueeze(0)
        pooled_h, pooled_w = self._pool_shape(token_count, height, width)
        return F.adaptive_avg_pool2d(feature.float(), (pooled_h, pooled_w)).flatten(2).transpose(1, 2)[0]

    def _encode_future_qwen_targets(self, future_images, device, dtype):
        selected = []
        for images in future_images:
            images = images if isinstance(images, (list, tuple)) else [images]
            if max(self.latent_future_image_indices) >= len(images):
                raise ValueError(
                    f"future_image_indices={self.latent_future_image_indices} exceed {len(images)} future images"
                )
            selected.append([images[index] for index in self.latent_future_image_indices])
        inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=selected, instructions=[""] * len(selected)
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            features, _deepstack = self._qwen.get_image_features(
                inputs["pixel_values"], inputs["image_grid_thw"]
            )
        grids = inputs["image_grid_thw"]
        per_sample, cursor = [], 0
        for _sample in selected:
            sample_tokens = []
            for _image in _sample:
                sample_tokens.append(
                    self._pool_future_image_feature(
                        features[cursor].detach(), grids[cursor], self.latent_tokens_per_image
                    )
                )
                cursor += 1
            per_sample.append(torch.cat(sample_tokens, dim=0))
        targets = torch.stack(per_sample).to(device=device, dtype=dtype)
        if targets.shape[1:] != (self.latent_cot_num_latents, self.qwen_hidden_size):
            raise ValueError(f"Unexpected future Qwen targets {tuple(targets.shape)}")
        return targets.detach()

    def _encode_qwen_latent_training(self, images, instructions, future_images):
        inputs = self._build_latent_qwen_inputs(images, instructions)
        input_ids, mask, base_embeddings, position_ids, visual_mask, deepstack = (
            self._prepare_qwen_embeddings(inputs)
        )
        pad_positions = self._ordered_positions(
            input_ids == self.latent_pad_token_id, self.latent_cot_num_latents, "future visual pad"
        )
        start_positions = self._ordered_positions(
            input_ids == self.latent_start_token_id, 1, "future visual start"
        )
        targets = self._encode_future_qwen_targets(
            future_images, base_embeddings.device, base_embeddings.dtype
        )

        teacher_embeddings = base_embeddings.clone()
        teacher_embeddings[input_ids == self.latent_pad_token_id] = targets.reshape(-1, targets.shape[-1])
        teacher_output = self._run_qwen_text(
            teacher_embeddings,
            mask,
            position_ids,
            visual_mask=visual_mask,
            deepstack=deepstack,
        )
        prediction_positions = torch.cat([start_positions, pad_positions[:, :-1]], dim=1)
        inferred = self._gather_hidden(teacher_output.last_hidden_state, prediction_positions)
        latent_loss = 1.0 - F.cosine_similarity(inferred.float(), targets.float(), dim=-1).mean()

        action_embeddings = base_embeddings.clone()
        action_latents = inferred if self.action_gradient_to_latent else inferred.detach()
        action_embeddings[input_ids == self.latent_pad_token_id] = action_latents.reshape(
            -1, action_latents.shape[-1]
        )
        action_output = self._run_qwen_text(
            action_embeddings,
            mask,
            position_ids,
            visual_mask=visual_mask,
            deepstack=deepstack,
        )
        action_queries = self._gather_action_queries(action_output.last_hidden_state, input_ids)
        metrics = {
            "latent_cot_loss": latent_loss.detach(),
            "latent_cot_cosine": (1.0 - latent_loss).detach(),
            "latent_pred_norm": inferred.float().norm(dim=-1).mean().detach(),
            "latent_target_norm": targets.float().norm(dim=-1).mean().detach(),
        }
        return action_output.last_hidden_state, mask, action_queries, latent_loss, metrics

    def _autoregressive_qwen_full(self, images, instructions):
        inputs = self._build_latent_qwen_inputs(images, instructions)
        input_ids, mask, embeddings, position_ids, visual_mask, deepstack = self._prepare_qwen_embeddings(inputs)
        pads = self._ordered_positions(
            input_ids == self.latent_pad_token_id, self.latent_cot_num_latents, "future visual pad"
        )
        starts = self._ordered_positions(input_ids == self.latent_start_token_id, 1, "future visual start")
        generated = []
        for index in range(self.latent_cot_num_latents):
            output = self._run_qwen_text(
                embeddings, mask, position_ids, visual_mask=visual_mask, deepstack=deepstack
            )
            source = starts if index == 0 else pads[:, index - 1 : index]
            latent = self._gather_hidden(output.last_hidden_state, source)[:, 0]
            generated.append(latent)
            batch = torch.arange(input_ids.shape[0], device=input_ids.device)
            embeddings[batch, pads[:, index]] = latent
        final = self._run_qwen_text(
            embeddings, mask, position_ids, visual_mask=visual_mask, deepstack=deepstack
        ).last_hidden_state
        return final, mask, input_ids, torch.stack(generated, dim=1)

    def _autoregressive_qwen_cached(self, images, instructions):
        inputs = self._build_latent_qwen_inputs(images, instructions)
        input_ids, mask, embeddings, position_ids, visual_mask, deepstack = self._prepare_qwen_embeddings(inputs)
        pads = self._ordered_positions(
            input_ids == self.latent_pad_token_id, self.latent_cot_num_latents, "future visual pad"
        )
        starts = self._ordered_positions(input_ids == self.latent_start_token_id, 1, "future visual start")
        if not torch.equal(starts, starts[:1].expand_as(starts)) or not torch.equal(pads, pads[:1].expand_as(pads)):
            return self._autoregressive_qwen_full(images, instructions)
        prefix_end = int(starts[0, 0]) + 1
        prefill = self._run_qwen_text(
            embeddings,
            mask,
            position_ids,
            visual_mask=visual_mask,
            deepstack=deepstack,
            cache_position=torch.arange(input_ids.shape[1], device=embeddings.device),
            use_cache=True,
        )
        past = prefill.past_key_values
        past.crop(prefix_end)
        prefix_hidden = prefill.last_hidden_state[:, :prefix_end]
        predicted = prefill.last_hidden_state[:, prefix_end - 1]
        generated, latent_hidden = [], []
        for index in range(self.latent_cot_num_latents):
            generated.append(predicted)
            position = int(pads[0, index])
            step = self._run_qwen_text(
                predicted.unsqueeze(1),
                mask[:, : position + 1],
                position_ids[:, :, position : position + 1],
                past_key_values=past,
                cache_position=torch.tensor([position], device=embeddings.device),
                use_cache=True,
            )
            past = step.past_key_values
            latent_hidden.append(step.last_hidden_state)
            predicted = step.last_hidden_state[:, -1]
        suffix_start = int(pads[0, -1]) + 1
        suffix = self._run_qwen_text(
            embeddings[:, suffix_start:],
            mask,
            position_ids[:, :, suffix_start:],
            past_key_values=past,
            cache_position=torch.arange(suffix_start, input_ids.shape[1], device=embeddings.device),
            use_cache=True,
        )
        hidden = torch.cat([prefix_hidden, *latent_hidden, suffix.last_hidden_state], dim=1)
        if hidden.shape[:2] != input_ids.shape:
            raise RuntimeError(f"Cached Qwen hidden shape mismatch: {tuple(hidden.shape)} vs {tuple(input_ids.shape)}")
        return hidden, mask, input_ids, torch.stack(generated, dim=1)

    def _append_dino_condition(self, qwen_tokens, qwen_mask, images):
        current_dino, dino_condition, patches = self._encode_dino_raw_and_condition(
            images, qwen_tokens.device, qwen_tokens.dtype
        )
        if dino_condition is None:
            return qwen_tokens, qwen_mask, current_dino, patches
        dino_mask = torch.ones(dino_condition.shape[:2], device=dino_condition.device, dtype=torch.bool)
        return (
            torch.cat([qwen_tokens, dino_condition], dim=1),
            torch.cat([qwen_mask, dino_mask], dim=1),
            current_dino,
            patches,
        )

    def forward(self, examples: List[dict] = None, **kwargs) -> Dict[str, torch.Tensor]:
        batch_images = [example["image"] for example in examples]
        future_images = [example.get("future_image") for example in examples]
        if any(images is None or len(images) == 0 for images in future_images):
            raise KeyError("VLMLatentCoT requires sample['future_image']")
        instructions = [example["lang"] for example in examples]
        current_images = self._select_current_condition_images(batch_images)
        qwen_tokens, qwen_mask, action_queries, latent_loss, latent_metrics = (
            self._encode_qwen_latent_training(current_images, instructions, future_images)
        )
        condition, condition_mask, current_dino, patches = self._append_dino_condition(
            qwen_tokens, qwen_mask, current_images
        )
        future_dino, future_patches = self._encode_future_dino_tokens_for_fdm(
            future_images, condition.device, condition.dtype
        )
        if not torch.equal(patches, future_patches):
            raise ValueError("current and future DINO patch indices must match")

        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        actions = torch.tensor(
            np.array([example["action"] for example in examples]),
            device=condition.device,
            dtype=condition.dtype,
        )[:, -self.action_horizon :]
        state_tensor = (
            torch.tensor(np.array(state), device=condition.device, dtype=condition.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            action_output = self.action_model(
                condition, actions, state_tensor, encoder_attention_mask=condition_mask
            )
            head_dtype = next(self.action_query_head.parameters()).dtype
            query_actions = self.action_query_head(action_queries.to(dtype=head_dtype))
            query_l1 = F.l1_loss(query_actions.float(), actions.float())

        base_fdm_loss, metrics = self._compute_fdm_loss(
            action_output, current_dino, future_dino, patches
        )
        query_rank_loss, query_metrics = self._compute_action_query_rank_loss(
            action_output, query_actions, current_dino, future_dino, patches
        )
        fdm_loss = base_fdm_loss + self.action_query_rank_weight * query_rank_loss
        total = (
            action_output["loss"]
            + self.action_query_loss_weight * query_l1.to(action_output["loss"].dtype)
            + self.latent_cot_loss_weight * latent_loss.to(action_output["loss"].dtype)
            + self.fdm_loss_weight * fdm_loss
        )
        metrics.update(query_metrics)
        metrics.update(latent_metrics)
        metrics.update({"fdm_loss_base": base_fdm_loss, "fdm_loss": fdm_loss})
        output = {
            "action_loss": total,
            "raw_action_loss": action_output["action_loss"].detach(),
            "action_query_l1": query_l1.detach(),
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
        current_images = self._select_current_condition_images(batch_images)
        instructions = [example["lang"] for example in examples]
        encode = self._autoregressive_qwen_cached if self.inference_use_cache else self._autoregressive_qwen_full
        qwen_tokens, qwen_mask, input_ids, _latents = encode(current_images, instructions)
        action_queries = self._gather_action_queries(qwen_tokens, input_ids)
        condition, condition_mask, _current, _patches = self._append_dino_condition(
            qwen_tokens, qwen_mask, current_images
        )
        use_state = getattr(self.action_model, "state_encoder", None) is not None
        state = [example["state"] for example in examples] if use_state and "state" in examples[0] else None
        state_tensor = (
            torch.from_numpy(np.array(state)).to(condition.device, dtype=condition.dtype)
            if state is not None
            else None
        )
        with torch.autocast("cuda", dtype=torch.float32):
            actions = self.action_model.predict_action(
                condition, state_tensor, encoder_attention_mask=condition_mask
            )
        return {"normalized_actions": actions.detach().float().cpu().numpy()}
