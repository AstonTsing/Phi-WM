from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import ViTConfig, ViTModel

from starVLA.model.modules.lewm.jepa import JEPA
from starVLA.model.modules.lewm.module import ARPredictor, Embedder, MLP, SIGReg


DEFAULT_VIT_TINY_PATH = "/root/tianyi/LDA-1B/playground/Pretrained_models/vit-tiny-patch16-224"


def build_vit_encoder(pretrained_vit_path: str | None = DEFAULT_VIT_TINY_PATH, image_size: int = 224):
    if pretrained_vit_path and Path(pretrained_vit_path).exists():
        return ViTModel.from_pretrained(pretrained_vit_path, add_pooling_layer=False)

    config = ViTConfig(
        image_size=image_size,
        patch_size=16,
        hidden_size=192,
        num_hidden_layers=12,
        num_attention_heads=3,
        intermediate_size=768,
    )
    return ViTModel(config, add_pooling_layer=False)


def build_lewm_model(
    image_size: int = 224,
    patch_size: int = 16,
    lewm_dim: int = 256,
    action_dim: int = 29,
    action_horizon: int = 16,
    pretrained_vit_path: str | None = DEFAULT_VIT_TINY_PATH,
    predictor_depth: int = 6,
    predictor_heads: int = 4,
    predictor_dim_head: int = 64,
    predictor_dropout: float = 0.1,
):
    del patch_size
    encoder = build_vit_encoder(pretrained_vit_path=pretrained_vit_path, image_size=image_size)
    encoder_dim = int(encoder.config.hidden_size)
    action_block_dim = int(action_dim) * int(action_horizon)

    model = JEPA(
        encoder=encoder,
        predictor=ARPredictor(
            num_frames=1,
            input_dim=lewm_dim,
            hidden_dim=lewm_dim,
            output_dim=lewm_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            mlp_dim=4 * lewm_dim,
            dim_head=predictor_dim_head,
            dropout=predictor_dropout,
            emb_dropout=0.0,
        ),
        action_encoder=Embedder(input_dim=action_block_dim, smoothed_dim=action_block_dim, emb_dim=lewm_dim),
        projector=MLP(input_dim=encoder_dim, hidden_dim=4 * lewm_dim, output_dim=lewm_dim),
        pred_proj=MLP(input_dim=lewm_dim, hidden_dim=4 * lewm_dim, output_dim=lewm_dim),
    )
    return model, SIGReg()


def lewm_forward_loss(model, sigreg, pixels, actions, sigreg_weight: float = 0.09):
    """
    Train o_t + action block -> o_{t+16}.

    Args:
        pixels: [B, 2, 3, H, W], frame 0 is o_t and frame 1 is o_{t+16}
        actions: [B, 16, action_dim]
    """
    if pixels.ndim != 5 or pixels.size(1) != 2:
        raise ValueError(f"Expected pixels [B, 2, C, H, W], got {tuple(pixels.shape)}")
    if actions.ndim != 3:
        raise ValueError(f"Expected actions [B, T, D], got {tuple(actions.shape)}")

    lewm_model = model.module if hasattr(model, "module") else model
    action_block = actions.reshape(actions.size(0), 1, -1)
    output = lewm_model.encode({"pixels": pixels, "action": action_block})
    emb = output["emb"]
    ctx_emb = emb[:, :1]
    target_emb = emb[:, 1:2]
    pred_emb = lewm_model.predict(ctx_emb, output["act_emb"])

    pred_loss = F.mse_loss(pred_emb, target_emb)
    sigreg_loss = sigreg(emb.transpose(0, 1))
    loss = pred_loss + sigreg_weight * sigreg_loss
    return {
        "loss": loss,
        "pred_loss": pred_loss,
        "sigreg_loss": sigreg_loss,
        "ctx_emb": ctx_emb,
        "target_emb": target_emb,
        "pred_emb": pred_emb,
    }
