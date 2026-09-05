"""LeWM-style action-conditioned predictor for future DINO patch tokens.

This module is a drop-in architectural alternative to
``DINOFeatureDynamics.DINOFeatureDynamicsPredictor``.  It keeps the DINO
patch-token interface, while following LeWM's predictor design more closely:

* the complete action chunk is embedded by a 1x1 Conv + MLP;
* every transformer block is conditioned through adaLN-Zero;
* attention and MLP residuals have condition-dependent gates;
* a final LayerNorm and prediction MLP map back to DINO feature space.

Attention is non-causal by default because the sequence dimension contains
spatial patch tokens rather than an autoregressive time sequence.  Set
``causal_attention=True`` for an exact LeWM-style attention mask ablation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .DINOFeatureDynamics import (
    fdm_distance_per_sample,
    mip_fdm_loss,
    pacer_dynamics_loss,
    pacer_feedback_loss,
)


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class _LeWMActionEmbedder(nn.Module):
    """Embed one flattened action chunk in the same style as LeWM's Embedder."""

    def __init__(self, action_block_dim: int, hidden_dim: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        embed_hidden_dim = int(hidden_dim * mlp_ratio)
        self.patch_embed = nn.Conv1d(
            action_block_dim,
            action_block_dim,
            kernel_size=1,
            stride=1,
        )
        self.embed = nn.Sequential(
            nn.Linear(action_block_dim, embed_hidden_dim),
            nn.SiLU(),
            nn.Linear(embed_hidden_dim, hidden_dim),
        )

    def forward(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """Return an action condition with shape ``[B, 1, hidden_dim]``."""
        action_chunk = action_chunk.transpose(1, 2)
        action_chunk = self.patch_embed(action_chunk)
        action_chunk = action_chunk.transpose(1, 2)
        return self.embed(action_chunk)


class _LeWMFeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # Keep LeWM's internal LayerNorm in addition to the modulated block norm.
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _LeWMAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        dropout: float,
        causal: bool,
    ) -> None:
        super().__init__()
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.dropout = float(dropout)
        self.causal = bool(causal)
        inner_dim = self.heads * self.dim_head

        # LeWM normalizes inside attention as well as before modulation.
        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, _ = x.shape
        x = self.norm(x)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        def split_heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch_size, num_tokens, self.heads, self.dim_head).transpose(1, 2)

        q, k, v = (split_heads(tensor) for tensor in (q, k, v))
        dropout_p = self.dropout if self.training else 0.0
        hidden = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=dropout_p,
            is_causal=self.causal,
        )
        hidden = hidden.transpose(1, 2).reshape(batch_size, num_tokens, -1)
        return self.to_out(hidden)


class _LeWMConditionalBlock(nn.Module):
    """LeWM/DiT-style adaLN-Zero conditioned transformer block."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float,
        causal_attention: bool,
    ) -> None:
        super().__init__()
        self.attn = _LeWMAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            causal=causal_attention,
        )
        self.mlp = _LeWMFeedForward(dim, mlp_dim, dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # As in LeWM, each conditioned residual branch starts closed.  This
        # makes the predictor an identity stack at initialization.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, hidden: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=-1)
        )
        hidden = hidden + gate_msa * self.attn(_modulate(self.norm1(hidden), shift_msa, scale_msa))
        hidden = hidden + gate_mlp * self.mlp(_modulate(self.norm2(hidden), shift_mlp, scale_mlp))
        return hidden


class DINOFeatureDynamicsPredictor(nn.Module):
    """Predict future DINO patch tokens using a LeWM-style conditional model.

    The public interface intentionally matches the original predictor so a
    framework can switch between the two implementations by changing only its
    import path.
    """

    def __init__(
        self,
        *,
        dino_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 768,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_patches: int = 2048,
        max_horizons: int = 1,
        dim_head: int | None = None,
        action_mlp_ratio: float = 4.0,
        output_mlp_ratio: float = 4.0,
        causal_attention: bool = False,
    ) -> None:
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.max_horizons = int(max_horizons)

        if self.max_horizons < 1:
            raise ValueError(f"max_horizons must be positive, got {self.max_horizons}")
        if int(max_patches) < 1:
            raise ValueError(f"max_patches must be positive, got {max_patches}")
        if int(num_heads) < 1:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if dim_head is None:
            if self.hidden_dim % int(num_heads) != 0:
                raise ValueError(
                    f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({num_heads}) "
                    "when dim_head is not provided"
                )
            dim_head = self.hidden_dim // int(num_heads)

        action_block_dim = self.action_horizon * self.action_dim
        self.token_proj = (
            nn.Linear(self.dino_dim, self.hidden_dim)
            if self.dino_dim != self.hidden_dim
            else nn.Identity()
        )
        self.patch_embedding = nn.Embedding(int(max_patches), self.hidden_dim)
        self.horizon_embedding = (
            nn.Embedding(self.max_horizons, self.hidden_dim)
            if self.max_horizons > 1
            else None
        )
        self.action_encoder = _LeWMActionEmbedder(
            action_block_dim=action_block_dim,
            hidden_dim=self.hidden_dim,
            mlp_ratio=float(action_mlp_ratio),
        )
        self.blocks = nn.ModuleList(
            [
                _LeWMConditionalBlock(
                    dim=self.hidden_dim,
                    heads=int(num_heads),
                    dim_head=int(dim_head),
                    mlp_dim=int(self.hidden_dim * float(mlp_ratio)),
                    dropout=float(dropout),
                    causal_attention=bool(causal_attention),
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)

        output_hidden_dim = int(self.hidden_dim * float(output_mlp_ratio))
        self.pred_proj = nn.Sequential(
            nn.Linear(self.hidden_dim, output_hidden_dim),
            nn.LayerNorm(output_hidden_dim),
            nn.GELU(),
            nn.Linear(output_hidden_dim, self.dino_dim),
        )

    def _get_horizon_condition(
        self,
        horizon_indices: torch.Tensor | int | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.horizon_embedding is None:
            return None
        if horizon_indices is None:
            horizon_indices = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif isinstance(horizon_indices, int):
            horizon_indices = torch.full(
                (batch_size,),
                horizon_indices,
                device=device,
                dtype=torch.long,
            )
        else:
            horizon_indices = horizon_indices.to(device=device, dtype=torch.long)
        if horizon_indices.shape != (batch_size,):
            raise ValueError(f"horizon_indices must be [B], got {tuple(horizon_indices.shape)}")
        if horizon_indices.numel() > 0 and (
            horizon_indices.max().item() >= self.max_horizons
            or horizon_indices.min().item() < 0
        ):
            raise ValueError(f"horizon_indices must be in [0, {self.max_horizons})")
        return self.horizon_embedding(horizon_indices)[:, None, :]

    def forward(
        self,
        current_tokens: torch.Tensor,
        actions: torch.Tensor,
        *,
        patch_indices: torch.Tensor | None = None,
        horizon_indices: torch.Tensor | int | None = None,
    ) -> torch.Tensor:
        if current_tokens.ndim != 3:
            raise ValueError(f"current_tokens must be [B,K,D], got {tuple(current_tokens.shape)}")
        if actions.ndim != 3:
            raise ValueError(f"actions must be [B,H,A], got {tuple(actions.shape)}")

        batch_size, num_tokens, token_dim = current_tokens.shape
        if token_dim != self.dino_dim:
            raise ValueError(f"expected DINO token dim {self.dino_dim}, got {token_dim}")
        if actions.shape != (batch_size, self.action_horizon, self.action_dim):
            raise ValueError(
                f"expected actions [B,{self.action_horizon},{self.action_dim}], "
                f"got {tuple(actions.shape)}"
            )

        parameter = next(self.parameters())
        compute_dtype = parameter.dtype
        current_tokens = current_tokens.to(dtype=compute_dtype)
        actions = actions.to(device=current_tokens.device, dtype=compute_dtype)

        hidden = self.token_proj(current_tokens)
        if patch_indices is None:
            patch_indices = torch.arange(num_tokens, device=current_tokens.device)
            patch_indices = patch_indices.unsqueeze(0).expand(batch_size, -1)
        patch_indices = patch_indices.to(device=current_tokens.device, dtype=torch.long)
        if patch_indices.shape != (batch_size, num_tokens):
            raise ValueError(f"patch_indices must be [B,K], got {tuple(patch_indices.shape)}")
        patch_indices = patch_indices.clamp(0, self.patch_embedding.num_embeddings - 1)
        hidden = hidden + self.patch_embedding(patch_indices)

        action_chunk = actions.reshape(batch_size, 1, -1)
        condition = self.action_encoder(action_chunk)
        horizon_condition = self._get_horizon_condition(
            horizon_indices,
            batch_size=batch_size,
            device=current_tokens.device,
        )
        if horizon_condition is not None:
            condition = condition + horizon_condition

        # condition is [B,1,D]; adaLN parameters broadcast over all K patches.
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.pred_proj(self.norm(hidden))


# Explicit alias for code that wants both predictor variants imported together.
LeWMDINOFeatureDynamicsPredictor = DINOFeatureDynamicsPredictor


__all__ = [
    "DINOFeatureDynamicsPredictor",
    "LeWMDINOFeatureDynamicsPredictor",
    "fdm_distance_per_sample",
    "pacer_dynamics_loss",
    "pacer_feedback_loss",
    "mip_fdm_loss",
]
