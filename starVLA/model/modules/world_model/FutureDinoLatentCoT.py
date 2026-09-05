"""Compact autoregressive latent reasoning over future DINO features."""

from __future__ import annotations

import torch
import torch.nn as nn


class FutureDinoLatentCoT(nn.Module):
    """Generate a short future-feature trajectory before action prediction."""

    def __init__(
        self,
        *,
        context_dim: int,
        dino_dim: int,
        num_latents: int = 8,
        hidden_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_latents = int(num_latents)
        self.context_proj = nn.Linear(int(context_dim), int(hidden_dim))
        self.context_norm = nn.LayerNorm(int(hidden_dim))
        self.target_proj = nn.Linear(int(dino_dim), int(hidden_dim))
        self.start_token = nn.Parameter(torch.zeros(1, int(hidden_dim)))
        self.position_embedding = nn.Embedding(self.num_latents, int(hidden_dim))
        self.cross_attention = nn.MultiheadAttention(
            int(hidden_dim), int(num_heads), dropout=float(dropout), batch_first=True
        )
        self.recurrence = nn.GRUCell(int(hidden_dim), int(hidden_dim))
        self.output_norm = nn.LayerNorm(int(hidden_dim))
        self.output = nn.Linear(int(hidden_dim), int(dino_dim))
        nn.init.normal_(self.start_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        *,
        context_mask: torch.Tensor | None = None,
        teacher_targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_tokens.ndim != 3:
            raise ValueError(f"context_tokens must be [B,N,D], got {tuple(context_tokens.shape)}")
        if teacher_targets is not None and teacher_targets.shape[:2] != (
            context_tokens.shape[0], self.num_latents
        ):
            raise ValueError(
                f"teacher_targets must be [B,{self.num_latents},D], got {tuple(teacher_targets.shape)}"
            )

        compute_dtype = self.context_proj.weight.dtype
        memory = self.context_norm(self.context_proj(context_tokens.to(dtype=compute_dtype)))
        if context_mask is None:
            valid_mask = torch.ones(memory.shape[:2], device=memory.device, dtype=torch.bool)
        else:
            valid_mask = context_mask.to(device=memory.device, dtype=torch.bool)
            if valid_mask.shape != memory.shape[:2]:
                raise ValueError(f"context_mask must be {tuple(memory.shape[:2])}, got {tuple(valid_mask.shape)}")

        weights = valid_mask.to(dtype=memory.dtype).unsqueeze(-1)
        hidden = (memory * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        previous = self.start_token.to(dtype=memory.dtype).expand(memory.shape[0], -1)
        positions = self.position_embedding.weight.to(dtype=memory.dtype)
        teacher_targets = teacher_targets.detach() if teacher_targets is not None else None

        predictions = []
        for index in range(self.num_latents):
            query = (hidden + positions[index]).unsqueeze(1)
            attended, _ = self.cross_attention(
                query,
                memory,
                memory,
                key_padding_mask=~valid_mask,
                need_weights=False,
            )
            hidden = self.recurrence(previous + attended[:, 0] + positions[index], hidden)
            prediction = self.output(self.output_norm(hidden))
            predictions.append(prediction)
            source = teacher_targets[:, index] if teacher_targets is not None else prediction
            previous = self.target_proj(source.to(dtype=compute_dtype))

        return torch.stack(predictions, dim=1)
