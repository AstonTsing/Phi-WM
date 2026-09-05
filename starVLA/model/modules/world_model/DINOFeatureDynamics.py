"""Lightweight DINO-token dynamics predictor for PhiWAM FDM training."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(hidden, hidden, hidden, need_weights=False)
        hidden = hidden + attn_out
        hidden = hidden + self.ffn(hidden)
        return hidden


class DINOFeatureDynamicsPredictor(nn.Module):
    """Predict future DINO patch tokens from current tokens and an action chunk."""

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
    ) -> None:
        super().__init__()
        self.dino_dim = int(dino_dim)
        self.action_dim = int(action_dim)
        self.action_horizon = int(action_horizon)
        self.hidden_dim = int(hidden_dim)
        self.max_horizons = int(max_horizons)

        self.token_proj = nn.Linear(self.dino_dim, self.hidden_dim)
        self.patch_embedding = nn.Embedding(int(max_patches), self.hidden_dim)
        self.horizon_embedding = (
            nn.Embedding(self.max_horizons, self.hidden_dim) if self.max_horizons > 1 else None
        )
        self.action_proj = nn.Sequential(
            nn.Linear(self.action_horizon * self.action_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                _ResidualAttentionBlock(
                    hidden_dim=self.hidden_dim,
                    num_heads=int(num_heads),
                    mlp_ratio=float(mlp_ratio),
                    dropout=float(dropout),
                )
                for _ in range(int(depth))
            ]
        )
        self.output = nn.Linear(self.hidden_dim, self.dino_dim)

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
        batch_size, num_tokens, _ = current_tokens.shape
        if actions.shape[0] != batch_size:
            raise ValueError("actions and current_tokens batch sizes must match")
        if actions.shape[1] != self.action_horizon or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"expected actions [B,{self.action_horizon},{self.action_dim}], got {tuple(actions.shape)}"
            )

        compute_dtype = self.token_proj.weight.dtype
        current_tokens = current_tokens.to(dtype=compute_dtype)
        actions = actions.to(dtype=compute_dtype)

        hidden = self.token_proj(current_tokens)
        if patch_indices is None:
            patch_indices = torch.arange(num_tokens, device=current_tokens.device).unsqueeze(0).expand(batch_size, -1)
        patch_indices = patch_indices.to(device=current_tokens.device, dtype=torch.long)
        if patch_indices.shape != (batch_size, num_tokens):
            raise ValueError(f"patch_indices must be [B,K], got {tuple(patch_indices.shape)}")
        hidden = hidden + self.patch_embedding(patch_indices.clamp_min(0).clamp_max(self.patch_embedding.num_embeddings - 1))

        if self.horizon_embedding is not None:
            if horizon_indices is None:
                horizon_indices = torch.zeros(batch_size, device=current_tokens.device, dtype=torch.long)
            elif isinstance(horizon_indices, int):
                horizon_indices = torch.full((batch_size,), horizon_indices, device=current_tokens.device, dtype=torch.long)
            else:
                horizon_indices = horizon_indices.to(device=current_tokens.device, dtype=torch.long)
            if horizon_indices.shape != (batch_size,):
                raise ValueError(f"horizon_indices must be [B], got {tuple(horizon_indices.shape)}")
            if horizon_indices.max().item() >= self.max_horizons or horizon_indices.min().item() < 0:
                raise ValueError(f"horizon_indices must be in [0, {self.max_horizons})")
            hidden = hidden + self.horizon_embedding(horizon_indices)[:, None, :]

        action_cond = self.action_proj(actions.reshape(batch_size, -1))
        hidden = hidden + action_cond[:, None, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(hidden)


def fdm_distance_per_sample(
    pred_tokens: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    normalize: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return mean-squared DINO-token distance for each sample."""
    if pred_tokens.shape != target_tokens.shape:
        raise ValueError(f"pred/target shape mismatch: pred={tuple(pred_tokens.shape)} target={tuple(target_tokens.shape)}")
    distance = (pred_tokens.float() - target_tokens.float()).square().mean(dim=(-1, -2))
    if normalize:
        scale = target_tokens.float().var(dim=(-1, -2), unbiased=False).clamp_min(eps)
        distance = distance / scale
    return distance


def _soft_rank(better: torch.Tensor, worse: torch.Tensor, *, margin: float, tau: float) -> torch.Tensor:
    if tau <= 0:
        raise ValueError(f"rank temperature must be positive, got {tau}")
    return F.softplus((better - worse + float(margin)) / float(tau)).mean()


def pacer_dynamics_loss(
    d_gt: torch.Tensor,
    d_neg: torch.Tensor = None,
    d_hard: torch.Tensor = None,
    *,
    neg_weight: float = 0.5,
    neg_margin: float = 0.1,
    neg_tau: float = 0.5,
    hard_weight: float = 0.5,
    hard_margin: float = 0.02,
    hard_tau: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Train FDM only on recorded actions and data-supported negative actions."""
    total = d_gt.mean()
    metrics = {"fdm_dyn": d_gt.mean().detach()}

    if d_neg is not None:
        rank_neg = _soft_rank(d_gt, d_neg, margin=neg_margin, tau=neg_tau)
        total = total + float(neg_weight) * rank_neg
        metrics["fdm_loss_neg"] = rank_neg.detach()
        metrics["fdm_dist_neg"] = d_neg.mean().detach()
        metrics["fdm_sensitivity"] = (d_neg.mean() / d_gt.mean().clamp_min(1e-8)).detach()

    if d_hard is not None:
        rank_hard = _soft_rank(d_gt, d_hard, margin=hard_margin, tau=hard_tau)
        total = total + float(hard_weight) * rank_hard
        metrics["fdm_loss_hard"] = rank_hard.detach()
        metrics["fdm_dist_hard"] = d_hard.mean().detach()
        metrics["fdm_sensitivity_hard"] = (d_hard.mean() / d_gt.mean().clamp_min(1e-8)).detach()

    metrics["fdm_branch_loss"] = total.detach()
    return total, metrics


def pacer_feedback_loss(
    d_stage1: torch.Tensor,
    d_stage0: torch.Tensor,
    d_direct: torch.Tensor = None,
    *,
    match_weight: float = 1.0,
    rank_weight: float = 1.0,
    rank_margin: float = 0.0,
    rank_tau: float = 0.5,
    use_stage0_direct: bool = True,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Policy-side FDM feedback; FDM weights are frozen by the caller."""
    match = d_stage1.mean()
    total = float(match_weight) * match
    metrics = {
        "feedback_match": match.detach(),
        "fdm_dist_stage1": d_stage1.mean().detach(),
        "fdm_dist_stage0": d_stage0.mean().detach(),
        "order_stage1_stage0": (d_stage1 < d_stage0).float().mean().detach(),
    }

    rank_s1_s0 = _soft_rank(d_stage1, d_stage0.detach(), margin=rank_margin, tau=rank_tau)
    total = total + float(rank_weight) * rank_s1_s0
    metrics["feedback_rank_stage1_stage0"] = rank_s1_s0.detach()
    metrics["feedback_saturation"] = ((d_stage1 - d_stage0) / float(rank_tau) < -3.0).float().mean().detach()

    if d_direct is not None:
        metrics["fdm_dist_direct"] = d_direct.mean().detach()
        metrics["order_stage0_direct"] = (d_stage0 < d_direct).float().mean().detach()
        if use_stage0_direct:
            rank_s0_dir = _soft_rank(d_stage0, d_direct.detach(), margin=rank_margin, tau=rank_tau)
            total = total + float(rank_weight) * rank_s0_dir
            metrics["feedback_rank_stage0_direct"] = rank_s0_dir.detach()

    metrics["feedback_loss"] = total.detach()
    return total, metrics


def mip_fdm_loss(
    pred_stage0: torch.Tensor,
    pred_stage1: torch.Tensor,
    target_tokens: torch.Tensor,
    *,
    stage0_weight: float = 0.25,
    rank_weight: float = 0.1,
    rank_margin: float = 0.05,
    rank_tau: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """MIP FDM objective: reconstruction plus stage1-better-than-stage0 ranking."""
    d0 = fdm_distance_per_sample(pred_stage0, target_tokens)
    d1 = fdm_distance_per_sample(pred_stage1, target_tokens)
    stage0 = d0.mean()
    stage1 = d1.mean()
    rank = _soft_rank(d1, d0.detach(), margin=rank_margin, tau=rank_tau)
    total = stage1 + float(stage0_weight) * stage0 + float(rank_weight) * rank
    return total, {
        "fdm_loss": total,
        "fdm_loss_stage0": stage0,
        "fdm_loss_stage1": stage1,
        "fdm_rank": rank,
    }
