#!/usr/bin/env python3
"""Evaluate whether a trained LIBERO FDM learned action-conditioned dynamics.

This diagnostic intentionally loads only the checkpoint's frozen DINO encoder
and FDM predictor.  It uses the same image packing, DINO preprocessing, action
normalization, and +8-frame target definition as FDM training, without loading
the Qwen policy or action head.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset
from starVLA.model.modules.dinov3_vit import DINOv3ViTModel
from starVLA.model.modules.world_model.DINOFeatureDynamics import DINOFeatureDynamicsPredictor


@dataclass
class Sample:
    suite: str
    dataset_name: str
    episode: int
    step: int
    language: str
    current_images: list[Image.Image]
    future_images: list[Image.Image]
    actions: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "/root/tianyi/starVLA/playground/Checkpoints/libero/"
            "libero_qwenoft_mip_dino_fdm_state7_100k/checkpoints/steps_60000_pytorch_model.pt"
        ),
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--episodes-per-suite", type=int, default=3)
    parser.add_argument("--steps-per-episode", type=int, default=4)
    parser.add_argument("--visual-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def checkpoint_run_dir(checkpoint: Path) -> Path:
    if checkpoint.parent.name != "checkpoints":
        raise ValueError(f"Expected <run>/checkpoints/<file>.pt, got {checkpoint}")
    return checkpoint.parents[1]


def suite_name(dataset_name: str) -> str:
    for name in ("spatial", "object", "goal", "10"):
        if f"libero_{name}_" in dataset_name:
            return f"libero_{name}"
    return dataset_name.replace("_no_noops_1.0.0_lerobot", "")


def load_samples(cfg, episodes_per_suite: int, steps_per_episode: int) -> list[Sample]:
    data_cfg = cfg.datasets.vla_data
    mixture = DATASET_NAMED_MIXTURES[str(data_cfg.data_mix)]
    samples: list[Sample] = []
    fractions = np.linspace(0.18, 0.82, steps_per_episode)

    for dataset_name, _weight, robot_type in mixture:
        dataset = make_LeRobotSingleDataset(
            Path(str(data_cfg.data_root_dir)),
            dataset_name,
            robot_type,
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
            data_cfg=data_cfg,
        )
        candidate_indices = np.linspace(
            0,
            len(dataset.trajectory_ids) - 1,
            max(episodes_per_suite * 4, episodes_per_suite),
            dtype=int,
        )
        accepted = 0
        for trajectory_index in candidate_indices:
            if accepted >= episodes_per_suite:
                break
            episode = int(dataset.trajectory_ids[trajectory_index])
            length = int(dataset.trajectory_lengths[trajectory_index])
            if length < 20:
                continue
            episode_samples: list[Sample] = []
            try:
                for fraction in fractions:
                    step = int(round(3 + fraction * (length - 12)))
                    step = min(max(step, 3), length - 9)
                    transformed = dataset.transforms(dataset.get_step_data(episode, step))
                    packed = dataset._pack_sample(transformed)
                    images = list(packed["image"])
                    future_images = list(packed["future_image"])
                    num_views = len(future_images)
                    if num_views <= 0 or len(images) % num_views:
                        raise ValueError(
                            f"Cannot recover view-major image packing: {len(images)} current, {num_views} future"
                        )
                    frames_per_view = len(images) // num_views
                    current_images = [images[(view + 1) * frames_per_view - 1] for view in range(num_views)]
                    actions = np.asarray(packed["action"], dtype=np.float32)[-8:]
                    episode_samples.append(
                        Sample(
                            suite=suite_name(dataset_name),
                            dataset_name=dataset_name,
                            episode=episode,
                            step=step,
                            language=str(packed["lang"]),
                            current_images=current_images,
                            future_images=future_images,
                            actions=actions,
                        )
                    )
            except Exception as exc:
                print(f"Skipping {dataset_name} episode {episode}: {exc}")
                continue
            samples.extend(episode_samples)
            accepted += 1
        if accepted != episodes_per_suite:
            raise RuntimeError(f"Collected only {accepted}/{episodes_per_suite} episodes from {dataset_name}")
    return samples


def strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    result = {key[len(prefix) :]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not result:
        raise KeyError(f"Checkpoint contains no keys beginning with {prefix!r}")
    return result


def build_models(cfg, checkpoint: Path, device: torch.device):
    fdm_cfg = cfg.framework.fdm
    dino_cfg = cfg.framework.dinov3
    action_cfg = cfg.framework.action_model
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"Memory-mapping checkpoint: {checkpoint}")
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    dino_state = strip_prefix(state_dict, "dino_model.")
    fdm_state = strip_prefix(state_dict, "fdm_predictor.")

    dino = DINOv3ViTModel.from_pretrained(str(dino_cfg.model_path))
    dino_result = dino.load_state_dict(dino_state, strict=True)
    fdm = DINOFeatureDynamicsPredictor(
        dino_dim=int(dino.config.hidden_size),
        action_dim=int(action_cfg.action_dim),
        action_horizon=int(action_cfg.action_horizon),
        hidden_dim=int(fdm_cfg.hidden_dim),
        depth=int(fdm_cfg.depth),
        num_heads=int(fdm_cfg.num_heads),
        mlp_ratio=float(fdm_cfg.mlp_ratio),
        dropout=float(fdm_cfg.dropout),
        max_patches=int(fdm_cfg.max_patches),
        max_horizons=1,
    )
    fdm_result = fdm.load_state_dict(fdm_state, strict=True)
    compatibility = {
        "dino_keys": len(dino_state),
        "fdm_keys": len(fdm_state),
        "dino_missing": list(dino_result.missing_keys),
        "dino_unexpected": list(dino_result.unexpected_keys),
        "fdm_missing": list(fdm_result.missing_keys),
        "fdm_unexpected": list(fdm_result.unexpected_keys),
    }
    del state_dict, dino_state, fdm_state

    dino = dino.to(device=device, dtype=dtype).eval()
    fdm = fdm.to(device=device, dtype=dtype).eval()
    return dino, fdm, compatibility


@torch.inference_mode()
def encode_dino(dino, image_batches: list[list[Image.Image]], input_hw, device, batch_size: int):
    height, width = map(int, input_hw)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    flat = []
    for images in image_batches:
        for image in images:
            array = np.asarray(image.convert("RGB").resize((width, height)), dtype=np.float32) / 255.0
            flat.append(torch.from_numpy(array).permute(2, 0, 1))
    pixels = (torch.stack(flat) - mean) / std
    outputs = []
    for start in range(0, len(pixels), batch_size):
        batch = pixels[start : start + batch_size].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            hidden = dino(pixel_values=batch).last_hidden_state
        registers = int(getattr(dino.config, "num_register_tokens", 0))
        outputs.append(hidden[:, 1 + registers :].float().cpu())
    tokens = torch.cat(outputs)
    num_samples = len(image_batches)
    num_views = len(image_batches[0])
    tokens = tokens.reshape(num_samples, num_views * tokens.shape[1], tokens.shape[2])
    return tokens


@torch.inference_mode()
def predict_fdm(fdm, current, actions, device, batch_size: int):
    predictions = []
    for start in range(0, len(current), batch_size):
        current_batch = current[start : start + batch_size].to(device)
        action_batch = actions[start : start + batch_size].to(device)
        patches = torch.arange(current_batch.shape[1], device=device).unsqueeze(0).expand(len(current_batch), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = fdm(current_batch, action_batch, patch_indices=patches)
        predictions.append(prediction.float().cpu())
    return torch.cat(predictions)


def mse_per_sample(prediction, target):
    return (prediction.float() - target.float()).square().mean(dim=(1, 2)).numpy()


def cosine_per_sample(left, right):
    left = left.float().flatten(1)
    right = right.float().flatten(1)
    return torch.nn.functional.cosine_similarity(left, right, dim=1).numpy()


def patch_change(tokens, current):
    return (tokens.float() - current.float()).square().mean(dim=2).sqrt().numpy()


def rowwise_correlation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = (left * right).sum(axis=1)
    denominator = np.sqrt((np.square(left).sum(axis=1) * np.square(right).sum(axis=1)).clip(1e-12))
    return np.nan_to_num(numerator / denominator)


def top_fraction_iou(left: np.ndarray, right: np.ndarray, fraction: float = 0.2) -> np.ndarray:
    count = max(1, int(round(left.shape[1] * fraction)))
    values = []
    for left_row, right_row in zip(left, right):
        left_set = set(np.argpartition(left_row, -count)[-count:].tolist())
        right_set = set(np.argpartition(right_row, -count)[-count:].tolist())
        values.append(len(left_set & right_set) / len(left_set | right_set))
    return np.asarray(values)


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 10000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def pca_rgb(current: np.ndarray, future: np.ndarray, prediction: np.ndarray):
    stacked = np.concatenate([current, future, prediction], axis=0).astype(np.float64)
    centered = stacked - stacked.mean(axis=0, keepdims=True)
    _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ vh[:3].T
    low = np.quantile(projected, 0.02, axis=0, keepdims=True)
    high = np.quantile(projected, 0.98, axis=0, keepdims=True)
    projected = np.clip((projected - low) / np.maximum(high - low, 1e-8), 0.0, 1.0)
    count = len(current)
    return projected[:count], projected[count : 2 * count], projected[2 * count :]


def save_feature_visualizations(
    output_dir: Path,
    samples: list[Sample],
    current: torch.Tensor,
    future: torch.Tensor,
    prediction: torch.Tensor,
    visual_indices: list[int],
):
    feature_dir = output_dir / "feature_maps"
    feature_dir.mkdir(parents=True, exist_ok=True)
    num_views = len(samples[0].current_images)
    patches_per_view = current.shape[1] // num_views
    side = int(round(math.sqrt(patches_per_view)))
    if side * side != patches_per_view:
        raise ValueError(f"Cannot render {patches_per_view} patches as a square")

    for index in visual_indices:
        sample = samples[index]
        fig, axes = plt.subplots(num_views, 8, figsize=(24, 3.5 * num_views), squeeze=False)
        for view in range(num_views):
            token_slice = slice(view * patches_per_view, (view + 1) * patches_per_view)
            current_tokens = current[index, token_slice].numpy()
            future_tokens = future[index, token_slice].numpy()
            pred_tokens = prediction[index, token_slice].numpy()
            current_rgb, future_rgb, pred_rgb = pca_rgb(current_tokens, future_tokens, pred_tokens)
            true_change = np.sqrt(np.mean((future_tokens - current_tokens) ** 2, axis=1)).reshape(side, side)
            pred_change = np.sqrt(np.mean((pred_tokens - current_tokens) ** 2, axis=1)).reshape(side, side)
            error = np.sqrt(np.mean((pred_tokens - future_tokens) ** 2, axis=1)).reshape(side, side)
            change_vmax = float(np.quantile(np.concatenate([true_change.ravel(), pred_change.ravel()]), 0.98))

            panels = [
                (np.asarray(sample.current_images[view]), "current image", None, None),
                (np.asarray(sample.future_images[view]), "future image (+8)", None, None),
                (current_rgb.reshape(side, side, 3), "current DINO PCA", None, None),
                (future_rgb.reshape(side, side, 3), "true future DINO PCA", None, None),
                (pred_rgb.reshape(side, side, 3), "FDM future DINO PCA", None, None),
                (true_change, "true |future-current|", "magma", change_vmax),
                (pred_change, "FDM |future-current|", "magma", change_vmax),
                (error, "FDM prediction error", "viridis", float(np.quantile(error, 0.98))),
            ]
            for column, (panel, title, cmap, vmax) in enumerate(panels):
                axes[view, column].imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=vmax, interpolation="nearest")
                axes[view, column].set_title(title, fontsize=9)
                axes[view, column].axis("off")
            axes[view, 0].set_ylabel(f"view {view}")
        fig.suptitle(
            f"{sample.suite} | episode {sample.episode}, step {sample.step}\n{sample.language}",
            fontsize=11,
        )
        fig.tight_layout()
        path = feature_dir / f"{index:03d}_{sample.suite}_ep{sample.episode}_step{sample.step}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_intervention_plot(
    output_dir: Path,
    samples: list[Sample],
    current: torch.Tensor,
    future: torch.Tensor,
    predictions: dict[str, torch.Tensor],
    visual_indices: list[int],
):
    num_views = len(samples[0].current_images)
    patches_per_view = current.shape[1] // num_views
    side = int(round(math.sqrt(patches_per_view)))
    view = 0
    columns = ["true action", "zero action", "negated action", "shuffled action"]
    fig, axes = plt.subplots(len(visual_indices), 7, figsize=(21, 3.2 * len(visual_indices)), squeeze=False)
    for row, index in enumerate(visual_indices):
        sample = samples[index]
        token_slice = slice(view * patches_per_view, (view + 1) * patches_per_view)
        true_change = patch_change(future[index : index + 1, token_slice], current[index : index + 1, token_slice])[0]
        maps = [
            patch_change(predictions[name][index : index + 1, token_slice], current[index : index + 1, token_slice])[0]
            for name in ("fdm_true", "fdm_zero", "fdm_negated", "fdm_shuffled")
        ]
        vmax = float(np.quantile(np.concatenate([true_change, *maps]), 0.98))
        axes[row, 0].imshow(sample.current_images[view])
        axes[row, 0].set_title("current image")
        axes[row, 1].imshow(sample.future_images[view])
        axes[row, 1].set_title("future image (+8)")
        axes[row, 2].imshow(true_change.reshape(side, side), cmap="magma", vmin=0, vmax=vmax)
        axes[row, 2].set_title("true change")
        for offset, (title, change_map) in enumerate(zip(columns, maps), start=3):
            axes[row, offset].imshow(change_map.reshape(side, side), cmap="magma", vmin=0, vmax=vmax)
            axes[row, offset].set_title(f"FDM: {title}")
        for axis in axes[row]:
            axis.axis("off")
        mean_xyz = sample.actions[:, :3].mean(axis=0)
        axes[row, 0].set_ylabel(
            f"{sample.suite}\nep{sample.episode} s{sample.step}\nmean xyz={mean_xyz.round(2)}",
            fontsize=8,
        )
    fig.suptitle("Action interventions: predicted DINO change, primary camera", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "action_interventions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    if len(visual_indices) < 2:
        return

    index = visual_indices[1]
    sample = samples[index]
    token_slice = slice(view * patches_per_view, (view + 1) * patches_per_view)
    current_tokens = current[index : index + 1, token_slice]
    true_future_feature = patch_change(future[index : index + 1, token_slice], current_tokens)[0]
    predicted_features = [
        patch_change(predictions[name][index : index + 1, token_slice], current_tokens)[0]
        for name in ("fdm_true", "fdm_negated", "fdm_zero")
    ]
    change_vmax = float(np.quantile(np.concatenate([true_future_feature, *predicted_features]), 0.98))
    panels = [
        (np.asarray(sample.current_images[view]), "Current Frame", None),
        (np.asarray(sample.future_images[view]), "Future Frame (t+8)", None),
        (true_future_feature.reshape(side, side), "True Feature Change", change_vmax),
        (predicted_features[0].reshape(side, side), "CWM: true action", change_vmax),
        (predicted_features[1].reshape(side, side), "CWM: negated action", change_vmax),
        (predicted_features[2].reshape(side, side), "CWM: zero action", change_vmax),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(18, 3.2), squeeze=False)
    for axis, (panel, title, vmax) in zip(axes[0], panels):
        axis.imshow(
            panel,
            cmap="magma" if vmax is not None else None,
            vmin=0 if vmax is not None else None,
            vmax=vmax,
            interpolation="nearest" if vmax is not None else None,
        )
        axis.set_title(title, fontsize=17, fontfamily="STIXGeneral")
        axis.axis("off")
    fig.tight_layout(pad=0.4, w_pad=0.35)
    fig.savefig(output_dir / "action_interventions_row2_features.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_aggregate_plots(output_dir: Path, errors: dict[str, np.ndarray], retrieval_ranks: np.ndarray, candidates: int):
    labels = ["persistence", "FDM true", "FDM zero", "FDM shuffled", "FDM negated"]
    keys = ["persistence", "fdm_true", "fdm_zero", "fdm_shuffled", "fdm_negated"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].boxplot([errors[key] for key in keys], tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("DINO token MSE (lower is better)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].set_title("Future-feature prediction baselines")
    bins = np.arange(0.5, candidates + 1.5, 1)
    axes[1].hist(retrieval_ranks, bins=bins, rwidth=0.85)
    axes[1].axhline(len(retrieval_ranks) / candidates, color="tab:red", linestyle="--", label="uniform chance")
    axes[1].set_xticks(range(1, candidates + 1))
    axes[1].set_xlabel("rank of the matched action (1 is best)")
    axes[1].set_ylabel("sample count")
    axes[1].set_title("Action retrieval from FDM error")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "aggregate_metrics.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_action_dimension_plot(output_dir: Path, dimension_metrics: dict[str, dict[str, float]]):
    labels = list(dimension_metrics)
    error_increase = [dimension_metrics[label]["target_mse_increase"] for label in labels]
    prediction_change = [dimension_metrics[label]["prediction_mse_vs_true_action"] for label in labels]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(labels, error_increase)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("target MSE increase")
    axes[0].set_title("Cost of shuffling one action dimension")
    axes[1].bar(labels, prediction_change)
    axes[1].set_ylabel("prediction MSE vs matched action")
    axes[1].set_title("FDM sensitivity to each action dimension")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_dir / "action_dimension_sensitivity.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summarize_group(indices, errors, delta_cosine, change_corr, change_iou):
    indices = np.asarray(indices, dtype=int)
    return {
        "samples": int(len(indices)),
        "mse_persistence": float(errors["persistence"][indices].mean()),
        "mse_fdm_true": float(errors["fdm_true"][indices].mean()),
        "mse_fdm_zero": float(errors["fdm_zero"][indices].mean()),
        "mse_fdm_shuffled": float(errors["fdm_shuffled"][indices].mean()),
        "mse_fdm_negated": float(errors["fdm_negated"][indices].mean()),
        "true_vs_persistence_improvement": float(
            1.0 - errors["fdm_true"][indices].mean() / errors["persistence"][indices].mean()
        ),
        "matched_action_advantage": float(
            errors["fdm_shuffled"][indices].mean() - errors["fdm_true"][indices].mean()
        ),
        "matched_action_win_rate": float(
            (errors["fdm_true"][indices] < errors["fdm_shuffled"][indices]).mean()
        ),
        "delta_cosine": float(delta_cosine[indices].mean()),
        "change_map_correlation": float(change_corr[indices].mean()),
        "change_top20_iou": float(change_iou[indices].mean()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    run_dir = checkpoint_run_dir(checkpoint)
    config_path = run_dir / "config.full.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"fdm_validation_{checkpoint.stem}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print("Collecting deterministic episode samples...")
    samples = load_samples(cfg, args.episodes_per_suite, args.steps_per_episode)
    print(f"Collected {len(samples)} samples from {len(set((s.suite, s.episode) for s in samples))} episodes")
    dino, fdm, compatibility = build_models(cfg, checkpoint, device)
    print("Strict DINO/FDM checkpoint loading passed:", compatibility)

    current = encode_dino(
        dino,
        [sample.current_images for sample in samples],
        cfg.framework.dinov3.input_size,
        device,
        args.batch_size,
    )
    future = encode_dino(
        dino,
        [sample.future_images for sample in samples],
        cfg.framework.dinov3.input_size,
        device,
        args.batch_size,
    )
    actions = torch.from_numpy(np.stack([sample.actions for sample in samples])).float()
    shuffled_actions = actions.roll(1, dims=0)
    action_variants = {
        "fdm_true": actions,
        "fdm_zero": torch.zeros_like(actions),
        "fdm_shuffled": shuffled_actions,
        "fdm_negated": -actions,
    }
    predictions = {
        name: predict_fdm(fdm, current, variant, device, args.batch_size)
        for name, variant in action_variants.items()
    }
    errors = {"persistence": mse_per_sample(current, future)}
    errors.update({name: mse_per_sample(prediction, future) for name, prediction in predictions.items()})

    true_delta = future - current
    pred_delta = predictions["fdm_true"] - current
    delta_cosine = cosine_per_sample(pred_delta, true_delta)
    true_change = patch_change(future, current)
    predicted_change = patch_change(predictions["fdm_true"], current)
    change_corr = rowwise_correlation(predicted_change, true_change)
    change_iou = top_fraction_iou(predicted_change, true_change)
    target_cosine = cosine_per_sample(predictions["fdm_true"], future)
    persistence_cosine = cosine_per_sample(current, future)
    delta_norm_ratio = (
        pred_delta.float().flatten(1).norm(dim=1) / true_delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    ).numpy()
    action_effect_ratio = (
        (predictions["fdm_true"] - predictions["fdm_shuffled"]).float().flatten(1).norm(dim=1)
        / true_delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    ).numpy()

    candidates = min(8, len(samples))
    candidate_errors = []
    for shift in range(candidates):
        candidate_prediction = predict_fdm(fdm, current, actions.roll(shift, dims=0), device, args.batch_size)
        candidate_errors.append(mse_per_sample(candidate_prediction, future))
    candidate_errors = np.stack(candidate_errors, axis=1)
    retrieval_ranks = 1 + (candidate_errors[:, 1:] < candidate_errors[:, :1]).sum(axis=1)
    retrieval_top1 = float((retrieval_ranks == 1).mean())
    retrieval_mean_rank = float(retrieval_ranks.mean())

    episode_top1 = []
    group_keys = sorted(set((sample.suite, sample.episode) for sample in samples))
    for key in group_keys:
        indices = [i for i, sample in enumerate(samples) if (sample.suite, sample.episode) == key]
        group_current = current[indices]
        group_future = future[indices]
        group_actions = actions[indices]
        group_errors = []
        for shift in range(len(indices)):
            group_prediction = predict_fdm(
                fdm,
                group_current,
                group_actions.roll(shift, dims=0),
                device,
                args.batch_size,
            )
            group_errors.append(mse_per_sample(group_prediction, group_future))
        group_errors = np.stack(group_errors, axis=1)
        episode_top1.extend((group_errors[:, 0] <= group_errors.min(axis=1) + 1e-12).tolist())

    action_dimension_names = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    action_dimension_metrics = {}
    for dimension, name in enumerate(action_dimension_names):
        shuffled_dimension_actions = actions.clone()
        shuffled_dimension_actions[:, :, dimension] = actions.roll(1, dims=0)[:, :, dimension]
        dimension_prediction = predict_fdm(
            fdm,
            current,
            shuffled_dimension_actions,
            device,
            args.batch_size,
        )
        dimension_error = mse_per_sample(dimension_prediction, future)
        action_dimension_metrics[name] = {
            "target_mse": float(dimension_error.mean()),
            "target_mse_increase": float((dimension_error - errors["fdm_true"]).mean()),
            "prediction_mse_vs_true_action": float(
                mse_per_sample(dimension_prediction, predictions["fdm_true"]).mean()
            ),
            "target_error_win_rate_for_matched_dimension": float(
                (errors["fdm_true"] < dimension_error).mean()
            ),
        }

    matched_advantage = errors["fdm_shuffled"] - errors["fdm_true"]
    overall_indices = list(range(len(samples)))
    overall = summarize_group(overall_indices, errors, delta_cosine, change_corr, change_iou)
    overall.update(
        {
            "mse_fdm_true_bootstrap_95ci": bootstrap_mean_ci(errors["fdm_true"], args.seed),
            "matched_action_advantage_bootstrap_95ci": bootstrap_mean_ci(matched_advantage, args.seed + 1),
            "target_cosine": float(target_cosine.mean()),
            "persistence_target_cosine": float(persistence_cosine.mean()),
            "predicted_delta_norm_ratio": float(delta_norm_ratio.mean()),
            "action_effect_vs_true_delta_norm": float(action_effect_ratio.mean()),
            "global_action_retrieval_candidates": candidates,
            "global_action_retrieval_top1": retrieval_top1,
            "global_action_retrieval_chance": 1.0 / candidates,
            "global_action_retrieval_mean_rank": retrieval_mean_rank,
            "within_episode_action_retrieval_top1": float(np.mean(episode_top1)),
            "within_episode_action_retrieval_chance": 1.0 / args.steps_per_episode,
        }
    )
    suites = {
        suite: summarize_group(
            [i for i, sample in enumerate(samples) if sample.suite == suite],
            errors,
            delta_cosine,
            change_corr,
            change_iou,
        )
        for suite in sorted(set(sample.suite for sample in samples))
    }
    num_views = len(samples[0].current_images)
    patches_per_view = current.shape[1] // num_views
    view_names = ["primary", "wrist"] if num_views == 2 else [f"view_{index}" for index in range(num_views)]
    by_view = {}
    for view, name in enumerate(view_names):
        token_slice = slice(view * patches_per_view, (view + 1) * patches_per_view)
        view_current = current[:, token_slice]
        view_future = future[:, token_slice]
        view_prediction = predictions["fdm_true"][:, token_slice]
        view_shuffled = predictions["fdm_shuffled"][:, token_slice]
        view_true_change = patch_change(view_future, view_current)
        view_pred_change = patch_change(view_prediction, view_current)
        view_persistence_error = mse_per_sample(view_current, view_future)
        view_true_error = mse_per_sample(view_prediction, view_future)
        view_shuffled_error = mse_per_sample(view_shuffled, view_future)
        by_view[name] = {
            "mse_persistence": float(view_persistence_error.mean()),
            "mse_fdm_true": float(view_true_error.mean()),
            "mse_fdm_shuffled": float(view_shuffled_error.mean()),
            "true_vs_persistence_improvement": float(
                1.0 - view_true_error.mean() / view_persistence_error.mean()
            ),
            "matched_action_advantage": float((view_shuffled_error - view_true_error).mean()),
            "matched_action_win_rate": float((view_true_error < view_shuffled_error).mean()),
            "delta_cosine": float(cosine_per_sample(view_prediction - view_current, view_future - view_current).mean()),
            "change_map_correlation": float(rowwise_correlation(view_pred_change, view_true_change).mean()),
            "change_top20_iou": float(top_fraction_iou(view_pred_change, view_true_change).mean()),
        }

    per_sample_rows = []
    for index, sample in enumerate(samples):
        row = {
            "index": index,
            "suite": sample.suite,
            "dataset": sample.dataset_name,
            "episode": sample.episode,
            "step": sample.step,
            "language": sample.language,
            "action_abs_mean": float(np.abs(sample.actions).mean()),
            "delta_cosine": float(delta_cosine[index]),
            "change_map_correlation": float(change_corr[index]),
            "change_top20_iou": float(change_iou[index]),
            "target_cosine": float(target_cosine[index]),
            "delta_norm_ratio": float(delta_norm_ratio[index]),
            "action_effect_ratio": float(action_effect_ratio[index]),
            "action_retrieval_rank": int(retrieval_ranks[index]),
        }
        row.update({f"mse_{name}": float(values[index]) for name, values in errors.items()})
        per_sample_rows.append(row)

    with (output_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_sample_rows[0]))
        writer.writeheader()
        writer.writerows(per_sample_rows)

    result = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "sampling": {
            "episodes_per_suite": args.episodes_per_suite,
            "steps_per_episode": args.steps_per_episode,
            "samples": len(samples),
            "episodes": len(group_keys),
            "seed": args.seed,
            "target_offset_frames": 8,
            "views": len(samples[0].current_images),
            "tokens_per_view": current.shape[1] // len(samples[0].current_images),
        },
        "checkpoint_compatibility": compatibility,
        "action_range": {
            "min": float(actions.min()),
            "max": float(actions.max()),
            "abs_mean": float(actions.abs().mean()),
        },
        "overall": overall,
        "by_suite": suites,
        "by_view": by_view,
        "by_action_dimension": action_dimension_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    visual_count = min(args.visual_samples, len(samples))
    visual_indices = np.linspace(0, len(samples) - 1, visual_count, dtype=int).tolist()
    save_feature_visualizations(output_dir, samples, current, future, predictions["fdm_true"], visual_indices)
    save_intervention_plot(output_dir, samples, current, future, predictions, visual_indices)
    save_aggregate_plots(output_dir, errors, retrieval_ranks, candidates)
    save_action_dimension_plot(output_dir, action_dimension_metrics)

    persistence_improvement = overall["true_vs_persistence_improvement"]
    action_advantage = overall["matched_action_advantage"]
    action_ci = overall["matched_action_advantage_bootstrap_95ci"]
    evidence_count = sum(
        [
            persistence_improvement > 0,
            action_advantage > 0 and action_ci[0] > 0,
            retrieval_top1 > 1.5 / candidates,
            overall["delta_cosine"] > 0,
            overall["change_map_correlation"] > 0,
        ]
    )
    verdict = "strong" if evidence_count >= 4 else "partial" if evidence_count >= 2 else "weak"
    lines = [
        "# FDM checkpoint validation",
        "",
        f"- Verdict: **{verdict} evidence of learned action-conditioned dynamics**",
        f"- Checkpoint: `{checkpoint}`",
        f"- Samples: {len(samples)} steps from {len(group_keys)} episodes across {len(suites)} LIBERO suites",
        f"- Strict checkpoint compatibility: DINO {compatibility['dino_keys']} keys, FDM {compatibility['fdm_keys']} keys, no missing/unexpected keys",
        "",
        "## Main quantitative results",
        "",
        f"- FDM(true action) MSE: {overall['mse_fdm_true']:.8f}",
        f"- Persistence(current token) MSE: {overall['mse_persistence']:.8f}",
        f"- Relative improvement over persistence: {100 * persistence_improvement:.2f}%",
        f"- FDM(shuffled action) MSE: {overall['mse_fdm_shuffled']:.8f}",
        f"- Matched-action advantage (shuffled - true): {action_advantage:.8f}, bootstrap 95% CI [{action_ci[0]:.8f}, {action_ci[1]:.8f}]",
        f"- Matched action beats shuffled action: {100 * overall['matched_action_win_rate']:.1f}%",
        f"- Global {candidates}-way action retrieval top-1: {100 * retrieval_top1:.1f}% (chance {100 / candidates:.1f}%)",
        f"- Within-episode action retrieval top-1: {100 * overall['within_episode_action_retrieval_top1']:.1f}% (chance {100 / args.steps_per_episode:.1f}%)",
        f"- Predicted/true feature-delta cosine: {overall['delta_cosine']:.4f}",
        f"- Spatial change-map correlation: {overall['change_map_correlation']:.4f}",
        f"- Spatial top-20% change IoU: {overall['change_top20_iou']:.4f}",
        f"- Primary-camera improvement over persistence: {100 * by_view['primary']['true_vs_persistence_improvement']:.2f}%",
        f"- Wrist-camera improvement over persistence: {100 * by_view['wrist']['true_vs_persistence_improvement']:.2f}%",
        "- Per-dimension shuffled-action MSE increase: "
        + ", ".join(
            f"{name}={values['target_mse_increase']:.6f}"
            for name, values in action_dimension_metrics.items()
        ),
        "",
        "## Interpretation limits",
        "",
        "These samples come from the training mixture, so this establishes in-distribution learning, not held-out generalization. "
        "Shuffled/zero/negated actions test action dependence. They cannot by themselves prove that an unsupported counterfactual action is physically correct; simulator rollouts or paired counterfactual data are required for that stronger claim.",
        "",
        "See `aggregate_metrics.png`, `action_interventions.png`, `feature_maps/`, `metrics.json`, and `per_sample.csv`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
