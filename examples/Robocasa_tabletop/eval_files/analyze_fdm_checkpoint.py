#!/usr/bin/env python3
"""Validate action-conditioned DINO dynamics on real RoboCasa-GR1 episodes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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

from examples.LIBERO.eval_files.analyze_fdm_checkpoint import (
    Sample,
    bootstrap_mean_ci,
    build_models,
    cosine_per_sample,
    encode_dino,
    mse_per_sample,
    patch_change,
    pca_rgb,
    predict_fdm,
    rowwise_correlation,
    top_fraction_iou,
)
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


CHECKPOINT = (
    "/root/tianyi/starVLA/playground/Checkpoints/robocasa/"
    "robocasa_qwenoft_mip_dino_fdm_state58_200k/checkpoints/steps_180000_pytorch_model.pt"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-datasets", type=int, default=8)
    parser.add_argument("--episodes-per-dataset", type=int, default=2)
    parser.add_argument("--steps-per-episode", type=int, default=4)
    parser.add_argument("--visual-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260829)
    return parser.parse_args()


def short_task_name(dataset_name: str) -> str:
    name = dataset_name.split(".")[-1]
    for suffix in ("_GR1ArmsAndWaistFourierHands_1000", "_1000"):
        name = name.removesuffix(suffix)
    return name


def future_offset(data_cfg) -> int:
    mixture = DATASET_NAMED_MIXTURES[str(data_cfg.data_mix)]
    robot_type = mixture[0][2]
    from starVLA.dataloader.gr00t_lerobot.registry import ROBOT_TYPE_CONFIG_MAP

    indices = list(ROBOT_TYPE_CONFIG_MAP[robot_type].future_observation_indices)
    if not indices:
        raise ValueError(f"No future observation indices for {robot_type}")
    return int(indices[-1])


def load_samples(cfg, args) -> tuple[list[Sample], list[str]]:
    data_cfg = cfg.datasets.vla_data
    mixture = DATASET_NAMED_MIXTURES[str(data_cfg.data_mix)]
    count = min(max(1, args.max_datasets), len(mixture))
    selected = np.unique(np.linspace(0, len(mixture) - 1, count, dtype=int)).tolist()
    fractions = np.linspace(0.15, 0.85, args.steps_per_episode)
    offset = future_offset(data_cfg)
    horizon = int(cfg.framework.action_model.action_horizon)
    samples: list[Sample] = []
    task_names: list[str] = []

    for mixture_index in selected:
        dataset_name, _weight, robot_type = mixture[mixture_index]
        task = short_task_name(dataset_name)
        task_names.append(task)
        dataset = make_LeRobotSingleDataset(
            Path(str(data_cfg.data_root_dir)),
            dataset_name,
            robot_type,
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
            data_cfg=data_cfg,
        )
        candidates = np.linspace(
            0,
            len(dataset.trajectory_ids) - 1,
            max(args.episodes_per_dataset * 4, args.episodes_per_dataset),
            dtype=int,
        )
        accepted = 0
        for trajectory_index in candidates:
            if accepted >= args.episodes_per_dataset:
                break
            episode = int(dataset.trajectory_ids[trajectory_index])
            length = int(dataset.trajectory_lengths[trajectory_index])
            if length <= offset + 4:
                continue
            episode_samples = []
            try:
                for fraction in fractions:
                    step = int(round(fraction * (length - offset - 1)))
                    step = min(max(step, 0), length - offset - 1)
                    packed = dataset._pack_sample(dataset.transforms(dataset.get_step_data(episode, step)))
                    current_images = list(packed["image"])
                    all_future_images = list(packed["future_image"])
                    if len(current_images) != 1 or not all_future_images:
                        raise ValueError(
                            f"Expected one current view and future frames, got "
                            f"{len(current_images)} and {len(all_future_images)}"
                        )
                    actions = np.asarray(packed["action"], dtype=np.float32)[-horizon:]
                    if actions.shape != (horizon, int(cfg.framework.action_model.action_dim)):
                        raise ValueError(f"Unexpected action shape {actions.shape}")
                    episode_samples.append(
                        Sample(
                            suite=task,
                            dataset_name=dataset_name,
                            episode=episode,
                            step=step,
                            language=str(packed["lang"]),
                            current_images=current_images,
                            future_images=[all_future_images[-1]],
                            actions=actions,
                        )
                    )
            except Exception as exc:
                print(f"Skipping {task} episode {episode}: {exc}", flush=True)
                continue
            samples.extend(episode_samples)
            accepted += 1
        if accepted != args.episodes_per_dataset:
            raise RuntimeError(f"Collected only {accepted}/{args.episodes_per_dataset} episodes from {task}")
    return samples, task_names


def action_dimension_names() -> list[str]:
    groups = (("left_arm", 7), ("right_arm", 7), ("left_hand", 6), ("right_hand", 6), ("waist", 3))
    return [f"{group}_{index}" for group, size in groups for index in range(size)]


def save_feature_maps(output_dir, samples, current, future, prediction, indices, offset):
    target_dir = output_dir / "feature_maps"
    target_dir.mkdir(parents=True, exist_ok=True)
    side = int(round(math.sqrt(current.shape[1])))
    for index in indices:
        sample = samples[index]
        current_tokens = current[index].numpy()
        future_tokens = future[index].numpy()
        pred_tokens = prediction[index].numpy()
        current_rgb, future_rgb, pred_rgb = pca_rgb(current_tokens, future_tokens, pred_tokens)
        true_change = np.sqrt(np.mean((future_tokens - current_tokens) ** 2, axis=1)).reshape(side, side)
        pred_change = np.sqrt(np.mean((pred_tokens - current_tokens) ** 2, axis=1)).reshape(side, side)
        error = np.sqrt(np.mean((pred_tokens - future_tokens) ** 2, axis=1)).reshape(side, side)
        vmax = float(np.quantile(np.concatenate([true_change.ravel(), pred_change.ravel()]), 0.98))
        panels = [
            (np.asarray(sample.current_images[0]), "current RGB", None, None),
            (np.asarray(sample.future_images[0]), f"future RGB (+{offset})", None, None),
            (current_rgb.reshape(side, side, 3), "current DINO PCA", None, None),
            (future_rgb.reshape(side, side, 3), "true future DINO PCA", None, None),
            (pred_rgb.reshape(side, side, 3), "FDM future DINO PCA", None, None),
            (true_change, "true |future-current|", "magma", vmax),
            (pred_change, "FDM |future-current|", "magma", vmax),
            (error, "FDM prediction error", "viridis", float(np.quantile(error, 0.98))),
        ]
        fig, axes = plt.subplots(1, 8, figsize=(24, 3.5))
        for axis, (panel, title, cmap, panel_vmax) in zip(axes, panels):
            axis.imshow(panel, cmap=cmap, vmin=0 if cmap else None, vmax=panel_vmax, interpolation="nearest")
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(f"{sample.suite} | episode {sample.episode}, step {sample.step}\n{sample.language}", fontsize=10)
        fig.tight_layout()
        fig.savefig(target_dir / f"{index:03d}_{sample.suite}_ep{sample.episode}_step{sample.step}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)


def save_interventions(output_dir, samples, current, future, predictions, indices, offset):
    side = int(round(math.sqrt(current.shape[1])))
    fig, axes = plt.subplots(len(indices), 7, figsize=(21, 3.2 * len(indices)), squeeze=False)
    variants = ("fdm_true", "fdm_zero", "fdm_negated", "fdm_shuffled")
    titles = ("true action", "zero action", "negated action", "shuffled action")
    for row, index in enumerate(indices):
        sample = samples[index]
        true_change = patch_change(future[index:index + 1], current[index:index + 1])[0]
        maps = [patch_change(predictions[name][index:index + 1], current[index:index + 1])[0] for name in variants]
        vmax = float(np.quantile(np.concatenate([true_change, *maps]), 0.98))
        axes[row, 0].imshow(sample.current_images[0]); axes[row, 0].set_title("current RGB")
        axes[row, 1].imshow(sample.future_images[0]); axes[row, 1].set_title(f"future RGB (+{offset})")
        axes[row, 2].imshow(true_change.reshape(side, side), cmap="magma", vmin=0, vmax=vmax)
        axes[row, 2].set_title("true change")
        for column, (title, values) in enumerate(zip(titles, maps), start=3):
            axes[row, column].imshow(values.reshape(side, side), cmap="magma", vmin=0, vmax=vmax)
            axes[row, column].set_title(f"FDM: {title}")
        for axis in axes[row]:
            axis.axis("off")
        axes[row, 0].set_ylabel(f"{sample.suite}\nep{sample.episode} s{sample.step}", fontsize=8)
    fig.suptitle("RoboCasa action interventions: predicted DINO change", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "action_interventions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_aggregate(output_dir, errors, ranks, candidates, dimension_metrics):
    labels = ["persistence", "FDM true", "FDM zero", "FDM shuffled", "FDM negated"]
    keys = ["persistence", "fdm_true", "fdm_zero", "fdm_shuffled", "fdm_negated"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].boxplot([errors[key] for key in keys], tick_labels=labels, showfliers=False)
    axes[0].set_ylabel("DINO token MSE (lower is better)"); axes[0].tick_params(axis="x", rotation=25)
    axes[1].hist(ranks, bins=np.arange(0.5, candidates + 1.5), rwidth=0.85)
    axes[1].axhline(len(ranks) / candidates, color="tab:red", linestyle="--", label="chance")
    axes[1].set_xlabel("matched-action rank (1 is best)"); axes[1].legend()
    fig.tight_layout(); fig.savefig(output_dir / "aggregate_metrics.png", dpi=160); plt.close(fig)

    names = list(dimension_metrics)
    values = [dimension_metrics[name]["target_mse_increase"] for name in names]
    fig, axis = plt.subplots(figsize=(16, 5))
    axis.bar(names, values); axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("target MSE increase after shuffling dimension")
    axis.tick_params(axis="x", rotation=60, labelsize=8)
    fig.tight_layout(); fig.savefig(output_dir / "action_dimension_sensitivity.png", dpi=160); plt.close(fig)


def summarize(indices, errors, delta_cosine, change_corr, change_iou):
    indices = np.asarray(indices, dtype=int)
    true_error = errors["fdm_true"][indices]
    shuffled_error = errors["fdm_shuffled"][indices]
    persistence = errors["persistence"][indices]
    return {
        "samples": int(len(indices)),
        "mse_persistence": float(persistence.mean()),
        "mse_fdm_true": float(true_error.mean()),
        "mse_fdm_zero": float(errors["fdm_zero"][indices].mean()),
        "mse_fdm_shuffled": float(shuffled_error.mean()),
        "mse_fdm_negated": float(errors["fdm_negated"][indices].mean()),
        "true_vs_persistence_improvement": float(1.0 - true_error.mean() / persistence.mean()),
        "matched_action_advantage": float((shuffled_error - true_error).mean()),
        "matched_action_win_rate": float((true_error < shuffled_error).mean()),
        "delta_cosine": float(delta_cosine[indices].mean()),
        "change_map_correlation": float(change_corr[indices].mean()),
        "change_top20_iou": float(change_iou[indices].mean()),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.full.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_dir / f"fdm_validation_{checkpoint.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    offset = future_offset(cfg.datasets.vla_data)

    print("Collecting deterministic RoboCasa episode samples...", flush=True)
    samples, tasks = load_samples(cfg, args)
    episode_keys = sorted(set((sample.suite, sample.episode) for sample in samples))
    print(f"Collected {len(samples)} samples from {len(episode_keys)} episodes and {len(tasks)} tasks", flush=True)
    dino, fdm, compatibility = build_models(cfg, checkpoint, device)
    print("Strict DINO/FDM loading passed:", compatibility, flush=True)
    current = encode_dino(dino, [sample.current_images for sample in samples], cfg.framework.dinov3.input_size, device, args.batch_size)
    future = encode_dino(dino, [sample.future_images for sample in samples], cfg.framework.dinov3.input_size, device, args.batch_size)
    actions = torch.from_numpy(np.stack([sample.actions for sample in samples])).float()
    variants = {
        "fdm_true": actions,
        "fdm_zero": torch.zeros_like(actions),
        "fdm_shuffled": actions.roll(1, dims=0),
        "fdm_negated": -actions,
    }
    predictions = {name: predict_fdm(fdm, current, value, device, args.batch_size) for name, value in variants.items()}
    errors = {"persistence": mse_per_sample(current, future)}
    errors.update({name: mse_per_sample(value, future) for name, value in predictions.items()})
    true_delta = future - current
    pred_delta = predictions["fdm_true"] - current
    delta_cosine = cosine_per_sample(pred_delta, true_delta)
    true_change = patch_change(future, current)
    pred_change = patch_change(predictions["fdm_true"], current)
    change_corr = rowwise_correlation(pred_change, true_change)
    change_iou = top_fraction_iou(pred_change, true_change)
    target_cosine = cosine_per_sample(predictions["fdm_true"], future)
    persistence_cosine = cosine_per_sample(current, future)
    delta_norm_ratio = (pred_delta.flatten(1).norm(dim=1) / true_delta.flatten(1).norm(dim=1).clamp_min(1e-8)).numpy()
    action_effect_ratio = (
        (predictions["fdm_true"] - predictions["fdm_shuffled"]).flatten(1).norm(dim=1)
        / true_delta.flatten(1).norm(dim=1).clamp_min(1e-8)
    ).numpy()

    candidates = min(8, len(samples))
    candidate_errors = []
    for shift in range(candidates):
        value = predict_fdm(fdm, current, actions.roll(shift, dims=0), device, args.batch_size)
        candidate_errors.append(mse_per_sample(value, future))
    candidate_errors = np.stack(candidate_errors, axis=1)
    ranks = 1 + (candidate_errors[:, 1:] < candidate_errors[:, :1]).sum(axis=1)

    episode_top1 = []
    for key in episode_keys:
        indices = [i for i, sample in enumerate(samples) if (sample.suite, sample.episode) == key]
        group_errors = []
        for shift in range(len(indices)):
            value = predict_fdm(fdm, current[indices], actions[indices].roll(shift, dims=0), device, args.batch_size)
            group_errors.append(mse_per_sample(value, future[indices]))
        group_errors = np.stack(group_errors, axis=1)
        episode_top1.extend((group_errors[:, 0] <= group_errors.min(axis=1) + 1e-12).tolist())

    dimension_metrics = {}
    for dimension, name in enumerate(action_dimension_names()):
        modified = actions.clone()
        modified[:, :, dimension] = actions.roll(1, dims=0)[:, :, dimension]
        value = predict_fdm(fdm, current, modified, device, args.batch_size)
        dimension_error = mse_per_sample(value, future)
        dimension_metrics[name] = {
            "target_mse": float(dimension_error.mean()),
            "target_mse_increase": float((dimension_error - errors["fdm_true"]).mean()),
            "prediction_mse_vs_true_action": float(mse_per_sample(value, predictions["fdm_true"]).mean()),
            "matched_dimension_win_rate": float((errors["fdm_true"] < dimension_error).mean()),
        }

    overall = summarize(range(len(samples)), errors, delta_cosine, change_corr, change_iou)
    advantage = errors["fdm_shuffled"] - errors["fdm_true"]
    overall.update({
        "mse_fdm_true_bootstrap_95ci": bootstrap_mean_ci(errors["fdm_true"], args.seed),
        "matched_action_advantage_bootstrap_95ci": bootstrap_mean_ci(advantage, args.seed + 1),
        "target_cosine": float(target_cosine.mean()),
        "persistence_target_cosine": float(persistence_cosine.mean()),
        "predicted_delta_norm_ratio": float(delta_norm_ratio.mean()),
        "action_effect_vs_true_delta_norm": float(action_effect_ratio.mean()),
        "global_action_retrieval_candidates": candidates,
        "global_action_retrieval_top1": float((ranks == 1).mean()),
        "global_action_retrieval_chance": 1.0 / candidates,
        "global_action_retrieval_mean_rank": float(ranks.mean()),
        "within_episode_action_retrieval_top1": float(np.mean(episode_top1)),
        "within_episode_action_retrieval_chance": 1.0 / args.steps_per_episode,
    })
    by_task = {
        task: summarize([i for i, sample in enumerate(samples) if sample.suite == task], errors, delta_cosine, change_corr, change_iou)
        for task in tasks
    }

    rows = []
    for index, sample in enumerate(samples):
        row = {
            "index": index, "task": sample.suite, "dataset": sample.dataset_name,
            "episode": sample.episode, "step": sample.step, "language": sample.language,
            "action_abs_mean": float(np.abs(sample.actions).mean()),
            "delta_cosine": float(delta_cosine[index]), "change_map_correlation": float(change_corr[index]),
            "change_top20_iou": float(change_iou[index]), "target_cosine": float(target_cosine[index]),
            "delta_norm_ratio": float(delta_norm_ratio[index]), "action_effect_ratio": float(action_effect_ratio[index]),
            "action_retrieval_rank": int(ranks[index]),
        }
        row.update({f"mse_{name}": float(values[index]) for name, values in errors.items()})
        rows.append(row)
    with (output_dir / "per_sample.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    result = {
        "checkpoint": str(checkpoint), "config": str(config_path),
        "sampling": {
            "datasets": len(tasks), "dataset_names": tasks,
            "episodes_per_dataset": args.episodes_per_dataset, "steps_per_episode": args.steps_per_episode,
            "samples": len(samples), "episodes": len(episode_keys), "seed": args.seed,
            "target_offset_frames": offset, "views": 1, "tokens_per_view": current.shape[1],
        },
        "checkpoint_compatibility": compatibility,
        "action_range": {"min": float(actions.min()), "max": float(actions.max()), "abs_mean": float(actions.abs().mean())},
        "overall": overall, "by_task": by_task, "by_view": {"ego": overall},
        "by_action_dimension": dimension_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    indices = np.linspace(0, len(samples) - 1, min(args.visual_samples, len(samples)), dtype=int).tolist()
    save_feature_maps(output_dir, samples, current, future, predictions["fdm_true"], indices, offset)
    save_interventions(output_dir, samples, current, future, predictions, indices, offset)
    save_aggregate(output_dir, errors, ranks, candidates, dimension_metrics)

    ci = overall["matched_action_advantage_bootstrap_95ci"]
    evidence = sum((
        overall["true_vs_persistence_improvement"] > 0,
        overall["matched_action_advantage"] > 0 and ci[0] > 0,
        overall["global_action_retrieval_top1"] > 1.5 / candidates,
        overall["delta_cosine"] > 0,
        overall["change_map_correlation"] > 0,
    ))
    verdict = "strong" if evidence >= 4 else "partial" if evidence >= 2 else "weak"
    lines = [
        "# RoboCasa FDM checkpoint validation", "",
        f"- Verdict: **{verdict} evidence of learned action-conditioned dynamics**",
        f"- Checkpoint: `{checkpoint}`",
        f"- Samples: {len(samples)} steps from {len(episode_keys)} episodes across {len(tasks)} RoboCasa tasks",
        f"- Target: DINO features at t+{offset} frames", "",
        "## Main quantitative results", "",
        f"- FDM(true action) MSE: {overall['mse_fdm_true']:.8f}",
        f"- Persistence MSE: {overall['mse_persistence']:.8f}",
        f"- Improvement over persistence: {100 * overall['true_vs_persistence_improvement']:.2f}%",
        f"- FDM(shuffled action) MSE: {overall['mse_fdm_shuffled']:.8f}",
        f"- Matched-action advantage: {overall['matched_action_advantage']:.8f}, 95% CI [{ci[0]:.8f}, {ci[1]:.8f}]",
        f"- Matched action beats shuffled: {100 * overall['matched_action_win_rate']:.1f}%",
        f"- Global {candidates}-way retrieval top-1: {100 * overall['global_action_retrieval_top1']:.1f}% (chance {100/candidates:.1f}%)",
        f"- Within-episode retrieval top-1: {100 * overall['within_episode_action_retrieval_top1']:.1f}% (chance {100/args.steps_per_episode:.1f}%)",
        f"- Predicted/true delta cosine: {overall['delta_cosine']:.4f}",
        f"- Spatial change-map correlation: {overall['change_map_correlation']:.4f}",
        f"- Spatial top-20% IoU: {overall['change_top20_iou']:.4f}", "",
        "## Interpretation limits", "",
        "The samples are in-distribution training-mixture episodes. Action interventions establish action dependence, "
        "but unsupported counterfactual physical correctness still requires simulator rollouts or paired counterfactual data.", "",
        "See `aggregate_metrics.png`, `action_interventions.png`, `action_dimension_sensitivity.png`, "
        "`feature_maps/`, `metrics.json`, and `per_sample.csv`.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
