#!/usr/bin/env python3
"""Render RoboCasa RGB, true DINO change, and FDM-predicted change videos."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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
from PIL import Image, ImageDraw, ImageFont
from torchvision.io import read_video

from examples.Robocasa_tabletop.eval_files.analyze_fdm_checkpoint import (
    CHECKPOINT,
    build_models,
    encode_dino,
    future_offset,
    mse_per_sample,
    predict_fdm,
    rowwise_correlation,
)
from starVLA.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--robot-type", default=None)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_actions(dataset, episode: int, count: int, horizon: int, action_dim: int):
    dataset.curr_traj_data = dataset.get_trajectory_data(episode)
    chunks = []
    for step in range(count):
        raw = {
            key: dataset.get_state_or_action(episode, "action", key, step)
            for key in dataset.modality_keys["action"]
        }
        raw.update(
            {
                key: dataset.get_state_or_action(episode, "state", key, step)
                for key in dataset.modality_keys["state"]
            }
        )
        raw = dataset._apply_action_mode(raw)
        transformed = dataset.transforms(raw)
        chunk = np.concatenate(
            [np.asarray(transformed[key], dtype=np.float32) for key in dataset.modality_keys["action"]],
            axis=1,
        )
        if chunk.shape != (horizon, action_dim):
            raise ValueError(f"Expected ({horizon},{action_dim}), got {chunk.shape} at step {step}")
        chunks.append(chunk)
    language_key = dataset.modality_keys["language"][0]
    language = str(dataset.get_language(episode, language_key, 0)[0])
    return torch.from_numpy(np.stack(chunks)), language


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def heatmap(values: np.ndarray, vmax: float, size: int):
    values = np.clip(values / max(vmax, 1e-8), 0.0, 1.0)
    rgb = (plt.get_cmap("magma")(values)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST)


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image: Image.Image):
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        if self.process.wait():
            raise RuntimeError(f"ffmpeg failed: {self.path}")


def labeled_panel(image: Image.Image, title: str, size: int, header: int):
    canvas = Image.new("RGB", (size, size + header), "white")
    canvas.paste(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), (0, header))
    draw = ImageDraw.Draw(canvas)
    title_font = font(18)
    bounds = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((size - bounds[2] + bounds[0]) / 2, 9), title, fill="black", font=title_font)
    return canvas


def render(output_dir, frames, current, future, prediction, offset, fps, episode, language):
    side = int(round(np.sqrt(current.shape[1])))
    true_change = (future - current).float().square().mean(dim=2).sqrt().numpy()
    pred_change = (prediction - current).float().square().mean(dim=2).sqrt().numpy()
    frame_mse = mse_per_sample(prediction, future)
    frame_corr = rowwise_correlation(pred_change, true_change)
    vmax = float(np.quantile(np.concatenate([true_change.ravel(), pred_change.ravel()]), 0.99))
    size, header, footer = 448, 48, 58
    prefix = output_dir / f"episode_{episode:06d}_ego"
    combined = VideoWriter(prefix.with_name(prefix.name + "_rgb_true_pred.mp4"), size * 3, size + header + footer, fps)
    rgb_writer = VideoWriter(prefix.with_name(prefix.name + "_real_rgb.mp4"), size, size, fps)
    true_writer = VideoWriter(prefix.with_name(prefix.name + "_true_dino_change.mp4"), size, size, fps)
    pred_writer = VideoWriter(prefix.with_name(prefix.name + "_predicted_dino_change.mp4"), size, size, fps)
    try:
        for step in range(len(current)):
            rgb = Image.fromarray(frames[step + offset]).convert("RGB")
            true_map = heatmap(true_change[step].reshape(side, side), vmax, size)
            pred_map = heatmap(pred_change[step].reshape(side, side), vmax, size)
            panels = (
                labeled_panel(rgb, f"REAL RGB frame t+{offset}", size, header),
                labeled_panel(true_map, f"TRUE DINO change t -> t+{offset}", size, header),
                labeled_panel(pred_map, "FDM predicted DINO change", size, header),
            )
            canvas = Image.new("RGB", (size * 3, size + header + footer), "white")
            for column, panel in enumerate(panels):
                canvas.paste(panel, (column * size, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (14, size + header + 7),
                f"ego | episode {episode} | t={step}->{step + offset} | "
                f"DINO MSE={frame_mse[step]:.5f} | change corr={frame_corr[step]:.3f} | vmax={vmax:.4f}",
                fill="black", font=font(16),
            )
            draw.text((14, size + header + 32), language, fill=(55, 55, 55), font=font(13))
            combined.write(canvas)
            rgb_writer.write(rgb.resize((size, size), Image.Resampling.BILINEAR))
            true_writer.write(true_map)
            pred_writer.write(pred_map)
    finally:
        combined.close(); rgb_writer.close(); true_writer.close(); pred_writer.close()
    return {
        "frames": int(len(current)), "fps": float(fps), "duration_seconds": float(len(current) / fps),
        "fixed_heatmap_vmax": vmax, "mean_dino_mse": float(frame_mse.mean()),
        "mean_change_map_correlation": float(frame_corr.mean()),
    }


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    run_dir = checkpoint.parents[1]
    config_path = run_dir / "config.full.yaml"
    if not config_path.exists():
        config_path = run_dir / "config.yaml"
    cfg = OmegaConf.load(config_path)
    mixture = DATASET_NAMED_MIXTURES[str(cfg.datasets.vla_data.data_mix)]
    default_dataset, _weight, default_robot_type = mixture[0]
    dataset_name = args.dataset or default_dataset
    robot_type = args.robot_type or next(item[2] for item in mixture if item[0] == dataset_name)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else run_dir / f"fdm_validation_{checkpoint.stem}" / "episode_videos"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_LeRobotSingleDataset(
        Path(str(cfg.datasets.vla_data.data_root_dir)), dataset_name, robot_type,
        delete_pause_frame=bool(cfg.datasets.vla_data.get("delete_pause_frame", False)),
        data_cfg=cfg.datasets.vla_data,
    )
    trajectory_index = dataset.get_trajectory_index(args.episode)
    episode_length = int(dataset.trajectory_lengths[trajectory_index])
    video_key = dataset.modality_keys["video"][0]
    video_path = dataset.get_video_path(args.episode, video_key.replace("video.", ""))
    frames, _audio, info = read_video(str(video_path), pts_unit="sec")
    frames = frames.numpy()
    if len(frames) != episode_length:
        raise ValueError(f"Video has {len(frames)} frames, metadata says {episode_length}")
    fps = float(args.fps or info["video_fps"])
    offset = future_offset(cfg.datasets.vla_data)
    prediction_frames = episode_length - offset
    horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)
    actions, language = load_actions(dataset, args.episode, prediction_frames, horizon, action_dim)
    device = torch.device(args.device)
    dino, fdm, compatibility = build_models(cfg, checkpoint, device)
    tokens = encode_dino(
        dino, [[Image.fromarray(frame)] for frame in frames],
        cfg.framework.dinov3.input_size, device, args.batch_size,
    )
    current, future = tokens[:-offset], tokens[offset:]
    prediction = predict_fdm(fdm, current, actions, device, args.batch_size)
    metrics = render(output_dir, frames, current, future, prediction, offset, fps, args.episode, language)
    metadata = {
        "checkpoint": str(checkpoint), "config": str(config_path), "dataset": dataset_name,
        "robot_type": robot_type, "episode": args.episode, "episode_frames": episode_length,
        "prediction_frames": prediction_frames, "future_offset": offset, "language": language,
        "source_video": str(video_path), "checkpoint_compatibility": compatibility, "ego": metrics,
    }
    path = output_dir / f"episode_{args.episode:06d}_metrics.json"
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(f"Saved episode videos to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
