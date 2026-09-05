#!/usr/bin/env python3
"""Render real RGB, true DINO change, and FDM-predicted change for one episode."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision.io import read_video

_repo_root = Path(__file__).resolve().parents[3]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from analyze_fdm_checkpoint import build_models, encode_dino, mse_per_sample, predict_fdm, rowwise_correlation
from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=(
            "/root/tianyi/starVLA/playground/Checkpoints/libero/"
            "libero_qwenoft_mip_dino_fdm_state7_100k/checkpoints/steps_60000_pytorch_model.pt"
        ),
    )
    parser.add_argument("--dataset", default="libero_object_no_noops_1.0.0_lerobot")
    parser.add_argument("--robot-type", default="libero_franka_video_fdm")
    parser.add_argument("--episode", type=int, default=41)
    parser.add_argument("--future-offset", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_episode_actions(dataset, episode: int, frame_count: int, horizon: int, future_offset: int):
    dataset.curr_traj_data = dataset.get_trajectory_data(episode)
    actions = []
    for step in range(frame_count - future_offset):
        raw = {
            key: dataset.get_state_or_action(episode, "action", key, step)
            for key in dataset.modality_keys["action"]
        }
        raw = dataset._apply_action_mode(raw)
        transformed = dataset.transforms(raw)
        chunk = np.concatenate(
            [np.asarray(transformed[key], dtype=np.float32) for key in dataset.modality_keys["action"]],
            axis=1,
        )
        if chunk.shape != (horizon, 7):
            raise ValueError(f"Expected action chunk [{horizon},7], got {chunk.shape} at step {step}")
        actions.append(chunk)
    language_key = dataset.modality_keys["language"][0]
    language = dataset.get_language(episode, language_key, 0)[0]
    return torch.from_numpy(np.stack(actions)), language


def load_camera_video(dataset, episode: int, video_key: str):
    path = dataset.get_video_path(episode, video_key.replace("video.", ""))
    frames, _audio, info = read_video(str(path), pts_unit="sec")
    return frames.numpy(), float(info["video_fps"]), path


def font(size: int, times_style: bool = False):
    candidates = []
    if times_style:
        candidates.append(str(Path(matplotlib.get_data_path()) / "fonts/ttf/STIXGeneral.ttf"))
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ]
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def heatmap_image(values: np.ndarray, vmax: float, size: int):
    normalized = np.clip(values / max(vmax, 1e-8), 0.0, 1.0)
    colored = (plt.get_cmap("magma")(normalized)[..., :3] * 255).astype(np.uint8)
    return Image.fromarray(colored).resize((size, size), Image.Resampling.NEAREST)


class VideoWriter:
    def __init__(self, path: Path, width: int, height: int, fps: float):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, image: Image.Image):
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is closed")
        self.process.stdin.write(np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes())

    def close(self):
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {self.path}")


def labeled_panel(image: Image.Image, title: str, panel_size: int, header: int):
    canvas = Image.new("RGB", (panel_size, panel_size + header), "white")
    image = image.convert("RGB").resize((panel_size, panel_size), Image.Resampling.BILINEAR)
    canvas.paste(image, (0, header))
    draw = ImageDraw.Draw(canvas)
    title_font = font(26, times_style=True)
    box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((panel_size - (box[2] - box[0])) / 2, 9), title, fill="black", font=title_font)
    return canvas


def render_view(
    output_dir: Path,
    view_name: str,
    frames: np.ndarray,
    current_tokens: torch.Tensor,
    future_tokens: torch.Tensor,
    predicted_tokens: torch.Tensor,
    future_offset: int,
    fps: float,
    episode: int,
    language: str,
):
    side = int(round(np.sqrt(current_tokens.shape[1])))
    true_change = (future_tokens - current_tokens).float().square().mean(dim=2).sqrt().numpy()
    predicted_change = (predicted_tokens - current_tokens).float().square().mean(dim=2).sqrt().numpy()
    frame_mse = mse_per_sample(predicted_tokens, future_tokens)
    frame_corr = rowwise_correlation(predicted_change, true_change)
    vmax = float(np.quantile(np.concatenate([true_change.ravel(), predicted_change.ravel()]), 0.99))

    panel_size = 448
    header = 48
    footer = 58
    combined_width = panel_size * 3
    combined_height = panel_size + header + footer
    combined_writer = VideoWriter(
        output_dir / f"episode_{episode:06d}_{view_name}_rgb_true_pred.mp4",
        combined_width,
        combined_height,
        fps,
    )
    rgb_writer = VideoWriter(output_dir / f"episode_{episode:06d}_{view_name}_real_rgb.mp4", panel_size, panel_size, fps)
    true_writer = VideoWriter(output_dir / f"episode_{episode:06d}_{view_name}_true_dino_change.mp4", panel_size, panel_size, fps)
    pred_writer = VideoWriter(output_dir / f"episode_{episode:06d}_{view_name}_predicted_dino_change.mp4", panel_size, panel_size, fps)
    body_font = font(17)

    try:
        for step in range(len(current_tokens)):
            rgb = Image.fromarray(frames[step + future_offset]).convert("RGB")
            true_map = heatmap_image(true_change[step].reshape(side, side), vmax, panel_size)
            pred_map = heatmap_image(predicted_change[step].reshape(side, side), vmax, panel_size)
            panels = [
                labeled_panel(rgb, "Real RGB Frame", panel_size, header),
                labeled_panel(true_map, "True Feature Change", panel_size, header),
                labeled_panel(pred_map, "CWM Predicted Change", panel_size, header),
            ]
            canvas = Image.new("RGB", (combined_width, combined_height), "white")
            for column, panel in enumerate(panels):
                canvas.paste(panel, (column * panel_size, 0))
            draw = ImageDraw.Draw(canvas)
            footer_text = (
                f"{view_name} | episode {episode} | t={step} -> {step + future_offset} | "
                f"DINO MSE={frame_mse[step]:.5f} | change-map corr={frame_corr[step]:.3f}"
            )
            draw.text((14, panel_size + header + 7), footer_text, fill="black", font=body_font)
            draw.text((14, panel_size + header + 31), language, fill=(55, 55, 55), font=font(14))
            combined_writer.write(canvas)
            rgb_writer.write(rgb.resize((panel_size, panel_size), Image.Resampling.BILINEAR))
            true_writer.write(true_map)
            pred_writer.write(pred_map)
    finally:
        combined_writer.close()
        rgb_writer.close()
        true_writer.close()
        pred_writer.close()

    return {
        "frames": int(len(current_tokens)),
        "fps": float(fps),
        "duration_seconds": float(len(current_tokens) / fps),
        "fixed_heatmap_vmax": vmax,
        "mean_dino_mse": float(frame_mse.mean()),
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
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else run_dir / f"fdm_validation_{checkpoint.stem}" / "episode_videos"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    dataset = make_LeRobotSingleDataset(
        Path(str(cfg.datasets.vla_data.data_root_dir)),
        args.dataset,
        args.robot_type,
        delete_pause_frame=bool(cfg.datasets.vla_data.get("delete_pause_frame", False)),
        data_cfg=cfg.datasets.vla_data,
    )
    trajectory_index = dataset.get_trajectory_index(args.episode)
    episode_length = int(dataset.trajectory_lengths[trajectory_index])
    video_data = {}
    source_paths = {}
    source_fps = None
    for name, key in (("primary", "video.primary_image"), ("wrist", "video.wrist_image")):
        frames, fps, path = load_camera_video(dataset, args.episode, key)
        if len(frames) != episode_length:
            raise ValueError(f"{name} video has {len(frames)} frames, metadata says {episode_length}")
        video_data[name] = frames
        source_paths[name] = str(path)
        source_fps = fps if source_fps is None else source_fps
        if abs(fps - source_fps) > 1e-6:
            raise ValueError(f"Camera FPS mismatch: {source_fps} vs {fps}")
    output_fps = float(args.fps or source_fps)

    action_horizon = int(cfg.framework.action_model.action_horizon)
    actions, language = load_episode_actions(
        dataset,
        args.episode,
        episode_length,
        action_horizon,
        args.future_offset,
    )
    dino, fdm, compatibility = build_models(cfg, checkpoint, device)
    all_view_tokens = []
    for name in ("primary", "wrist"):
        images = [[Image.fromarray(frame)] for frame in video_data[name]]
        tokens = encode_dino(
            dino,
            images,
            cfg.framework.dinov3.input_size,
            device,
            args.batch_size,
        )
        all_view_tokens.append(tokens)
    all_tokens = torch.cat(all_view_tokens, dim=1)
    current = all_tokens[: -args.future_offset]
    future = all_tokens[args.future_offset :]
    prediction = predict_fdm(fdm, current, actions, device, args.batch_size)

    patches_per_view = all_view_tokens[0].shape[1]
    view_metrics = {}
    for view, name in enumerate(("primary", "wrist")):
        token_slice = slice(view * patches_per_view, (view + 1) * patches_per_view)
        view_metrics[name] = render_view(
            output_dir,
            name,
            video_data[name],
            current[:, token_slice],
            future[:, token_slice],
            prediction[:, token_slice],
            args.future_offset,
            output_fps,
            args.episode,
            language,
        )

    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "dataset": args.dataset,
        "episode": args.episode,
        "episode_frames": episode_length,
        "prediction_frames": len(current),
        "future_offset": args.future_offset,
        "language": language,
        "source_videos": source_paths,
        "checkpoint_compatibility": compatibility,
        "views": view_metrics,
    }
    (output_dir / f"episode_{args.episode:06d}_metrics.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Saved episode videos to {output_dir}")


if __name__ == "__main__":
    main()
