#!/usr/bin/env python3
"""Export and plot aligned LIBERO LeWM/original training losses from local logs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRICS = (
    "action_dit_loss",
    "raw_action_loss",
    "mip_action_loss0",
    "mip_action_loss1",
    "fdm_loss_stage0",
    "fdm_loss_stage1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-log",
        default=(
            "/root/tianyi/starVLA/playground/Checkpoints/libero/"
            "libero_qwenoft_mip_dino_fdm_state7_100k/train.log"
        ),
    )
    parser.add_argument(
        "--lewm-metrics",
        default=(
            "/root/tianyi/starVLA/playground/Checkpoints/libero/"
            "libero_acteffect_lewm_state7_100k/metrics.jsonl"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "/root/tianyi/starVLA/playground/Checkpoints/libero/"
            "libero_acteffect_lewm_state7_100k/training_comparison_vs_original"
        ),
    )
    parser.add_argument("--smooth-points", type=int, default=20)
    parser.add_argument("--zoom-start", type=int, default=10000)
    parser.add_argument("--summary-window", type=int, default=5000)
    return parser.parse_args()


def load_jsonl(path: Path) -> dict[int, dict[str, float]]:
    rows = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping incomplete JSONL line {line_number} in {path}")
                continue
            if "step" in row:
                rows[int(row["step"])] = row
    return rows


def load_original_train_log(path: Path) -> dict[int, dict[str, float]]:
    text = path.read_text(errors="replace").replace("\r", "\n")
    starts = list(re.finditer(r"Step\s+(\d+), Loss:", text))
    rows = {}
    for index, match in enumerate(starts):
        step = int(match.group(1))
        end = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        block = text[match.end() : end]
        row = {"step": step}
        for metric in METRICS:
            pattern = re.escape(f"'{metric}'") + r"\s*:\s*([0-9.eE+-]+)"
            value = re.search(pattern, block)
            if value:
                row[metric] = float(value.group(1))
        if all(metric in row for metric in METRICS):
            rows[step] = row
    return rows


def moving_average(values: np.ndarray, points: int) -> np.ndarray:
    points = max(1, min(int(points), len(values)))
    if points == 1:
        return values.copy()
    kernel = np.ones(points, dtype=np.float64) / points
    left = points // 2
    right = points - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def export_aligned_csv(
    path: Path,
    steps: list[int],
    original: dict[int, dict[str, float]],
    lewm: dict[int, dict[str, float]],
) -> None:
    fields = ["step"]
    for metric in METRICS:
        fields.extend(
            [
                f"original_{metric}",
                f"lewm_{metric}",
                f"lewm_minus_original_{metric}",
                f"lewm_relative_change_percent_{metric}",
            ]
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            row = {"step": step}
            for metric in METRICS:
                base = float(original[step][metric])
                candidate = float(lewm[step][metric])
                row[f"original_{metric}"] = base
                row[f"lewm_{metric}"] = candidate
                row[f"lewm_minus_original_{metric}"] = candidate - base
                row[f"lewm_relative_change_percent_{metric}"] = 100.0 * (candidate / base - 1.0)
            writer.writerow(row)


def make_summary(
    path: Path,
    steps: list[int],
    original: dict[int, dict[str, float]],
    lewm: dict[int, dict[str, float]],
    window: int,
) -> list[dict[str, float | int | str]]:
    end = steps[-1]
    start = max(steps[0], end - int(window) + 100)
    selected = [step for step in steps if step >= start]
    rows = []
    for metric in METRICS:
        base = float(np.mean([original[step][metric] for step in selected]))
        candidate = float(np.mean([lewm[step][metric] for step in selected]))
        rows.append(
            {
                "metric": metric,
                "window_start": selected[0],
                "window_end": selected[-1],
                "points": len(selected),
                "original_mean": base,
                "lewm_mean": candidate,
                "lewm_minus_original": candidate - base,
                "lewm_relative_change_percent": 100.0 * (candidate / base - 1.0),
            }
        )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def plot_metrics(
    path: Path,
    steps: list[int],
    original: dict[int, dict[str, float]],
    lewm: dict[int, dict[str, float]],
    smooth_points: int,
    *,
    start_step: int | None,
) -> None:
    selected = [step for step in steps if start_step is None or step >= start_step]
    x = np.asarray(selected)
    figure, axes = plt.subplots(3, 2, figsize=(16, 14), sharex=True)
    axes = axes.ravel()

    for axis, metric in zip(axes, METRICS):
        base = np.asarray([original[step][metric] for step in selected], dtype=np.float64)
        candidate = np.asarray([lewm[step][metric] for step in selected], dtype=np.float64)
        base_smooth = moving_average(base, smooth_points)
        candidate_smooth = moving_average(candidate, smooth_points)

        axis.plot(x, base, color="tab:blue", alpha=0.13, linewidth=0.7)
        axis.plot(x, candidate, color="tab:orange", alpha=0.13, linewidth=0.7)
        axis.plot(x, base_smooth, color="tab:blue", linewidth=2.0, label="Original FDM")
        axis.plot(x, candidate_smooth, color="tab:orange", linewidth=2.0, label="LeWM")
        axis.set_title(metric)
        axis.set_ylabel("loss (lower is better)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
        if start_step is None:
            axis.set_yscale("log")
        else:
            combined = np.concatenate([base_smooth, candidate_smooth])
            low, high = np.quantile(combined, [0.005, 0.995])
            padding = max((high - low) * 0.12, high * 0.02)
            axis.set_ylim(max(0.0, low - padding), high + padding)

    for axis in axes[-2:]:
        axis.set_xlabel("training step")
    mode = "full training, logarithmic y-axis" if start_step is None else f"step >= {start_step:,}"
    figure.suptitle(
        f"LIBERO Original FDM vs LeWM ({mode})\n"
        f"raw logs: faint lines; {smooth_points}-point moving average: solid lines; common end: {steps[-1]:,}",
        fontsize=15,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def write_markdown_summary(path: Path, summary: list[dict[str, float | int | str]], common_end: int) -> None:
    start = int(summary[0]["window_start"])
    end = int(summary[0]["window_end"])
    lines = [
        "# LIBERO LeWM training-loss comparison",
        "",
        f"Aligned local-log data through step {common_end:,}.",
        f"The table uses the most recent common window, steps {start:,}–{end:,}.",
        "Negative relative change means LeWM has a lower loss.",
        "",
        "| Metric | Original mean | LeWM mean | Relative change |",
        "|---|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| `{row['metric']}` | {row['original_mean']:.8f} | {row['lewm_mean']:.8f} | "
            f"{row['lewm_relative_change_percent']:+.2f}% |"
        )
    lines.extend(
        [
            "",
            "Files:",
            "",
            "- `training_losses_full.png`: full aligned history with logarithmic y-axes.",
            "- `training_losses_after_10k.png`: enlarged post-10k history with linear y-axes.",
            "- `aligned_training_metrics.csv`: every common logged step and per-metric differences.",
            "- `recent_window_summary.csv`: recent-window means shown above.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    original_path = Path(args.original_log).expanduser().resolve()
    lewm_path = Path(args.lewm_metrics).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original = load_original_train_log(original_path)
    lewm = load_jsonl(lewm_path)
    steps = sorted(
        step
        for step in original.keys() & lewm.keys()
        if all(metric in original[step] and metric in lewm[step] for metric in METRICS)
    )
    if not steps:
        raise RuntimeError("No aligned training steps with all requested metrics")

    export_aligned_csv(output_dir / "aligned_training_metrics.csv", steps, original, lewm)
    summary = make_summary(
        output_dir / "recent_window_summary.csv",
        steps,
        original,
        lewm,
        args.summary_window,
    )
    plot_metrics(
        output_dir / "training_losses_full.png",
        steps,
        original,
        lewm,
        args.smooth_points,
        start_step=None,
    )
    plot_metrics(
        output_dir / "training_losses_after_10k.png",
        steps,
        original,
        lewm,
        args.smooth_points,
        start_step=args.zoom_start,
    )
    write_markdown_summary(output_dir / "summary.md", summary, steps[-1])

    print(f"Original log entries: {len(original)}")
    print(f"LeWM log entries: {len(lewm)}")
    print(f"Aligned points: {len(steps)}, steps {steps[0]}–{steps[-1]}")
    print(f"Output: {output_dir}")
    for row in summary:
        print(
            f"{row['metric']}: original={row['original_mean']:.8f}, "
            f"LeWM={row['lewm_mean']:.8f}, change={row['lewm_relative_change_percent']:+.2f}%"
        )


if __name__ == "__main__":
    main()
