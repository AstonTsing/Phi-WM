"""Utilities for packing multi-view robot images into temporal image sequences."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _as_numpy_image(image) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3:
        raise ValueError(f"Expected an HWC image, got shape {arr.shape}")
    return arr


def fuse_multiview_by_time(view_sequences: Sequence[Sequence], mode: str = "horizontal") -> list[np.ndarray]:
    """Fuse camera views at each timestep while preserving temporal order.

    Args:
        view_sequences: Sequence of camera streams, each shaped as a sequence of
            HWC images with equal temporal length.
        mode: ``horizontal`` or ``vertical``.

    Returns:
        A list of fused HWC images ordered by timestep.
    """
    if not view_sequences:
        return []

    lengths = [len(seq) for seq in view_sequences]
    if len(set(lengths)) != 1:
        raise ValueError(f"All camera streams must have the same length, got {lengths}")

    axis = {"horizontal": 1, "vertical": 0}.get(mode)
    if axis is None:
        raise ValueError(f"Unsupported multiview fusion mode: {mode}")

    fused = []
    for timestep in range(lengths[0]):
        frames = [_as_numpy_image(seq[timestep]) for seq in view_sequences]
        fused.append(np.concatenate(frames, axis=axis))
    return fused


def pack_multiview_sequence(view_sequences: Sequence[Sequence], mode: str | None = None) -> list[np.ndarray]:
    """Pack camera streams into the model image sequence expected by StarVLA."""
    if mode in (None, "", "none", "None"):
        return [_as_numpy_image(frame) for seq in view_sequences for frame in seq]
    if mode in ("primary_only", "primary", "first_view", "single_view"):
        if not view_sequences:
            return []
        return [_as_numpy_image(frame) for frame in view_sequences[0]]
    if mode in ("horizontal_by_time", "horizontal"):
        return fuse_multiview_by_time(view_sequences, mode="horizontal")
    if mode in ("vertical_by_time", "vertical"):
        return fuse_multiview_by_time(view_sequences, mode="vertical")
    raise ValueError(f"Unsupported pack_multiview mode: {mode}")
