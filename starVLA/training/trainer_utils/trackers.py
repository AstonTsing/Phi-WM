# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""Metric trackers.

Which backends are active comes from ``cfg.trackers`` (a list, or a
comma-separated string so it can be overridden from the command line):

    trackers: [jsonl, tensorboard]

Backends are imported lazily, so a run that only wants TensorBoard does not
need ``wandb`` installed or an API key present.
"""

import json
import numbers
import os
from typing import Dict, Iterable, List

from starVLA.training.trainer_utils.overwatch import initialize_overwatch


logger = initialize_overwatch(__name__)

SUPPORTED = ("jsonl", "tensorboard", "wandb")


def _numeric_only(metrics: Dict) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, numbers.Number) and not isinstance(value, bool)
    }


class JsonlTracker:
    """Appends one JSON object per logged step to ``metrics.jsonl``."""

    name = "jsonl"

    def __init__(self, config):
        self.path = os.path.join(config.output_dir, "metrics.jsonl")
        self._handle = open(self.path, "a", buffering=1)
        logger.info(f"📈 jsonl metrics -> {self.path}")

    def log(self, metrics: Dict, step: int) -> None:
        self._handle.write(json.dumps({"step": step, **_numeric_only(metrics)}) + "\n")

    def close(self) -> None:
        self._handle.close()


class TensorBoardTracker:
    name = "tensorboard"

    def __init__(self, config):
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = os.path.join(config.output_dir, "tensorboard")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        logger.info(f"📈 tensorboard -> {self.log_dir}   (tensorboard --logdir {self.log_dir})")

    def log(self, metrics: Dict, step: int) -> None:
        for key, value in _numeric_only(metrics).items():
            self.writer.add_scalar(key, value, global_step=step)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()


class WandbTracker:
    name = "wandb"

    def __init__(self, config):
        import wandb

        self._wandb = wandb
        wandb.init(
            name=config.run_id,
            dir=os.path.join(config.output_dir, "wandb"),
            project=config.wandb_project,
            entity=config.wandb_entity,
            group="vla-train",
        )

    def log(self, metrics: Dict, step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        self._wandb.finish()


_BACKENDS = {
    "jsonl": JsonlTracker,
    "tensorboard": TensorBoardTracker,
    "wandb": WandbTracker,
}


class TrackerGroup:
    """Fans a metric dict out to every active backend."""

    def __init__(self, trackers: Iterable):
        self.trackers = list(trackers)

    def log(self, metrics: Dict, step: int) -> None:
        for tracker in self.trackers:
            tracker.log(metrics, step)

    def close(self) -> None:
        for tracker in self.trackers:
            try:
                tracker.close()
            except Exception as exc:  # a failing tracker must not fail the run
                logger.warning(f"tracker `{tracker.name}` failed to close: {exc}")

    def __len__(self) -> int:
        return len(self.trackers)


def resolve_tracker_names(config) -> List[str]:
    names = getattr(config, "trackers", None)
    if names is None:
        names = ["wandb"]  # historical default for configs predating this key
    if isinstance(names, str):
        names = names.split(",")
    resolved = []
    for name in names:
        name = str(name).strip().lower()
        if not name or name in resolved:
            continue
        if name not in SUPPORTED:
            raise ValueError(f"unknown tracker `{name}`, expected one of {SUPPORTED}")
        resolved.append(name)
    return resolved


def build_trackers(config, enabled: bool = True) -> TrackerGroup:
    """Instantiate the backends named by ``cfg.trackers`` on the main process."""
    if not enabled:
        return TrackerGroup([])

    trackers = []
    for name in resolve_tracker_names(config):
        try:
            trackers.append(_BACKENDS[name](config))
        except Exception as exc:
            # Losing a tracker is annoying; losing a multi-hour training run
            # because a logging backend could not authenticate is worse.
            logger.warning(f"tracker `{name}` disabled: {type(exc).__name__}: {exc}")
    if not trackers:
        logger.warning("no metric tracker active; metrics will only reach the console log")
    return TrackerGroup(trackers)
