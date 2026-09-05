"""Configuration for the standalone StarVLA Gento evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class CameraConfig:
    name: str
    topic: str
    source_hw: tuple[int, int]
    crop_tlhw: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if not self.name or not self.topic:
            raise ValueError("camera name and topic must be non-empty")
        if len(self.crop_tlhw) != 4:
            raise ValueError(f"camera {self.name!r} crop_tlhw must have four values")
        if len(self.source_hw) != 2 or any(value <= 0 for value in self.source_hw):
            raise ValueError(f"camera {self.name!r} has invalid source_hw {self.source_hw}")
        top, left, height, width = self.crop_tlhw
        if top < 0 or left < 0 or height <= 0 or width <= 0:
            raise ValueError(f"camera {self.name!r} has invalid crop {self.crop_tlhw}")


@dataclass(frozen=True)
class GentoConfig:
    host: str
    port: int
    unnorm_key: str
    task: str
    fps: float
    feedback_topic: str
    right_arm_command_topic: str
    left_arm_command_topic: str
    right_gripper_command_topic: str
    cameras: tuple[CameraConfig, ...]
    initial_right_gripper: float = 0.0
    max_joint_delta_per_step: float = 0.1
    max_feedback_age_s: float = 0.25
    max_camera_age_s: float = 0.5
    startup_timeout_s: float = 10.0
    request_timeout_s: float = 30.0
    replan_interval_steps: int = 35
    max_inference_latency_s: float = 0.9

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("host must be non-empty")
        if not 1 <= self.port <= 65535:
            raise ValueError("port must be in [1, 65535]")
        if not self.unnorm_key or not self.task:
            raise ValueError("unnorm_key and task must be non-empty")
        if self.fps != 50:
            raise ValueError(f"this checkpoint was trained at 50 Hz, got fps={self.fps}")
        if [camera.name for camera in self.cameras] != ["overhead", "waist", "wrist_right"]:
            raise ValueError(
                "cameras must be ordered exactly as overhead, waist, wrist_right"
            )
        if not 0.0 <= self.initial_right_gripper <= 1.0:
            raise ValueError("initial_right_gripper must be in [0, 1]")
        for name in (
            "max_joint_delta_per_step",
            "max_feedback_age_s",
            "max_camera_age_s",
            "startup_timeout_s",
            "request_timeout_s",
            "max_inference_latency_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 1 <= self.replan_interval_steps < 50:
            raise ValueError("replan_interval_steps must be in [1, 49]")


def _required(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing {section}.{key}")
    return mapping[key]


def load_config(path: str | Path) -> GentoConfig:
    """Load and validate a standalone Gento runtime YAML."""
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise TypeError("runtime YAML must contain a mapping")

    server = _required(raw, "server", "root")
    robot = _required(raw, "robot", "root")
    camera_values = _required(raw, "cameras", "root")
    if not isinstance(server, dict) or not isinstance(robot, dict):
        raise TypeError("server and robot sections must be mappings")
    if not isinstance(camera_values, list):
        raise TypeError("cameras must be a list so camera ordering is explicit")

    cameras = tuple(
        CameraConfig(
            name=str(_required(value, "name", "cameras[]")),
            topic=str(_required(value, "topic", "cameras[]")),
            source_hw=tuple(int(item) for item in _required(value, "source_hw", "cameras[]")),
            crop_tlhw=tuple(int(item) for item in _required(value, "crop_tlhw", "cameras[]")),
        )
        for value in camera_values
    )

    return GentoConfig(
        host=str(_required(server, "host", "server")),
        port=int(_required(server, "port", "server")),
        unnorm_key=str(_required(server, "unnorm_key", "server")),
        task=str(_required(server, "task", "server")),
        request_timeout_s=float(server.get("request_timeout_s", 30.0)),
        replan_interval_steps=int(server.get("replan_interval_steps", 35)),
        max_inference_latency_s=float(server.get("max_inference_latency_s", 0.9)),
        fps=float(_required(robot, "fps", "robot")),
        feedback_topic=str(_required(robot, "feedback_topic", "robot")),
        right_arm_command_topic=str(
            _required(robot, "right_arm_command_topic", "robot")
        ),
        left_arm_command_topic=str(
            _required(robot, "left_arm_command_topic", "robot")
        ),
        right_gripper_command_topic=str(
            _required(robot, "right_gripper_command_topic", "robot")
        ),
        initial_right_gripper=float(robot.get("initial_right_gripper", 0.0)),
        max_joint_delta_per_step=float(robot.get("max_joint_delta_per_step", 0.1)),
        max_feedback_age_s=float(robot.get("max_feedback_age_s", 0.25)),
        max_camera_age_s=float(robot.get("max_camera_age_s", 0.5)),
        startup_timeout_s=float(robot.get("startup_timeout_s", 10.0)),
        cameras=cameras,
    )
