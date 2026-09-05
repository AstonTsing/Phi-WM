"""Pure StarVLA protocol and Gento observation adapters."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.RealRobot.deploy_files.gento_action_adapter import (
    EXPECTED_ACTION_KEYS,
    EXPECTED_STATE_DIM,
    EXPECTED_VIDEO_KEYS,
    decode_action_response,
    relative_to_absolute_actions,
    validate_server_metadata,
)

CAMERA_ORDER = ("overhead", "waist", "wrist_right")
STATE_ORDER = (
    *(f"Joint{index}_R" for index in range(1, 8)),
    "gripper_R",
    *(f"Joint{index}_R_effort" for index in range(1, 8)),
)
RIGHT_ACTION_ORDER = (*((f"Joint{index}_R") for index in range(1, 8)), "gripper_R")
EXPECTED_STATE_KEYS = (
    "state.right_joints",
    "state.right_gripper",
    "state.right_joint_efforts",
)


def build_example(observation: Mapping[str, Any], task: str) -> dict[str, Any]:
    """Build the exact three-view, 15-state request used during training."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")

    images: list[np.ndarray] = []
    for name in CAMERA_ORDER:
        if name not in observation:
            raise KeyError(f"missing camera {name!r}")
        image = np.asarray(observation[name])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(
                f"camera {name!r} must be RGB uint8 HWC, got {image.dtype} {image.shape}"
            )
        images.append(np.ascontiguousarray(image))

    values = []
    for name in STATE_ORDER:
        if name not in observation:
            raise KeyError(f"missing state value {name!r}")
        value = float(observation[name])
        if not np.isfinite(value):
            raise ValueError(f"state value {name!r} must be finite")
        values.append(value)
    state = np.asarray([values], dtype=np.float32)
    if state.shape != (1, EXPECTED_STATE_DIM):
        raise AssertionError(f"unexpected state shape {state.shape}")
    return {"image": images, "lang": task, "state": state}


def action_rows(actions: np.ndarray) -> list[dict[str, float]]:
    """Convert an absolute ``(50, 8)`` chunk to named robot commands."""
    actions = np.asarray(actions, dtype=np.float32)
    if actions.shape != (50, 8) or not np.isfinite(actions).all():
        raise ValueError(f"expected finite action shape (50, 8), got {actions.shape}")
    return [
        {name: float(value) for name, value in zip(RIGHT_ACTION_ORDER, row, strict=True)}
        for row in actions
    ]


class GentoPolicyClient:
    """Thin client that validates the checkpoint contract before inference."""

    def __init__(
        self,
        host: str,
        port: int,
        unnorm_key: str,
        *,
        connect_timeout: float = 10.0,
        request_timeout: float = 30.0,
        expected_checkpoint: str | Path | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._unnorm_key = unnorm_key
        self._connect_timeout = connect_timeout
        self._request_timeout = request_timeout
        self._expected_checkpoint = expected_checkpoint
        self._client: WebsocketClientPolicy | None = None
        self.metadata: dict[str, Any] = {}
        self._connect()

    def _connect(self) -> None:
        client = WebsocketClientPolicy(
            host=self._host,
            port=self._port,
            connect_timeout=self._connect_timeout,
            request_timeout=self._request_timeout,
        )
        self._client = client
        metadata = client.get_server_metadata()
        try:
            validate_server_metadata(metadata)
            if self._expected_checkpoint is not None:
                actual_checkpoint = metadata.get("ckpt_path")
                if not isinstance(actual_checkpoint, str) or not actual_checkpoint:
                    raise ValueError("server metadata has no valid ckpt_path")
                actual = Path(actual_checkpoint).expanduser().resolve()
                expected = Path(self._expected_checkpoint).expanduser().resolve()
                if actual != expected:
                    raise ValueError(
                        f"server checkpoint mismatch: expected {expected}, got {actual}"
                    )
            if list(metadata.get("state_keys", ())) != list(EXPECTED_STATE_KEYS):
                raise ValueError(
                    f"expected state_keys={list(EXPECTED_STATE_KEYS)!r}, "
                    f"got {metadata.get('state_keys')!r}"
                )
            if list(metadata.get("action_keys", ())) != list(EXPECTED_ACTION_KEYS):
                raise ValueError("server action key order does not match Gento training")
            if list(metadata.get("video_keys", ())) != list(EXPECTED_VIDEO_KEYS):
                raise ValueError("server camera order does not match Gento training")
            available = metadata.get("available_unnorm_keys", ())
            if self._unnorm_key not in available:
                raise ValueError(
                    f"unnorm_key {self._unnorm_key!r} is unavailable; server has {list(available)!r}"
                )
        except BaseException:
            self.close()
            raise
        self.metadata = dict(metadata)

    def infer(self, example: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Return relative model output and q0-anchored absolute robot commands."""
        request_id = uuid.uuid4().hex
        if self._client is None:
            self._connect()
        try:
            response = self._client.predict_action(
                {
                    "type": "infer",
                    "request_id": request_id,
                    "payload": {
                        "examples": [example],
                        "unnorm_key": self._unnorm_key,
                    },
                }
            )
        except BaseException:
            self.close()
            raise
        if response.get("request_id") != request_id:
            raise RuntimeError("StarVLA response request_id mismatch")
        relative = decode_action_response(response)
        absolute = relative_to_absolute_actions(relative, example["state"][0])
        return relative, action_rows(absolute)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
