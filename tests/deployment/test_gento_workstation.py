from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from deployment.gento.config import CameraConfig, GentoConfig, load_config
from deployment.gento.eval import ChunkPlanner, run, save_observation_snapshot
from deployment.gento.policy_client import (
    CAMERA_ORDER,
    RIGHT_ACTION_ORDER,
    STATE_ORDER,
    action_rows,
    build_example,
)
from deployment.gento.ros_bridge import GentoROSBridge, decode_crop_rgb


def _observation() -> dict:
    observation = {
        name: np.zeros((20, 30, 3), dtype=np.uint8)
        for name in CAMERA_ORDER
    }
    observation.update({name: float(index) for index, name in enumerate(STATE_ORDER)})
    return observation


def test_checked_in_gento_config_matches_training_contract() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "deployment/gento/config/gento_insert_usb.yaml"
    )
    config = load_config(path)

    assert config.fps == 50
    assert [camera.name for camera in config.cameras] == list(CAMERA_ORDER)
    assert [camera.crop_tlhw for camera in config.cameras] == [
        (195, 375, 285, 265),
        (60, 580, 560, 580),
        (240, 290, 480, 750),
    ]
    assert config.task == "Pull out the left‑side USB plug, then insert it into the right side."


def test_config_rejects_reordered_cameras() -> None:
    cameras = tuple(
        CameraConfig(name=name, topic=f"/{name}", source_hw=(1, 1), crop_tlhw=(0, 0, 1, 1))
        for name in reversed(CAMERA_ORDER)
    )
    with pytest.raises(ValueError, match="ordered exactly"):
        GentoConfig(
            host="127.0.0.1",
            port=10093,
            unnorm_key="new_embodiment",
            task="task",
            fps=50,
            feedback_topic="/feedback",
            right_arm_command_topic="/right",
            left_arm_command_topic="/left",
            right_gripper_command_topic="/gripper",
            cameras=cameras,
        )


def test_build_example_preserves_camera_and_state_order() -> None:
    observation = _observation()
    for index, name in enumerate(CAMERA_ORDER):
        observation[name].fill(index)

    example = build_example(observation, "task")

    assert [int(image[0, 0, 0]) for image in example["image"]] == [0, 1, 2]
    np.testing.assert_array_equal(
        example["state"],
        np.arange(15, dtype=np.float32)[None],
    )


def test_save_observation_snapshot_writes_three_views_and_state(tmp_path) -> None:
    observation = _observation()
    save_observation_snapshot(observation, tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {
        "overhead.png",
        "waist.png",
        "wrist_right.png",
        "state.json",
    }


def test_action_rows_maps_all_dimensions_in_order() -> None:
    actions = np.tile(np.arange(8, dtype=np.float32), (50, 1))
    rows = action_rows(actions)

    assert tuple(rows[0]) == RIGHT_ACTION_ORDER
    assert list(rows[0].values()) == list(range(8))


def test_decode_crop_rgb_converts_bgr_and_uses_tlhw() -> None:
    bgr = np.zeros((8, 10, 3), dtype=np.uint8)
    bgr[2:6, 3:8] = (10, 20, 30)
    ok, encoded = cv2.imencode(".png", bgr)
    assert ok

    rgb = decode_crop_rgb(encoded.tobytes(), (2, 3, 4, 5), (8, 10))

    assert rgb.shape == (4, 5, 3)
    np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
    assert rgb.flags.c_contiguous

    with pytest.raises(ValueError, match="source_hw"):
        decode_crop_rgb(encoded.tobytes(), (2, 3, 4, 5), (9, 10))


def _runtime_config() -> GentoConfig:
    return GentoConfig(
        host="127.0.0.1",
        port=10093,
        unnorm_key="new_embodiment",
        task="task",
        fps=50,
        feedback_topic="/feedback",
        right_arm_command_topic="/right",
        left_arm_command_topic="/left",
        right_gripper_command_topic="/gripper",
        cameras=tuple(
            CameraConfig(name=name, topic=f"/{name}", source_hw=(1, 1), crop_tlhw=(0, 0, 1, 1))
            for name in CAMERA_ORDER
        ),
    )


def test_ros_bridge_clamps_absolute_joint_target_per_control_step() -> None:
    bridge = GentoROSBridge(_runtime_config(), commanding=False)
    bridge._on_feedback(
        SimpleNamespace(
            arm_positions=[0.0] * 14,
            arm_efforts=[0.0] * 14,
        )
    )
    command = {
        **{name: 0.25 for name in RIGHT_ACTION_ORDER[:-1]},
        "gripper_R": 1.0,
    }

    first = bridge.send_policy_action(command)
    second = bridge.send_policy_action(command)

    np.testing.assert_allclose(list(first.values())[:7], 0.1)
    np.testing.assert_allclose(list(second.values())[:7], 0.2)
    assert first["gripper_R"] == 1.0


def test_ros_bridge_rejects_commands_without_fresh_feedback() -> None:
    bridge = GentoROSBridge(_runtime_config(), commanding=False)
    command = {
        **{name: 0.0 for name in RIGHT_ACTION_ORDER[:-1]},
        "gripper_R": 0.0,
    }
    with pytest.raises(RuntimeError, match="stale joint feedback"):
        bridge.send_policy_action(command)


def test_commanding_mode_requires_exact_checkpoint_handshake() -> None:
    with pytest.raises(ValueError, match="requires --expected-checkpoint"):
        run(_runtime_config(), commanding=True)


class _FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.closed = False

    def infer(self, example):
        self.calls += 1
        relative = np.zeros((50, 8), dtype=np.float32)
        actions = [
            {name: float(index) for index, name in enumerate(RIGHT_ACTION_ORDER)}
            for _ in range(50)
        ]
        return relative, actions

    def close(self):
        self.closed = True


def test_chunk_planner_runs_inference_off_thread_and_emits_50_rows() -> None:
    client = _FakeClient()
    planner = ChunkPlanner(client, "task")
    planner.set_enabled(True)
    try:
        assert planner.poll(_observation()) is None
        deadline = __import__("time").monotonic() + 2
        actions = []
        while len(actions) < 50 and __import__("time").monotonic() < deadline:
            observation = _observation() if planner.needs_observation else None
            action = planner.poll(observation)
            if action is not None:
                actions.append(action)
        assert len(actions) == 50
        assert client.calls >= 1
    finally:
        planner.close()
    assert client.closed
