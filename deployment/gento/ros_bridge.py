"""Small ROS2 bridge for running Gento directly from a StarVLA checkout.

ROS imports are intentionally delayed until :meth:`GentoROSBridge.connect` so
the pure deployment adapters and their tests do not require a ROS installation.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from deployment.gento.config import GentoConfig
from deployment.gento.policy_client import RIGHT_ACTION_ORDER

logger = logging.getLogger(__name__)

LEFT_JOINTS = tuple(f"Joint{index}_L" for index in range(1, 8))
RIGHT_JOINTS = tuple(f"Joint{index}_R" for index in range(1, 8))


def decode_crop_rgb(
    payload: bytes,
    crop_tlhw: tuple[int, int, int, int],
    source_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Decode one compressed ROS image and apply the training ROI."""
    import cv2

    encoded = np.frombuffer(payload, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("OpenCV failed to decode a compressed camera frame")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if source_hw is not None and rgb.shape[:2] != source_hw:
        raise ValueError(
            f"decoded image shape {rgb.shape[:2]} does not match expected source_hw {source_hw}"
        )
    top, left, height, width = crop_tlhw
    if top + height > rgb.shape[0] or left + width > rgb.shape[1]:
        raise ValueError(
            f"crop {crop_tlhw} exceeds decoded image shape {rgb.shape}"
        )
    return np.ascontiguousarray(rgb[top : top + height, left : left + width])


class GentoROSBridge:
    """Subscribe to Gento feedback/cameras and optionally publish commands."""

    def __init__(self, config: GentoConfig, *, commanding: bool = False) -> None:
        self.config = config
        self.commanding = commanding
        self._lock = threading.RLock()
        self._arm_positions = np.zeros(14, dtype=np.float64)
        self._arm_efforts = np.zeros(14, dtype=np.float64)
        self._feedback_received_at: float | None = None
        self._camera_payloads: dict[str, tuple[bytes, float]] = {}
        self._right_gripper = float(config.initial_right_gripper)
        self._last_right_command: np.ndarray | None = None
        self._node = None
        self._executor = None
        self._spin_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._owns_rclpy = False
        self._left_publisher = None
        self._right_publisher = None
        self._gripper_publisher = None
        self._JointcmdArm = None
        self._Float32 = None

    def _on_feedback(self, message: Any) -> None:
        positions = np.asarray(message.arm_positions, dtype=np.float64)
        efforts = np.asarray(message.arm_efforts, dtype=np.float64)
        if positions.size < 14 or efforts.size < 14:
            logger.error(
                "Ignoring malformed joint feedback: positions=%d efforts=%d",
                positions.size,
                efforts.size,
            )
            return
        if not np.isfinite(positions[:14]).all() or not np.isfinite(efforts[:14]).all():
            logger.error("Ignoring non-finite joint feedback")
            return
        with self._lock:
            self._arm_positions = positions[:14].copy()
            self._arm_efforts = efforts[:14].copy()
            self._feedback_received_at = time.monotonic()

    def _on_camera(self, name: str, message: Any) -> None:
        payload = bytes(message.data)
        if not payload:
            return
        with self._lock:
            self._camera_payloads[name] = (payload, time.monotonic())

    def connect(self) -> None:
        """Create ROS subscriptions and wait for one valid sample from every input."""
        import rclpy
        from marvin_msgs.msg import JointcmdArm, Jointfeedback
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import Float32

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._JointcmdArm = JointcmdArm
        self._Float32 = Float32
        self._node = Node("starvla_gento_workstation")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(
            Jointfeedback,
            self.config.feedback_topic,
            self._on_feedback,
            qos,
        )
        for camera in self.config.cameras:
            self._node.create_subscription(
                CompressedImage,
                camera.topic,
                lambda message, name=camera.name: self._on_camera(name, message),
                qos,
            )

        if self.commanding:
            self._left_publisher = self._node.create_publisher(
                JointcmdArm, self.config.left_arm_command_topic, qos
            )
            self._right_publisher = self._node.create_publisher(
                JointcmdArm, self.config.right_arm_command_topic, qos
            )
            self._gripper_publisher = self._node.create_publisher(
                Float32, self.config.right_gripper_command_topic, qos
            )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._stop.clear()
        self._spin_thread = threading.Thread(
            target=self._spin,
            daemon=True,
            name="starvla-gento-ros2",
        )
        self._spin_thread.start()
        self._wait_until_ready()
        logger.info(
            "Gento ROS bridge ready (commanding=%s, cameras=%s)",
            self.commanding,
            [camera.name for camera in self.config.cameras],
        )
        logger.warning(
            "No gripper feedback is available in the validated Gento interface; "
            "state gripper_R starts at %.1f and then tracks this process's last command.",
            self.config.initial_right_gripper,
        )

    def _spin(self) -> None:
        while not self._stop.is_set():
            try:
                self._executor.spin_once(timeout_sec=0.02)
            except Exception:
                if not self._stop.is_set():
                    logger.exception("ROS2 executor error")
                    time.sleep(0.05)

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self.config.startup_timeout_s
        camera_names = {camera.name for camera in self.config.cameras}
        while time.monotonic() < deadline:
            with self._lock:
                ready = (
                    self._feedback_received_at is not None
                    and camera_names.issubset(self._camera_payloads)
                )
            if ready:
                return
            time.sleep(0.05)
        with self._lock:
            missing_cameras = sorted(camera_names - self._camera_payloads.keys())
            feedback_missing = self._feedback_received_at is None
        raise TimeoutError(
            "Gento inputs were not ready before timeout: "
            f"feedback_missing={feedback_missing}, missing_cameras={missing_cameras}"
        )

    def get_observation(self) -> dict[str, Any]:
        """Return one coherent inference snapshot and reject stale inputs."""
        now = time.monotonic()
        with self._lock:
            positions = self._arm_positions.copy()
            efforts = self._arm_efforts.copy()
            feedback_at = self._feedback_received_at
            payloads = dict(self._camera_payloads)
            gripper = self._right_gripper
        if feedback_at is None or now - feedback_at > self.config.max_feedback_age_s:
            age = math.inf if feedback_at is None else now - feedback_at
            raise RuntimeError(f"joint feedback is stale ({age:.3f}s)")

        observation: dict[str, Any] = {}
        for index, name in enumerate(RIGHT_JOINTS):
            observation[name] = float(positions[7 + index])
            observation[f"{name}_effort"] = float(efforts[7 + index])
        observation["gripper_R"] = float(gripper)

        for camera in self.config.cameras:
            if camera.name not in payloads:
                raise RuntimeError(f"camera {camera.name!r} has no frame")
            payload, received_at = payloads[camera.name]
            age = now - received_at
            if age > self.config.max_camera_age_s:
                raise RuntimeError(f"camera {camera.name!r} is stale ({age:.3f}s)")
            observation[camera.name] = decode_crop_rgb(
                payload,
                camera.crop_tlhw,
                camera.source_hw,
            )
        return observation

    def _require_fresh_feedback(self) -> None:
        with self._lock:
            received_at = self._feedback_received_at
        age = math.inf if received_at is None else time.monotonic() - received_at
        if age > self.config.max_feedback_age_s:
            raise RuntimeError(f"refusing to command with stale joint feedback ({age:.3f}s)")

    def _feedback_hold(self) -> tuple[np.ndarray, np.ndarray, float]:
        with self._lock:
            left = self._arm_positions[:7].copy()
            right = self._arm_positions[7:14].copy()
            gripper = float(self._right_gripper)
        return left, right, gripper

    def _publish(self, left: np.ndarray, right: np.ndarray, gripper: float) -> None:
        if not self.commanding:
            return
        if self._node is None:
            raise RuntimeError("ROS bridge is not connected")
        if not (
            np.isfinite(left).all()
            and np.isfinite(right).all()
            and math.isfinite(gripper)
            and 0.0 <= gripper <= 1.0
        ):
            raise ValueError("refusing to publish a non-finite or invalid command")
        stamp = self._node.get_clock().now().to_msg()
        left_message = self._JointcmdArm()
        left_message.header.stamp = stamp
        left_message.positions = left.tolist()
        right_message = self._JointcmdArm()
        right_message.header.stamp = stamp
        right_message.positions = right.tolist()
        gripper_message = self._Float32()
        gripper_message.data = float(gripper)
        self._left_publisher.publish(left_message)
        self._right_publisher.publish(right_message)
        self._gripper_publisher.publish(gripper_message)

    def hold(self) -> None:
        """Echo current joint feedback; used whenever policy execution is off."""
        self._require_fresh_feedback()
        left, right, gripper = self._feedback_hold()
        self._last_right_command = right.copy()
        self._publish(left, right, gripper)

    def send_policy_action(self, action: Mapping[str, float]) -> dict[str, float]:
        """Clamp and publish one absolute right-arm action row."""
        self._require_fresh_feedback()
        if tuple(action) != RIGHT_ACTION_ORDER:
            raise ValueError(
                f"action keys must be ordered exactly as {RIGHT_ACTION_ORDER}, got {tuple(action)}"
            )
        values = np.asarray([action[name] for name in RIGHT_ACTION_ORDER], dtype=np.float64)
        if values.shape != (8,) or not np.isfinite(values).all():
            raise ValueError("policy action must contain eight finite values")
        if not 0.0 <= values[7] <= 1.0:
            raise ValueError("right gripper command must be in [0, 1]")

        left, feedback_right, _ = self._feedback_hold()
        reference = feedback_right if self._last_right_command is None else self._last_right_command
        delta = np.clip(
            values[:7] - reference,
            -self.config.max_joint_delta_per_step,
            self.config.max_joint_delta_per_step,
        )
        right = reference + delta
        gripper = float(values[7])
        self._publish(left, right, gripper)
        self._last_right_command = right.copy()
        with self._lock:
            self._right_gripper = gripper
        return {
            **{name: float(value) for name, value in zip(RIGHT_JOINTS, right, strict=True)},
            "gripper_R": gripper,
        }

    def disconnect(self) -> None:
        self._stop.set()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        if self._executor is not None:
            try:
                self._executor.remove_node(self._node)
                self._executor.shutdown()
            except Exception:
                logger.exception("Failed to stop ROS2 executor cleanly")
        if self._node is not None:
            self._node.destroy_node()
        if self._owns_rclpy:
            import rclpy

            rclpy.try_shutdown()
        self._node = None
        logger.info("Gento ROS bridge disconnected")
