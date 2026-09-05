"""Standalone 50 Hz Gento evaluation loop for a StarVLA policy server."""

from __future__ import annotations

import argparse
import json
import logging
import select
import sys
import termios
import time
import tty
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from deployment.gento.config import GentoConfig, load_config
from deployment.gento.policy_client import GentoPolicyClient, build_example
from deployment.gento.ros_bridge import GentoROSBridge

logger = logging.getLogger(__name__)


def save_observation_snapshot(observation: dict[str, Any], directory: Path) -> None:
    """Save the first cropped RGB views and ordered raw state for inspection."""
    import cv2

    from deployment.gento.policy_client import CAMERA_ORDER, STATE_ORDER

    directory.mkdir(parents=True, exist_ok=True)
    for name in CAMERA_ORDER:
        rgb = np.asarray(observation[name])
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(directory / f"{name}.png"), bgr):
            raise OSError(f"failed to save camera snapshot {name!r}")
    state = {name: float(observation[name]) for name in STATE_ORDER}
    (directory / "state.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


class ChunkPlanner:
    """Run inference off the 50 Hz control thread and reject obsolete results."""

    def __init__(
        self,
        client: GentoPolicyClient,
        task: str,
        *,
        fps: float = 50,
        replan_interval_steps: int = 35,
        max_inference_latency_s: float = 0.9,
    ) -> None:
        self._client = client
        self._task = task
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gento-policy")
        self._future: Future | None = None
        self._future_generation: int | None = None
        self._actions: deque[dict[str, float]] = deque()
        self._generation = 0
        self._submitted_at: float | None = None
        self._actions_since_request = 0
        self._fps = float(fps)
        self._replan_interval_steps = int(replan_interval_steps)
        self._max_inference_latency_s = float(max_inference_latency_s)
        self.enabled = False

    def set_enabled(self, enabled: bool) -> None:
        if self.enabled == enabled:
            return
        self._generation += 1
        self._actions.clear()
        self._actions_since_request = 0
        self.enabled = enabled

    def poll(self, observation: dict[str, Any] | None) -> dict[str, float] | None:
        if (
            self._future is not None
            and not self._future.done()
            and self._submitted_at is not None
            and time.monotonic() - self._submitted_at > self._max_inference_latency_s
        ):
            raise TimeoutError(
                f"inference exceeded {self._max_inference_latency_s:.3f}s freshness limit"
            )
        if self._future is not None and self._future.done():
            future = self._future
            generation = self._future_generation
            submitted_at = self._submitted_at
            self._future = None
            self._future_generation = None
            self._submitted_at = None
            relative, actions = future.result()
            if self.enabled and generation == self._generation:
                if submitted_at is None:
                    raise RuntimeError("planner lost the inference submission timestamp")
                latency = time.monotonic() - submitted_at
                if latency > self._max_inference_latency_s:
                    raise TimeoutError(
                        f"inference result is too stale ({latency:.3f}s > "
                        f"{self._max_inference_latency_s:.3f}s)"
                    )
                skip = max(0, int(latency * self._fps))
                if skip >= len(actions):
                    raise TimeoutError(
                        f"inference consumed the complete {len(actions) / self._fps:.3f}s horizon"
                    )
                self._actions.clear()
                self._actions.extend(actions[skip:])
                self._actions_since_request = 0
                logger.info(
                    "New action chunk: latency=%.3fs skipped=%d remaining=%d "
                    "joint rel max=%.4frad gripper=[%.1f, %.1f]",
                    latency,
                    skip,
                    len(self._actions),
                    float(np.max(np.abs(relative[:, :7]))),
                    float(relative[:, 7].min()),
                    float(relative[:, 7].max()),
                )

        if not self.enabled:
            return None
        if self._future is None and (
            not self._actions
            or self._actions_since_request >= self._replan_interval_steps
        ):
            if observation is None:
                raise ValueError("an observation is required to start a new inference")
            example = build_example(observation, self._task)
            generation = self._generation
            self._future_generation = generation
            self._submitted_at = time.monotonic()
            self._future = self._executor.submit(self._client.infer, example)
        if self._actions:
            self._actions_since_request += 1
            return self._actions.popleft()
        return None

    @property
    def needs_observation(self) -> bool:
        return self.enabled and self._future is None and (
            not self._actions
            or self._actions_since_request >= self._replan_interval_steps
        )

    def close(self) -> None:
        self.set_enabled(False)
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._client.close()


class TerminalKeys:
    """Non-blocking one-character keyboard input with terminal restoration."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._settings = None

    def __enter__(self) -> "TerminalKeys":
        if sys.stdin.isatty():
            self._fd = sys.stdin.fileno()
            self._settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        return self

    def read(self) -> str | None:
        if self._fd is None:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        return sys.stdin.read(1) if readable else None

    def __exit__(self, *_args) -> None:
        if self._fd is not None and self._settings is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._settings)


def run(
    config: GentoConfig,
    *,
    commanding: bool,
    start_enabled: bool = False,
    expected_checkpoint: str | Path | None = None,
    snapshot_dir: Path | None = None,
) -> None:
    """Connect to the policy first, then run the ROS2 control/shadow loop."""
    if commanding and start_enabled:
        raise ValueError("commanding mode must start OFF and be enabled manually with S")
    if commanding and expected_checkpoint is None:
        raise ValueError("commanding mode requires --expected-checkpoint")

    client = GentoPolicyClient(
        config.host,
        config.port,
        config.unnorm_key,
        connect_timeout=config.startup_timeout_s,
        request_timeout=config.request_timeout_s,
        expected_checkpoint=expected_checkpoint,
    )
    planner = ChunkPlanner(
        client,
        config.task,
        fps=config.fps,
        replan_interval_steps=config.replan_interval_steps,
        max_inference_latency_s=config.max_inference_latency_s,
    )
    bridge = GentoROSBridge(config, commanding=commanding)
    snapshot_saved = False
    try:
        bridge.connect()
        planner.set_enabled(start_enabled)
        logger.info(
            "%s mode. Policy starts %s. Press S to toggle; Q or Esc exits.",
            "COMMANDING" if commanding else "SHADOW",
            "ON" if start_enabled else "OFF",
        )
        period = 1.0 / config.fps
        with TerminalKeys() as keyboard:
            while True:
                started = time.perf_counter()
                key = keyboard.read()
                if key in ("q", "Q", "\x1b"):
                    break
                if key in ("s", "S"):
                    planner.set_enabled(not planner.enabled)
                    logger.info("Policy %s", "ON" if planner.enabled else "OFF")

                try:
                    observation = bridge.get_observation() if planner.needs_observation else None
                    if observation is not None and snapshot_dir is not None and not snapshot_saved:
                        save_observation_snapshot(observation, snapshot_dir)
                        snapshot_saved = True
                        logger.info("Saved first policy observation to %s", snapshot_dir)
                    action = planner.poll(observation)
                    if action is not None:
                        if commanding:
                            bridge.send_policy_action(action)
                    elif commanding:
                        bridge.hold()
                except Exception:
                    planner.set_enabled(False)
                    logger.exception("Policy/robot error; policy switched OFF")
                    if commanding:
                        try:
                            bridge.hold()
                        except Exception:
                            logger.exception("Failed to publish feedback hold")

                remaining = period - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        try:
            planner.close()
        finally:
            if commanding:
                try:
                    bridge.hold()
                except Exception:
                    logger.exception("Failed to publish final feedback hold")
            bridge.disconnect()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, type=Path)
    parser.add_argument(
        "--enable-commanding",
        action="store_true",
        help="Allow ROS action publication. Policy still starts OFF until S is pressed.",
    )
    parser.add_argument(
        "--start-enabled",
        action="store_true",
        help="Start inference immediately in shadow mode; rejected with --enable-commanding.",
    )
    parser.add_argument("--host", default=None, help="Override the YAML model-server host")
    parser.add_argument("--port", type=int, default=None, help="Override the YAML model-server port")
    parser.add_argument(
        "--expected-checkpoint",
        type=Path,
        default=None,
        help="Require the server handshake to advertise this exact checkpoint.",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="Save the first cropped camera views and ordered raw state here.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    if args.host is not None or args.port is not None:
        values = dict(config.__dict__)
        if args.host is not None:
            values["host"] = args.host
        if args.port is not None:
            values["port"] = args.port
        config = GentoConfig(**values)
    run(
        config,
        commanding=args.enable_commanding,
        start_enabled=args.start_enabled,
        expected_checkpoint=args.expected_checkpoint,
        snapshot_dir=args.snapshot_dir,
    )


if __name__ == "__main__":
    main()
