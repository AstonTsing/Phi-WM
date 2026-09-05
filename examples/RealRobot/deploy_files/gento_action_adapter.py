from __future__ import annotations

from typing import Any, Mapping

import numpy as np

EXPECTED_CHUNK_SIZE = 50
EXPECTED_ACTION_DIM = 8
EXPECTED_STATE_DIM = 15
EXPECTED_ACTION_KEYS = ["action.right_joints", "action.right_gripper"]
EXPECTED_VIDEO_KEYS = ["video.overhead", "video.waist", "video.wrist_right"]


def relative_to_absolute_actions(actions: np.ndarray, raw_state: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    raw_state = np.asarray(raw_state, dtype=np.float32)
    if actions.shape != (EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
        raise ValueError(f"expected action shape (50, 8), got {actions.shape}")
    if raw_state.shape != (EXPECTED_STATE_DIM,):
        raise ValueError(f"expected state shape (15,), got {raw_state.shape}")
    if not np.isfinite(actions).all() or not np.isfinite(raw_state).all():
        raise ValueError("actions and state must be finite")
    absolute = actions.copy()
    absolute[:, :7] += raw_state[None, :7]
    return absolute


def validate_server_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("action_mode") != "rel":
        raise ValueError(f"expected action_mode='rel', got {metadata.get('action_mode')!r}")
    if metadata.get("action_chunk_size") != EXPECTED_CHUNK_SIZE:
        raise ValueError(
            f"expected action_chunk_size={EXPECTED_CHUNK_SIZE}, got {metadata.get('action_chunk_size')!r}"
        )
    if metadata.get("action_dim") != EXPECTED_ACTION_DIM:
        raise ValueError(f"expected action_dim={EXPECTED_ACTION_DIM}, got {metadata.get('action_dim')!r}")
    if metadata.get("state_dim") != EXPECTED_STATE_DIM:
        raise ValueError(f"expected state_dim={EXPECTED_STATE_DIM}, got {metadata.get('state_dim')!r}")
    if metadata.get("uses_state") is not True:
        raise ValueError(f"expected uses_state=True, got {metadata.get('uses_state')!r}")
    if list(metadata.get("action_keys", [])) != EXPECTED_ACTION_KEYS:
        raise ValueError(
            f"expected action_keys={EXPECTED_ACTION_KEYS}, got {metadata.get('action_keys')!r}"
        )
    if list(metadata.get("video_keys", [])) != EXPECTED_VIDEO_KEYS:
        raise ValueError(
            f"expected video_keys={EXPECTED_VIDEO_KEYS}, got {metadata.get('video_keys')!r}"
        )


def decode_action_response(response: Mapping[str, Any]) -> np.ndarray:
    if response.get("status") != "ok":
        raise ValueError(f"expected status='ok', got {response.get('status')!r}")
    data = response.get("data")
    if not isinstance(data, Mapping) or "actions" not in data:
        raise ValueError("response data must contain 'actions'")
    actions = np.asarray(data["actions"], dtype=np.float32)
    if actions.shape != (1, EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
        raise ValueError(f"expected action shape (1, 50, 8), got {actions.shape}")
    actions = actions[0]
    if not np.isfinite(actions).all():
        raise ValueError("actions must be finite")
    gripper = actions[:, 7]
    if np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError("gripper values must be in [0, 1]")
    return actions


def compute_replay_metrics(
    pred_rel: np.ndarray,
    target_rel: np.ndarray,
    pred_abs: np.ndarray,
) -> dict[str, Any]:
    pred_rel = np.asarray(pred_rel, dtype=np.float32)
    target_rel = np.asarray(target_rel, dtype=np.float32)
    pred_abs = np.asarray(pred_abs, dtype=np.float32)

    for array in (pred_rel, target_rel, pred_abs):
        if array.shape != (EXPECTED_CHUNK_SIZE, EXPECTED_ACTION_DIM):
            raise ValueError(f"expected action shape (50, 8), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError("actions must be finite")

    joint_err = pred_rel[:, :7] - target_rel[:, :7]
    per_joint_rmse_rad = np.sqrt(np.mean(joint_err**2, axis=0))
    joint_rmse_rad = float(np.sqrt(np.mean(joint_err**2)))
    joint_rmse_deg = float(np.degrees(joint_rmse_rad))

    gripper_agreement_rate = float(np.mean(pred_rel[:, 7] == target_rel[:, 7]))
    max_abs_rel_joint_offset = float(np.max(np.abs(pred_rel[:, :7])))
    max_adjacent_abs_target_jump = float(np.max(np.abs(np.diff(pred_abs[:, :7], axis=0))))

    return {
        "joint_rmse_rad": joint_rmse_rad,
        "joint_rmse_deg": joint_rmse_deg,
        "per_joint_rmse_rad": per_joint_rmse_rad,
        "gripper_agreement_rate": gripper_agreement_rate,
        "max_abs_rel_joint_offset": max_abs_rel_joint_offset,
        "max_adjacent_abs_target_jump": max_adjacent_abs_target_jump,
    }
