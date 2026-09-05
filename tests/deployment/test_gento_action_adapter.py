import numpy as np
import pytest

from examples.RealRobot.deploy_files.gento_action_adapter import (
    EXPECTED_ACTION_KEYS,
    EXPECTED_VIDEO_KEYS,
    compute_replay_metrics,
    decode_action_response,
    relative_to_absolute_actions,
    validate_server_metadata,
)


def _valid_metadata():
    return {
        "action_mode": "rel",
        "action_chunk_size": 50,
        "action_dim": 8,
        "state_dim": 15,
        "uses_state": True,
        "action_keys": list(EXPECTED_ACTION_KEYS),
        "video_keys": list(EXPECTED_VIDEO_KEYS),
    }


def test_relative_to_absolute_adds_q0_only_to_joints():
    state = np.r_[np.arange(7), 1.0, np.zeros(7)].astype(np.float32)
    rel = np.zeros((50, 8), dtype=np.float32)
    rel[:, :7] = 0.25
    rel[:, 7] = 1.0
    absolute = relative_to_absolute_actions(rel, state)
    expected_joints = np.tile(state[:7] + 0.25, (50, 1))
    np.testing.assert_allclose(absolute[:, :7], expected_joints)
    np.testing.assert_array_equal(absolute[:, 7], 1.0)


def test_non_finite_relative_action_is_rejected():
    rel = np.zeros((50, 8), dtype=np.float32)
    rel[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        relative_to_absolute_actions(rel, np.zeros(15, dtype=np.float32))


def test_relative_to_absolute_rejects_invalid_shapes():
    with pytest.raises(ValueError, match="expected action shape"):
        relative_to_absolute_actions(np.zeros((49, 8), dtype=np.float32), np.zeros(15, dtype=np.float32))
    with pytest.raises(ValueError, match="expected state shape"):
        relative_to_absolute_actions(np.zeros((50, 8), dtype=np.float32), np.zeros(14, dtype=np.float32))


def test_validate_server_metadata_accepts_valid_metadata():
    validate_server_metadata(_valid_metadata())


def test_validate_server_metadata_rejects_action_mode_mismatch():
    metadata = _valid_metadata()
    metadata["action_mode"] = "abs"
    with pytest.raises(ValueError):
        validate_server_metadata(metadata)


def test_validate_server_metadata_rejects_video_key_order():
    metadata = _valid_metadata()
    metadata["video_keys"] = list(reversed(EXPECTED_VIDEO_KEYS))
    with pytest.raises(ValueError):
        validate_server_metadata(metadata)


def test_decode_action_response_returns_squeezed_actions():
    actions = np.zeros((1, 50, 8), dtype=np.float32)
    actions[..., 7] = 0.5
    decoded = decode_action_response({"status": "ok", "data": {"actions": actions}})
    assert decoded.shape == (50, 8)
    np.testing.assert_array_equal(decoded[..., 7], 0.5)


def test_decode_action_response_rejects_error_status():
    with pytest.raises(ValueError):
        decode_action_response({"status": "error", "data": {}})


def test_decode_action_response_rejects_bad_shape():
    actions = np.zeros((1, 49, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="expected action shape"):
        decode_action_response({"status": "ok", "data": {"actions": actions}})


def test_decode_action_response_rejects_non_finite_values():
    actions = np.zeros((1, 50, 8), dtype=np.float32)
    actions[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        decode_action_response({"status": "ok", "data": {"actions": actions}})


def test_decode_action_response_rejects_gripper_out_of_range():
    actions = np.zeros((1, 50, 8), dtype=np.float32)
    actions[..., 7] = 1.5
    with pytest.raises(ValueError, match="gripper"):
        decode_action_response({"status": "ok", "data": {"actions": actions}})


def test_compute_replay_metrics_reports_expected_fields():
    pred_rel = np.zeros((50, 8), dtype=np.float32)
    target_rel = np.zeros((50, 8), dtype=np.float32)
    pred_rel[:, :7] = 0.1
    target_rel[:, :7] = 0.0
    pred_rel[:, 7] = 1.0
    target_rel[:, 7] = 1.0
    pred_abs = np.zeros((50, 8), dtype=np.float32)
    pred_abs[:, :7] = np.arange(7, dtype=np.float32) + 0.1
    pred_abs[1:, :7] += 0.5

    metrics = compute_replay_metrics(pred_rel, target_rel, pred_abs)

    assert metrics["joint_rmse_rad"] == pytest.approx(0.1)
    assert metrics["joint_rmse_deg"] == pytest.approx(np.degrees(0.1))
    assert metrics["per_joint_rmse_rad"].shape == (7,)
    assert metrics["gripper_agreement_rate"] == pytest.approx(1.0)
    assert metrics["max_abs_rel_joint_offset"] == pytest.approx(0.1)
    assert metrics["max_adjacent_abs_target_jump"] == pytest.approx(0.5)


def _valid_replay_arrays():
    pred_rel = np.zeros((50, 8), dtype=np.float32)
    target_rel = np.zeros((50, 8), dtype=np.float32)
    pred_abs = np.zeros((50, 8), dtype=np.float32)
    return pred_rel, target_rel, pred_abs


@pytest.mark.parametrize(
    "mutator",
    [
        lambda arrays: (np.zeros((49, 8), dtype=np.float32), arrays[1], arrays[2]),
        lambda arrays: (arrays[0], np.zeros((49, 8), dtype=np.float32), arrays[2]),
        lambda arrays: (arrays[0], arrays[1], np.zeros((49, 8), dtype=np.float32)),
    ],
)
def test_compute_replay_metrics_rejects_invalid_shapes(mutator):
    arrays = _valid_replay_arrays()
    with pytest.raises(ValueError, match="expected action shape"):
        compute_replay_metrics(*mutator(arrays))


def test_compute_replay_metrics_rejects_non_finite_values():
    pred_rel, target_rel, pred_abs = _valid_replay_arrays()
    pred_rel[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compute_replay_metrics(pred_rel, target_rel, pred_abs)
