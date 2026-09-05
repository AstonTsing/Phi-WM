from types import SimpleNamespace

from deployment.model_server.policy_norm_processor import PolicyNormProcessor
from deployment.model_server.policy_wrapper import PolicyServerWrapper


def test_metadata_exposes_gento_action_semantics():
    wrapper = PolicyServerWrapper.__new__(PolicyServerWrapper)
    wrapper._ckpt_path = "/tmp/model.pt"
    wrapper._action_chunk_size = 50
    wrapper._available_unnorm_keys = ["new_embodiment"]
    wrapper._default_unnorm_key = None
    wrapper._multiview_pack = "none"
    wrapper._policy_image_history = 1
    wrapper._uses_state = True
    wrapper._action_mode = "rel"
    wrapper._action_dim = 8
    wrapper._state_dim = 15
    wrapper._video_keys = ["video.overhead", "video.waist", "video.wrist_right"]

    assert wrapper.metadata["action_mode"] == "rel"
    assert wrapper.metadata["action_dim"] == 8
    assert wrapper.metadata["state_dim"] == 15
    assert wrapper.metadata["video_keys"] == wrapper._video_keys


def test_processor_video_keys_preserve_data_config_order():
    processor = PolicyNormProcessor.__new__(PolicyNormProcessor)
    processor._data_config = SimpleNamespace(
        video_keys=["video.overhead", "video.waist", "video.wrist_right"]
    )

    assert processor.video_keys == [
        "video.overhead",
        "video.waist",
        "video.wrist_right",
    ]
