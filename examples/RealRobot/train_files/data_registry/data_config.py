"""Gento real-robot data configuration and dataset mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform


class GentoRightArmVideoFDMDataConfig:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    video_keys = [
        "video.overhead",
        "video.waist",
        "video.wrist_right",
    ]
    state_keys = [
        "state.right_joints",
        "state.right_gripper",
        "state.right_joint_efforts",
    ]
    action_keys = [
        "action.right_joints",
        "action.right_gripper",
    ]
    action_key_dims = {
        "action.right_joints": 7,
        "action.right_gripper": 1,
    }
    state_key_dims = {
        "state.right_joints": 7,
        "state.right_gripper": 1,
        "state.right_joint_efforts": 7,
    }
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(50))
    future_video_keys = [
        "future_video.overhead",
        "future_video.waist",
        "future_video.wrist_right",
    ]
    future_observation_indices = [50]

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
            "future_video": ModalityConfig(
                delta_indices=self.future_observation_indices,
                modality_keys=self.future_video_keys,
            ),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.right_joints": "min_max",
                    "state.right_gripper": "binary",
                    "state.right_joint_efforts": "min_max",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    # Relative joint targets are narrow and heavy-tailed, so min_max
                    # would squeeze 99% of the mass into ~20% of [-1, 1].
                    "action.right_joints": "mean_std",
                    "action.right_gripper": "binary",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "gento_right_arm_video_fdm": GentoRightArmVideoFDMDataConfig(),
}


ROBOT_TYPE_TO_EMBODIMENT_TAG = {}


DATASET_NAMED_MIXTURES = {
    "gento_insert_usb_video_fdm": [
        (
            "gento_insert_usb_0831_0901_right_crop_mp4",
            1.0,
            "gento_right_arm_video_fdm",
        ),
    ],
}
