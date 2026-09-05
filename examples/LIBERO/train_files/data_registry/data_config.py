"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(8))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys), # igore state modality for now since some datasets don't have state and we want to be able to use them, can add back later if needed
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.x": "min_max",
                    "state.y": "min_max",
                    "state.z": "min_max",
                    "state.roll": "min_max",
                    "state.pitch": "min_max",
                    "state.yaw": "min_max",
                    "state.gripper": "binary",
                },
            ),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


class Libero4in1VideoFDMDataConfig(Libero4in1DataConfig):
    observation_indices = [-3, -2, -1, 0]
    future_video_keys = [
        "future_video.primary_image",
        "future_video.wrist_image",
    ]
    future_observation_indices = [8]

    def modality_config(self):
        modality_configs = super().modality_config()
        modality_configs["future_video"] = ModalityConfig(
            delta_indices=self.future_observation_indices,
            modality_keys=self.future_video_keys,
        )
        return modality_configs


class Libero4in1RolloutFDMDataConfig(Libero4in1VideoFDMDataConfig):
    observation_indices = [0]
    future_observation_indices = [2, 4, 6, 8]


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
    "libero_franka_video_fdm": Libero4in1VideoFDMDataConfig(),
    "libero_franka_rollout_fdm": Libero4in1RolloutFDMDataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    # Per Proposal A, embodiment_tag now lives as a classvar on each DataConfig.
    # The registry derives ROBOT_TYPE_TO_EMBODIMENT_TAG automatically. Kept as
    # an empty dict for backward compat (it is honored as legacy override).
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}

DATASET_NAMED_MIXTURES["libero_all_video_fdm"] = [
    (dataset_name, weight, "libero_franka_video_fdm")
    for dataset_name, weight, _robot_type in DATASET_NAMED_MIXTURES["libero_all"]
]

DATASET_NAMED_MIXTURES["libero_goal_video_fdm"] = [
    (dataset_name, weight, "libero_franka_video_fdm")
    for dataset_name, weight, _robot_type in DATASET_NAMED_MIXTURES["libero_goal"]
]


DATASET_NAMED_MIXTURES["libero_all_rollout_fdm"] = [
    (dataset_name, weight, "libero_franka_rollout_fdm")
    for dataset_name, weight, _robot_type in DATASET_NAMED_MIXTURES["libero_all"]
]

DATASET_NAMED_MIXTURES["libero_all_phiwam"] = [
    (dataset_name, weight, "libero_franka_rollout_fdm")
    for dataset_name, weight, _robot_type in DATASET_NAMED_MIXTURES["libero_all"]
]
