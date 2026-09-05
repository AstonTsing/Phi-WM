import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from collections import deque
from typing import Dict

import imageio
import numpy as np
import torch
import tqdm
import tyro

os.environ["TOKENIZERS_PARALLELISM"] = "false"
from examples.LIBERO.eval_files.model2libero_interface import ModelClient


def _patch_torch_load_for_liberoplus_init_states() -> None:
    original_load = torch.load

    def load_with_trusted_liberoplus_defaults(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = load_with_trusted_liberoplus_defaults


_patch_torch_load_for_liberoplus_init_states()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


def _pack_eval_multiview(primary_image: np.ndarray, wrist_image: np.ndarray, pack_mode: str) -> np.ndarray:
    if pack_mode in ("primary_only", "primary", "first_view", "single_view"):
        return primary_image
    if pack_mode in ("horizontal_by_time", "horizontal"):
        return np.concatenate([primary_image, wrist_image], axis=1)
    if pack_mode in ("vertical_by_time", "vertical"):
        return np.concatenate([primary_image, wrist_image], axis=0)
    if pack_mode in ("none", "None", "", None):
        return primary_image
    raise ValueError(f"Unsupported multiview_pack for LIBERO eval: {pack_mode}")


def _build_policy_image_history(
    image_history: deque,
    primary_image: np.ndarray,
    wrist_image: np.ndarray,
    history_len: int,
    pack_mode: str,
) -> list[np.ndarray]:
    if pack_mode in ("none", "None", "", None):
        return [primary_image, wrist_image]
    fused_current = _pack_eval_multiview(primary_image, wrist_image, pack_mode)
    image_history.append(fused_current)
    while len(image_history) < history_len:
        image_history.appendleft(fused_current.copy())
    return list(image_history)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 1  # LIBERO-plus uses many task variants; default to one rollout per task.
    start_idx: int = -1
    end_idx: int = -1

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    log_path: str = "experiments/libero/logs"
    output_dir: str = ""
    save_videos: bool = False

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""
    unnorm_key: str = "franka"
    image_history: int = -1
    multiview_pack: str = "auto"
    use_state: bool = True

    post_process_action: bool = True

    job_name: str = "test"


def eval_libero(args: Args) -> None:
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    if args.start_idx < 0:
        args.start_idx = 0
    if args.end_idx < 0 or args.end_idx > num_tasks_in_suite:
        args.end_idx = num_tasks_in_suite
    if args.start_idx >= args.end_idx:
        raise ValueError(
            f"Invalid task slice [{args.start_idx}, {args.end_idx}) for {args.task_suite_name} "
            f"with {num_tasks_in_suite} tasks"
        )

    if args.output_dir:
        args.log_path = os.path.join(args.output_dir, "logs", args.task_suite_name)
        args.video_out_path = os.path.join(args.output_dir, "videos", args.task_suite_name)

    pathlib.Path(args.log_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client_model = ModelClient(
        host=args.host,
        port=args.port,
        unnorm_key=args.unnorm_key,
    )
    server_meta = client_model.server_metadata
    server_uses_state = bool(server_meta.get("uses_state", False))
    eval_uses_state = bool(args.use_state and server_uses_state)
    if args.use_state and not server_uses_state:
        logging.warning("--args.use-state was set, but the policy server reports uses_state=False; state will not be sent.")
    eval_multiview_pack = args.multiview_pack
    if eval_multiview_pack == "auto":
        eval_multiview_pack = server_meta.get("multiview_pack", "none")
    eval_image_history = args.image_history
    if eval_image_history < 0:
        eval_image_history = int(server_meta.get(
            "policy_image_history",
            4 if eval_multiview_pack not in ("none", "None", "", None) else 0,
        ))
    logging.info(
        f"Policy image adapter: multiview_pack={eval_multiview_pack}, "
        f"image_history={eval_image_history}, uses_state={eval_uses_state}"
    )

    disturb_res: Dict[str, Dict[str, int]] = {}
    LIBERO_HOME = os.environ.get("LIBERO_HOME", "path_to_LIBERO-plus_home")
    with open(os.path.join(LIBERO_HOME, "libero/libero/benchmark/task_classification.json")) as f:
        TASK_MAPPING = json.load(f)[args.task_suite_name]
    ID2CATEGORY = {}
    for item in TASK_MAPPING:
        category = item["category"]
        item_name = item["name"]
        ID2CATEGORY[item["id"]] = (category, item_name)
        if category not in disturb_res:
            disturb_res[category] = {"total_count": 0, "success_count": 0}

    # Start evaluation

    total_episodes, total_successes = 0, 0
    logging.info(
        f"Processing {args.task_suite_name} task slice [{args.start_idx}, {args.end_idx}) "
        f"of {num_tasks_in_suite}"
    )
    for task_id in tqdm.tqdm(range(args.start_idx, args.end_idx)):

        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):

            logging.info(f"\nTask: {task_description}")

            # Reset environment
            client_model.reset(task_description=task_description)  # Reset the client connection
            env.reset()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            full_actions = []
            policy_image_history = deque(maxlen=max(eval_image_history, 1))

            logging.info(f"Starting episode {task_episodes + 1}...")
            step = 0

            # full_actions = np.load("./debug/action.npy")

            while t < max_steps + args.num_steps_wait:

                # try:
                # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                # and we need to wait for them to fall
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # IMPORTANT: rotate 180 degrees to match train preprocessing
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

                # Save preprocessed image for replay video
                replay_images.append(img)

                gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
                gripper_state = np.asarray([float(gripper_qpos.mean() > 0.0)], dtype=np.float32)
                state = np.concatenate(
                    (
                        obs["robot0_eef_pos"],
                        _quat2axisangle(obs["robot0_eef_quat"]),
                        gripper_state,
                    )
                ).astype(np.float32)
                if state.shape != (7,):
                    raise ValueError(f"Expected LIBERO-plus state shape (7,), got {state.shape}")

                policy_images = _build_policy_image_history(
                    policy_image_history,
                    img,
                    wrist_img,
                    eval_image_history,
                    eval_multiview_pack,
                )

                example_dict = {
                    "image": policy_images,
                    "vggt_image": [img, wrist_img],
                    "lang": str(task_description),
                }
                if eval_uses_state:
                    example_dict["state"] = np.expand_dims(state, axis=0)

                start_time = time.time()

                # response = client_model.step(example=example_dict)
                response = client_model.step(example=example_dict, step=step)

                end_time = time.time()
                # print(f"time: {end_time - start_time}")

                # #
                raw_action = response["raw_action"]

                world_vector_delta = np.asarray(raw_action.get("world_vector"), dtype=np.float32).reshape(-1)
                rotation_delta = np.asarray(raw_action.get("rotation_delta"), dtype=np.float32).reshape(-1)
                open_gripper = np.asarray(raw_action.get("open_gripper"), dtype=np.float32).reshape(-1)
                gripper = _binarize_gripper_open(open_gripper)

                if not (world_vector_delta.size == 3 and rotation_delta.size == 3 and open_gripper.size == 1):
                    logging.warning(
                        f"Unexpected action sizes: "
                        f"wv={world_vector_delta.shape}, rot={rotation_delta.shape}, grip={gripper.shape}. "
                        f"Falling back to LIBERO_DUMMY_ACTION."
                    )
                    raise ValueError(
                        f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                        f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                    )
                else:
                    delta_action = np.concatenate([world_vector_delta, rotation_delta, gripper], axis=0)

                full_actions.append(delta_action)

                # __import__("ipdb").set_trace()
                # see ../robosuite/controllers/controller_factory.py
                obs, reward, done, info = env.step(delta_action.tolist())
                if done:
                    task_successes += 1
                    total_successes += 1
                    disturb_res[ID2CATEGORY[task_id + 1][0]]["success_count"] += 1
                    break
                t += 1
                step += 1

            task_episodes += 1
            total_episodes += 1
            disturb_res[ID2CATEGORY[task_id + 1][0]]["total_count"] += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"

            if args.save_videos:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path)
                    / f"rollout_{ID2CATEGORY[task_id+1][1]}_episode{episode_idx}_{suffix}.mp4",
                    [np.asarray(x) for x in replay_images],
                    fps=25,
                )

            if full_actions:
                full_actions = np.stack(full_actions)
            # np.save(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.npy", full_actions)

            # print(pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4")
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
    result_name = f"{args.start_idx}_to_{args.end_idx}.json"
    with open(os.path.join(args.log_path, result_name), "w", encoding="utf-8") as f:
        json.dump(disturb_res, f)
    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s | %(message)s",
        datefmt="%m/%d [%H:%M:%S]",
        force=True,
    )
    if os.getenv("DEBUG", False):
        start_debugpy_once()
    tyro.cli(eval_libero)
