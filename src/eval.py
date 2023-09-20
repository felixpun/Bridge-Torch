#!/usr/bin/env python3

import sys
import os
import json, argparse
import time
from datetime import datetime
import traceback

import matplotlib
import matplotlib.pyplot as plt
from absl import app, flags, logging
matplotlib.use("Agg")

import numpy as np

import torch
from PIL import Image

from agents.resnet import resnetv1_configs
from agents.gc_bc import GCBCAgent

# bridge_data_robot imports
from widowx_envs.widowx_env import BridgeDataRailRLPrivateWidowX
from multicam_server.topic_utils import IMTopic

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_multi_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_multi_string("config_path", None, "Path to config of agent", required=True)
flags.DEFINE_string("video_save_path", None, "Path to save video")
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)
flags.DEFINE_integer("num_timesteps", 120, "num timesteps")
flags.DEFINE_bool("blocking", False, "Use the blocking controller")
flags.DEFINE_spaceseplist("goal_eep", None, "Goal position")
flags.DEFINE_spaceseplist("initial_eep", None, "Initial position")
flags.DEFINE_bool("high_res", False, "Save high-res video and goal")

STEP_DURATION = 0.2
NO_PITCH_ROLL = False
NO_YAW = False
STICKY_GRIPPER_NUM_STEPS = 1

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def unnormalize_action(action, mean, std):
    return action * std + mean


def load_checkpoint(checkpoint_path, config):

    # create agent
    if config.method == 'gc_bc':
        encoder = resnetv1_configs[config.encoder](**config.encoder_kwargs)
        agent = GCBCAgent(encoder)

    # hydrate agent with parameters from checkpoint
    state_dict = torch.load(checkpoint_path)
    agent.load_state_dict(state_dict["module"])
    agent.cuda()
    agent.eval()

    # load action metadata
    action_mean = np.array(config.act_mean)
    action_std = np.array(config.act_std)

    return agent, action_mean, action_std


def main(_):
    
    # policies is a dict from method to (agent, action_mean, action_std)
    policies = {}
    for checkpoint_path, config_path in zip(
        FLAGS.checkpoint_path, FLAGS.config_path
    ):
        assert os.path.exists(checkpoint_path), checkpoint_path
        config = json.load(open(config_path))
        config = argparse.Namespace(**config)
        agent, action_mean, action_std = load_checkpoint(
            checkpoint_path, config
        )
        checkpoint_name = '-'.join(checkpoint_path.split("/")[-2:])
        method = config.method
        policies[f"{method}-{checkpoint_name}"] = (agent, action_mean, action_std)

    if FLAGS.initial_eep is not None:
        assert isinstance(FLAGS.initial_eep, list)
        initial_eep = [float(e) for e in FLAGS.initial_eep]
        start_state = np.concatenate([initial_eep, [0, 0, 0, 1]])
    else:
        start_state = None

    # set up environment
    env_params = {
        "fix_zangle": 0.1,
        "move_duration": 0.2,
        "adaptive_wait": True,
        "move_to_rand_start_freq": 1,
        "override_workspace_boundaries": [
            [0.1, -0.15, -0.1, -1.57, 0],
            [0.45, 0.25, 0.18, 1.57, 0],
        ],
        "action_clipping": "xyz",
        # "action_clipping": None,
        "catch_environment_except": False,
        "start_state": start_state,
        "return_full_image": FLAGS.high_res,
        "camera_topics": [IMTopic("/D435/color/image_raw", flip=True)],
    }
    env = BridgeDataRailRLPrivateWidowX(env_params, fixed_image_size=128)

    # load image goal
    image_goal = None
    if FLAGS.goal_image_path is not None:
        image_goal = np.array(Image.open(FLAGS.goal_image_path)).transpose(2, 0, 1)

    # goal sampling loop
    while True:
        # ask for new goal
        if image_goal is None:
            print("Taking a new goal...")
            ch = "y"
        else:
            ch = input("Taking a new goal? [y/n]")
        if ch == "y":
            if FLAGS.goal_eep is not None:
                assert isinstance(FLAGS.goal_eep, list)
                goal_eep = [float(e) for e in FLAGS.goal_eep]
            else:
                low_bound = [0.24, -0.1, 0.05, -1.57, 0]
                high_bound = [0.4, 0.20, 0.15, 1.57, 0]
                goal_eep = np.random.uniform(low_bound[:3], high_bound[:3])
            env.controller().open_gripper(True)
            try:
                env.controller().move_to_state(goal_eep, 0, duration=1.5)
            except Exception as e:
                continue
            input("Press [Enter] when ready for taking the goal image. ")
            obs = env.current_obs()
            image_goal = (
                obs["image"].reshape(3, 128, 128) * 255
            ).astype(np.uint8)
            full_goal_image = obs["full_image"][0]

        # ask for which policy to use
        if len(policies) == 1:
            policy_idx = 0
            input("Press [Enter] to start.")
        else:
            print("policies:")
            for i, name in enumerate(policies.keys()):
                print(f"{i}) {name}")
            policy_idx = int(input("select policy: "))

        policy_name = list(policies.keys())[policy_idx]
        agent, action_mean, action_std = policies[policy_name]
        try:
            env.reset()
            env.start()
        except Exception as e:
            continue

        # move to initial position
        try:
            if FLAGS.initial_eep is not None:
                assert isinstance(FLAGS.initial_eep, list)
                initial_eep = [float(e) for e in FLAGS.initial_eep]
                env.controller().move_to_state(initial_eep, 0, duration=1.5)
        except Exception as e:
            continue

        # do rollout
        obs = env.current_obs()
        last_tstep = time.time()
        images = []
        full_images = []
        t = 0
        # keep track of our own gripper state to implement sticky gripper
        is_gripper_closed = False
        num_consecutive_gripper_change_actions = 0
        try:
            while t < FLAGS.num_timesteps:
                if time.time() > last_tstep + STEP_DURATION or FLAGS.blocking:
                    image_obs = (
                        obs["image"].reshape(3, 128, 128) * 255
                    ).astype(np.uint8)
                    if FLAGS.high_res:
                        full_images.append(Image.fromarray(obs["full_image"][0]))
                    obs = {"image": image_obs, "proprio": obs["state"]}
                    goal_obs = {
                        "image": image_goal,
                    }

                    last_tstep = time.time()

                    action = np.array(
                        agent.sample_actions(obs, goal_obs, argmax=True).squeeze().cpu()
                    )
                    action = unnormalize_action(action, action_mean, action_std)
                    action += np.random.normal(0, FIXED_STD)

                    # sticky gripper logic
                    if (action[-1] < 0.5) != is_gripper_closed:
                        num_consecutive_gripper_change_actions += 1
                    else:
                        num_consecutive_gripper_change_actions = 0

                    if (
                        num_consecutive_gripper_change_actions
                        >= STICKY_GRIPPER_NUM_STEPS
                    ):
                        is_gripper_closed = not is_gripper_closed
                        num_consecutive_gripper_change_actions = 0

                    action[-1] = 0.0 if is_gripper_closed else 1.0

                    # remove degrees of freedom
                    if NO_PITCH_ROLL:
                        action[3] = 0
                        action[4] = 0
                    if NO_YAW:
                        action[5] = 0

                    # perform environment step
                    obs, rew, done, info = env.step(
                        action, last_tstep + STEP_DURATION, blocking=FLAGS.blocking
                    )

                    # save image
                    image_formatted = np.concatenate((image_goal, image_obs), axis=0)
                    images.append(Image.fromarray(image_formatted))

                    t += 1
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)

        # save video
        if FLAGS.video_save_path is not None:
            os.makedirs(FLAGS.video_save_path, exist_ok=True)
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            save_path = os.path.join(
                FLAGS.video_save_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.gif",
            )
            print(f"Saving Video at {save_path}")
            images[0].save(
                save_path,
                format="GIF",
                append_images=images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
        # save high-res video
        if FLAGS.high_res:
            base_path = os.path.join(FLAGS.video_save_path, "high_res")
            os.makedirs(base_path, exist_ok=True)
            print(f"Saving Video and Goal at {base_path}")
            curr_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            video_path = os.path.join(
                base_path,
                f"{curr_time}_{policy_name}_sticky_{STICKY_GRIPPER_NUM_STEPS}.gif",
            )
            full_images[0].save(
                video_path,
                format="GIF",
                append_images=full_images[1:],
                save_all=True,
                duration=200,
                loop=0,
            )
            goal_path = os.path.join(base_path, f"{curr_time}_{policy_name}.png")
            plt.imshow(full_goal_image)
            plt.axis("off")
            plt.savefig(goal_path, bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    app.run(main)
