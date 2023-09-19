import sys
import os
import json, argparse, pickle
import numpy as np
from PIL import Image
import torch
from agents.resnet import resnetv1_configs
from agents.gc_bc import GCBCAgent
from absl import app, flags, logging

np.set_printoptions(suppress=True)

logging.set_verbosity(logging.WARNING)

FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint", required=True)
flags.DEFINE_string("config_path", None, "Path to config of agent", required=True)
flags.DEFINE_string(
    "traj_dir",
    None,
    "Dir of a trajectory",
)
flags.DEFINE_string(
    "goal_image_path",
    None,
    "Path to a single goal image",
)

FIXED_STD = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def squash(path, im_size=128):  # squash from 480x640 to im_size
    im = Image.open(path)
    im = im.resize((im_size, im_size), Image.Resampling.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out

def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]

def normalize_action(action, mean, std):
    return (action - mean) / std

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
    
    checkpoint_path = FLAGS.checkpoint_path
    config_path = FLAGS.config_path
    assert os.path.exists(checkpoint_path), checkpoint_path
    config = json.load(open(config_path))
    config = argparse.Namespace(**config)
    agent, action_mean, action_std = load_checkpoint(
        checkpoint_path, config
    )

    lang = os.path.join(FLAGS.traj_dir, 'lang.txt')
    print(open(lang).read())

    obs_images_dir = os.path.join(FLAGS.traj_dir, 'images0')
    obs_image_fnames = sorted(os.listdir(obs_images_dir), key=lambda x: int(x.split('.')[0].split('_')[-1]))

    if FLAGS.goal_image_path:
        image_goal_path = FLAGS.goal_image_path
    else:
        image_goal_path = os.path.join(obs_images_dir, obs_image_fnames[-1])
    image_goal = squash(image_goal_path).transpose(2, 0, 1)

    gold_actions = pickle.load(open(os.path.join(FLAGS.traj_dir, 'policy_out.pkl'), 'rb'))
    states, next_states = process_state(FLAGS.traj_dir)

    all_mse = 0
    for fname, gold_action, state, next_state in zip(
        obs_image_fnames[:-1], gold_actions, states, next_states
    ):
        # load obs image
        image_obs = squash(os.path.join(obs_images_dir, fname)).transpose(2, 0, 1)

        obs = {"image": image_obs}
        goal_obs = {"image": image_goal}

        action = np.array(
            agent.sample_actions(obs, goal_obs, argmax=True).squeeze().cpu()
        )

        gold_action = gold_action['actions']
        gold_action[:6] = (next_state - state)[:6]
        gold_action = normalize_action(gold_action, action_mean, action_std)

        mse = np.square(np.subtract(gold_action, action)).sum()
        all_mse += mse

        print(fname)
        print(gold_action[:6], gold_action[6])
        print(action[:6], action[6])
        print()

    print('Avg MSE:', all_mse / len(gold_actions))


if __name__ == "__main__":
    app.run(main)
