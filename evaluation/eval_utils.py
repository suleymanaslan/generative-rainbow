from env import StarPilotEnv
from rainbow.agent import Agent
import torch
import cv2
import numpy as np


def to_img(obs):
    if len(obs.shape) == 4:
        img = obs[-1].permute(1, 2, 0).cpu().numpy()
    else:
        img = obs.permute(1, 2, 0).cpu().numpy()
    return cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def to_img_padded(obs, pad):
    return np.pad(to_img(obs), pad, constant_values=1)


def format_img(img_list):
    return [(img * 255).astype(np.uint8) for img in img_list]


def to_moviepy_list(img_list):
    return np.array(format_img(img_list))


def init_evaluation(use_backgrounds=False):
    torch.manual_seed(828)
    np.random.seed(828)
    train_env = StarPilotEnv(history_length=16, num_levels=10,
                             start_level=0, distribution_mode="easy",
                             use_backgrounds=use_backgrounds, view_mode="rgb")
    test_env = StarPilotEnv(history_length=train_env.window, num_levels=20,
                            start_level=int(100e3), distribution_mode=train_env.distribution_mode,
                            use_backgrounds=train_env.use_backgrounds, view_mode=train_env.view_mode)
    agent = Agent(train_env, atoms=51, v_min=-20.0, v_max=20.0, batch_size=8, multi_step=3,
                  discount=0.99, norm_clip=10.0, lr=5e-4, adam_eps=1.5e-4, hidden_size=512,
                  noisy_std=0.1, gan_lr_mult=1e-4, training_mode="branch")
    agent_folder = "../trained_agent/background" if use_backgrounds else "../trained_agent/base"
    agent.load(f"{agent_folder}")
    return train_env, test_env, agent, agent_folder
