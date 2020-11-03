from env import StarPilotEnv
from rainbow.agent import Agent
import torch
import numpy as np
import cv2
import imageio


def to_img(obs):
    if len(obs.shape) == 4:
        img = obs[-1].permute(1, 2, 0).cpu().numpy()
    else:
        img = obs.permute(1, 2, 0).cpu().numpy()
    return cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def to_img_padded(obs, pad):
    img = to_img(obs)
    return np.pad(img, pad, constant_values=1)


def format_img(img_list):
    return [(img * 255).astype(np.uint8) for img in img_list]


def save_frames(env, folder):
    observation, ep_reward, done = env.reset(), 0, False
    count = 0
    while not done:
        action, generated_next_observation = agent.act(observation, get_generated=True)
        next_observation, reward, done, info = env.step(action)
        imgs = [to_img_padded(observation, pad),
                to_img_padded(generated_next_observation, pad),
                to_img_padded(next_observation, pad)]
        plot_img = np.concatenate(format_img(imgs), axis=1)
        count += 1
        imageio.imwrite(f"{folder}/{count:04d}.png", plot_img)
        observation = next_observation


torch.manual_seed(828)
np.random.seed(828)
use_backgrounds = False
pad = [(5, 5), (5, 5), (0, 0)]

train_env = StarPilotEnv(history_length=16, num_levels=10,
                         start_level=0, distribution_mode="easy",
                         use_backgrounds=use_backgrounds, view_mode="rgb")
test_env = StarPilotEnv(history_length=train_env.window, num_levels=20,
                        start_level=int(100e3), distribution_mode=train_env.distribution_mode,
                        use_backgrounds=train_env.use_backgrounds, view_mode=train_env.view_mode)
agent = Agent(train_env, atoms=51, v_min=-20.0, v_max=20.0, batch_size=8, multi_step=3,
              discount=0.99, norm_clip=10.0, lr=5e-4, adam_eps=1.5e-4, hidden_size=512,
              noisy_std=0.1, gan_lr_mult=1e-4, training_mode="branch")
agent_folder = "trained_agent/background" if use_backgrounds else "trained_agent/base"
agent.load(f"{agent_folder}")

save_frames(train_env, f"{agent_folder}/frames/train")
save_frames(test_env, f"{agent_folder}/frames/test")
