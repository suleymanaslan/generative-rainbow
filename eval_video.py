from env import StarPilotEnv
from rainbow.agent import Agent
import torch
import numpy as np
import cv2
import moviepy.video.io.ImageSequenceClip


def to_img(obs):
    if len(obs.shape) == 4:
        img = obs[-1].permute(1, 2, 0).cpu().numpy()
    else:
        img = obs.permute(1, 2, 0).cpu().numpy()
    return cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)


def to_moviepy_list(img_list):
    return np.array([(img * 255).astype(np.uint8) for img in img_list])


def create_video(env, filename, fps, seconds):
    vid_imgs = []
    observation, ep_reward, done = env.reset(), 0, False
    count = 0
    while count < fps * seconds:
        action, generated_next_observation = agent.act(observation, get_generated=True)
        next_observation, reward, done, info = env.step(action)
        imgs = [to_img(observation), to_img(generated_next_observation), to_img(next_observation)]
        vid_imgs.append(np.concatenate(to_moviepy_list(imgs), axis=1))
        count += 1
        observation = next_observation
        if done:
            observation, ep_reward, done = env.reset(), 0, False
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(vid_imgs, fps=60)
    clip.write_videofile(f"{filename}.mp4")


torch.manual_seed(828)
np.random.seed(828)
use_backgrounds = True

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
create_video(train_env, f"{agent_folder}/train_envs", fps=60, seconds=30)
create_video(test_env, f"{agent_folder}/test_envs", fps=60, seconds=30)
