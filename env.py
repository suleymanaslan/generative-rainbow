# adapted from https://github.com/Kaixhin/Rainbow

from collections import deque
import time
import torch
import cv2
import gym
import numpy as np
from IPython import display
from PIL import Image


class Env:
    def __init__(self, history_length, action_size=None, view_mode="gray"):
        self.device = torch.device("cuda:0")
        self.wrapped_env = self._get_env()
        if action_size is None:
            action_size = self.wrapped_env.action_space.n
        self.action_size = action_size
        self.action_space = [i for i in range(self.action_size)]
        self.window = history_length
        self.obs_shape = (self.window,
                          self.wrapped_env.observation_space.shape[2],
                          self.wrapped_env.observation_space.shape[0],
                          self.wrapped_env.observation_space.shape[1])
        self.view_mode = view_mode
        assert self.view_mode in ["gray", "rgb"]
        self.state_buffer = deque([], maxlen=self.window)

    def _get_env(self):
        return gym.make("CarRacing-v0")

    def _reset_buffer(self):
        for _ in range(self.window):
            self.state_buffer.append(torch.zeros(96, 96, device=self.device))

    def _process_observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255)
        return observation

    def render(self):
        self.wrapped_env.render()

    def reset(self):
        self._reset_buffer()
        observation = self.wrapped_env.reset()
        observation = self._process_observation(observation)
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def close(self):
        self.wrapped_env.close()

    def _step(self, action, frame_buffer, render=False):
        reward = 0
        for t in range(4):
            observation_t, reward_t, done, info = self.wrapped_env.step(action)
            if render:
                self.render()
            reward += reward_t
            if t == 2:
                frame_buffer[0] = self._process_observation(observation_t)
            elif t == 3:
                frame_buffer[1] = self._process_observation(observation_t)
            if done:
                break
        observation = frame_buffer.max(0)[0]
        self.state_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0), reward, done, info

    @staticmethod
    def action_to_mca(action):
        action_r = action // 5
        action_m = action % 5
        rotation, movement, brake = 0.0, 0.0, 0.0

        if action_r == 0:
            rotation = -1.0
        elif action_r == 1:
            rotation = -0.5
        elif action_r == 2:
            rotation = 0.0
        elif action_r == 3:
            rotation = +0.5
        elif action_r == 4:
            rotation = +1.0

        if action_m == 0:
            movement = 0.0
            brake = +0.8
        elif action_m == 1:
            movement = 0.0
            brake = +0.4
        elif action_m == 2:
            movement = 0.0
            brake = 0.0
        elif action_m == 3:
            movement = +0.5
            brake = 0.0
        elif action_m == 4:
            movement = +1.0
            brake = 0.0

        return np.array([rotation, movement, brake])

    def step(self, action):
        frame_buffer = torch.zeros(2, 96, 96, device=self.device)
        action = self.action_to_mca(action)
        return self._step(action, frame_buffer)


class StarPilotEnv(Env):
    def __init__(self, history_length, num_levels, start_level, distribution_mode, use_backgrounds,
                 action_size=None, view_mode="gray"):
        self.num_levels = num_levels
        self.start_level = start_level
        self.distribution_mode = distribution_mode
        self.use_backgrounds = use_backgrounds
        super(StarPilotEnv, self).__init__(history_length, action_size, view_mode)
        self.generated_buffer = deque([], maxlen=self.window)

    def _get_env(self):
        return gym.make("procgen:procgen-starpilot-v0", num_levels=self.num_levels, start_level=self.start_level,
                        distribution_mode=self.distribution_mode, use_backgrounds=self.use_backgrounds)

    def _reset_buffer(self):
        blank_obs = torch.zeros(3, 64, 64, device=self.device) if self.view_mode == "rgb" else \
            torch.zeros(64, 64, device=self.device)
        for _ in range(self.window):
            self.state_buffer.append(blank_obs)
            self.generated_buffer.append(blank_obs)

    def reset(self):
        self._reset_buffer()
        observation = self.wrapped_env.reset()
        observation = self._process_observation(observation)
        self.state_buffer.append(observation)
        self.generated_buffer.append(observation)
        return torch.stack(list(self.state_buffer), 0)

    def _process_observation(self, observation):
        if self.view_mode == "gray":
            observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).div_(255)
        if self.view_mode == "rgb":
            observation = observation.permute(2, 0, 1)
        return observation

    def render(self):
        render_img = self.state_buffer[-1].permute(1, 2, 0) if self.view_mode == "rgb" else self.state_buffer[-1]
        render_img = cv2.resize((render_img.cpu().numpy() * 255).astype(np.uint8), (0, 0), fx=4.0, fy=4.0,
                                interpolation=cv2.INTER_NEAREST)
        display.clear_output(wait=True)
        display.display(Image.fromarray(render_img))
        time.sleep(1 / 60)

    def step(self, action, generated_observation=None):
        observation, reward, done, info = self.wrapped_env.step(action)
        observation = self._process_observation(observation)
        self.state_buffer.append(observation)
        if generated_observation is not None:
            alpha = np.random.rand()
            generated_observation = observation * alpha + generated_observation * (1 - alpha)
            self.generated_buffer.append(generated_observation)
            info["generated_observation"] = torch.stack(list(self.generated_buffer), 0)
        return torch.stack(list(self.state_buffer), 0), reward, done, info
