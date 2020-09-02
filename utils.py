import time
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer:
    def __init__(self, episodes, replay_frequency, reward_clip, max_steps, learning_start_step, target_update):
        self.episodes = episodes
        self.replay_frequency = replay_frequency
        self.reward_clip = reward_clip
        self.max_steps = max_steps
        self.learning_start_step = learning_start_step
        self.target_update = target_update
        self.rewards = []
        self.ep_rewards = []
        self.ep_steps = []
        self.avg_ep_rewards = None
        self.model_dir = None

    def print_and_log(self, text):
        print(text)
        print(text, file=open(f'{self.model_dir}/log.txt', 'a'))

    def train(self, env, agent, mem, notebook_file=None, load_file=None):
        training_timestamp = str(int(time.time()))
        self.model_dir = f'trained_models/model_{training_timestamp}/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if notebook_file:
            shutil.copy2(f'./{notebook_file}.ipynb', self.model_dir)

        if load_file:
            agent.load(f"trained_models/{load_file}")

        self.print_and_log(f"{datetime.now()}, start training")
        priority_weight_increase = (1 - mem.priority_weight) / (self.max_steps - self.learning_start_step)
        steps = 0
        for episode_ix in range(1, self.episodes + 1):
            observation, ep_reward, ep_step, done = env.reset(), 0, 0, False
            while not done:
                if steps % self.replay_frequency == 0:
                    agent.reset_noise()
                action = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                self.rewards.append(reward)
                ep_reward += reward
                ep_step += 1
                steps += 1
                if self.reward_clip > 0:
                    reward = max(min(reward, self.reward_clip), -self.reward_clip) / self.reward_clip
                mem.append(observation, action, reward, done)
                if steps >= self.learning_start_step:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
                    if steps % self.replay_frequency == 0:
                        agent.learn(mem)
                    if steps % self.target_update == 0:
                        agent.update_target_net()
                observation = next_observation
            self.ep_rewards.append(ep_reward)
            self.ep_steps.append(steps)
            if episode_ix == 1 or episode_ix % 1 == 0:
                self.print_and_log(f"{datetime.now()}, episode:{episode_ix:4d}, step:{steps:5d}, "
                                   f"reward:{ep_reward:10.4f}")
        self.print_and_log(f"{datetime.now()}, end training")

    def save(self, agent):
        agent.save(self.model_dir)

        plt.style.use('default')
        self.avg_ep_rewards = [np.array(self.ep_rewards[max(0, i - 150):max(1, i)]).mean()
                               for i in range(len(self.ep_rewards))]
        plt.figure(figsize=(10, 6))
        axes = plt.gca()
        axes.set_ylim([0, 5])
        plt.plot(self.ep_steps, self.ep_rewards, alpha=0.5)
        plt.plot(self.ep_steps, self.avg_ep_rewards, linewidth=3)
        plt.xlabel('steps')
        plt.ylabel('episode reward')
        plt.savefig(f"{self.model_dir}/training.png")
        plt.show()

    def eval(self, env, agent):
        agent.eval()
        for _ in range(10):
            observation, ep_reward, ep_step, done = env.reset(), 0, 0, False
            env.render()
            while not done:
                action = agent.act_e_greedy(observation)
                next_observation, reward, done, info = env.step(action, render=True)
                ep_reward += reward
                ep_step += 1
                observation = next_observation
            self.print_and_log(f"{datetime.now()}, eval ep_step:{ep_step:5d}, reward:{ep_reward:10.4f}")
