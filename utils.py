import time
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer:
    def __init__(self, episodes, replay_frequency, reward_clip, max_steps, learning_start_step,
                 target_update, gan_steps, eval_steps):
        self.episodes = episodes
        self.replay_frequency = replay_frequency
        self.reward_clip = reward_clip
        self.max_steps = max_steps
        self.learning_start_step = learning_start_step
        self.target_update = target_update
        self.gan_steps = gan_steps
        self.eval_steps = eval_steps
        self.ep_rewards = []
        self.ep_steps = []
        self.eval_ep_rewards = []
        self.eval_ep_steps = []
        self.avg_ep_rewards = None
        self.model_dir = None

    def print_and_log(self, text):
        print(text)
        print(text, file=open(f'{self.model_dir}/log.txt', 'a'))

    def train(self, env, test_env, agent, mem, notebook_file=None):
        training_timestamp = str(int(time.time()))
        self.model_dir = f'trained_models/model_{training_timestamp}/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if notebook_file:
            shutil.copy2(f'./{notebook_file}.ipynb', self.model_dir)

        self.print_and_log(f"{datetime.now()}, start training")
        priority_weight_increase = (1 - mem.priority_weight) / (self.max_steps - self.learning_start_step)
        finished = False
        steps = 0
        for episode_ix in range(1, self.episodes + 1):
            observation, ep_reward, ep_step, done = env.reset(), 0, 0, False
            while not done:
                if steps % self.replay_frequency == 0:
                    agent.reset_noise()
                action, _ = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                ep_reward += reward
                ep_step += 1
                steps += 1
                if steps >= self.max_steps:
                    finished = True
                    break
                if steps % self.eval_steps == 0:
                    self.eval(test_env, agent, steps)
                if self.reward_clip > 0:
                    reward = max(min(reward, self.reward_clip), -self.reward_clip) / self.reward_clip
                mem.append(observation, action, reward, done)
                if steps >= self.learning_start_step:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
                    if steps % self.gan_steps == 0:
                        agent.learn_gan(mem, self)
                    if steps % self.replay_frequency == 0:
                        agent.learn(mem)
                    if steps % self.target_update == 0:
                        agent.update_target_net()
                observation = next_observation
            self.ep_rewards.append(ep_reward)
            self.ep_steps.append(steps)
            if episode_ix == 1 or episode_ix % 1000 == 0:
                self.plot()
            if episode_ix == 1 or episode_ix % 10 == 0:
                self.print_and_log(f"{datetime.now()}, episode:{episode_ix:5d}, step:{steps:6d}, "
                                   f"reward:{ep_reward:4.1f}"),
            if finished:
                break
        self.print_and_log(f"{datetime.now()}, end training")

    def plot(self, close=True):
        self.avg_ep_rewards = [np.array(self.ep_rewards[max(0, i - 150):max(1, i)]).mean()
                               for i in range(len(self.ep_rewards))]
        max_reward = 10
        plt.style.use('default')
        sns.set()
        plt.figure(figsize=(10, 6))
        plt.gca().set_ylim([0, max_reward])
        plt.gca().set_xlim([0, self.max_steps])
        plt.yticks(np.arange(0, max_reward + 1, max_reward // 10))
        plt.xticks(np.arange(0, self.max_steps + 1, self.max_steps // 6))
        plt.plot(self.ep_steps, self.ep_rewards, alpha=0.5)
        plt.plot(self.ep_steps, self.avg_ep_rewards, linewidth=3)
        if len(self.eval_ep_rewards) > 0:
            plt.plot(self.eval_ep_steps, self.eval_ep_rewards, linewidth=3)
        plt.xlabel('steps')
        plt.ylabel('episode reward')
        plt.savefig(f"{self.model_dir}/training.png")
        if close:
            plt.close()

    def save(self, agent):
        agent.save(self.model_dir)

        np.save(f"{self.model_dir}/ep_rewards.npy", self.ep_rewards)
        np.save(f"{self.model_dir}/ep_steps.npy", self.ep_steps)
        np.save(f"{self.model_dir}/eval_ep_rewards.npy", self.eval_ep_rewards)
        np.save(f"{self.model_dir}/eval_ep_steps.npy", self.eval_ep_steps)

        self.plot(close=False)
        plt.show()

    def eval(self, env, agent, steps):
        eval_reward = 0
        for _ in range(env.num_levels):
            observation, ep_reward, ep_step, done = env.reset(), 0, 0, False
            while not done:
                action, _ = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                ep_reward += reward
                ep_step += 1
                observation = next_observation
            eval_reward += ep_reward
        eval_reward /= env.num_levels
        self.eval_ep_rewards.append(eval_reward)
        self.eval_ep_steps.append(steps)
        self.print_and_log(f"{datetime.now()}, eval reward:{eval_reward:4.1f}")
