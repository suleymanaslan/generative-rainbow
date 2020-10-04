import time
import os
import shutil
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


class Trainer:
    def __init__(self, max_steps, replay_frequency, reward_clip, learning_start_step,
                 target_update, gan_steps, gan_scale_steps, eval_steps, plot_steps, training_mode="joint"):
        self.max_steps = max_steps
        self.replay_frequency = replay_frequency
        self.reward_clip = reward_clip
        self.learning_start_step = learning_start_step
        self.target_update = target_update
        self.training_mode = training_mode
        assert self.training_mode in ["joint", "separate", "frozen", "dqn_only", "gan_only"]
        self.gan_steps = gan_steps
        self.gan_scale_steps = gan_scale_steps
        self.eval_steps = eval_steps
        self.plot_steps = plot_steps
        self.ep_rewards = []
        self.ep_steps = []
        self.train_ep_rewards = []
        self.test_ep_rewards = []
        self.eval_ep_steps = []
        self.avg_ep_rewards = None
        self.model_dir = None

    def print_and_log(self, text):
        print(text)
        print(text, file=open(f'{self.model_dir}/log.txt', 'a'))

    def train(self, env, train_env, test_env, agent, mem, file=None):
        training_timestamp = str(int(time.time()))
        self.model_dir = f'trained_models/model_{training_timestamp}/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if file:
            shutil.copy2(f'./{file}', self.model_dir)

        self.print_and_log(f"{datetime.now()}, start training")
        priority_weight_increase = (1 - mem.priority_weight) / (self.max_steps - self.learning_start_step)
        finished = False
        episode = 0
        steps = 0
        while not finished:
            observation, ep_reward, done = env.reset(), 0, False
            while not done:
                action, _ = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1
                if steps % self.eval_steps == 0:
                    self.eval(train_env, test_env, agent, steps)
                if steps % self.plot_steps == 0:
                    self.save(agent)
                if steps >= self.max_steps:
                    finished = True
                    break
                if steps % self.replay_frequency == 0:
                    agent.reset_noise()
                if self.reward_clip > 0:
                    reward = max(min(reward, self.reward_clip), -self.reward_clip) / self.reward_clip
                mem.append(observation, action, reward, done)
                if steps >= self.learning_start_step:
                    mem.priority_weight = min(mem.priority_weight + priority_weight_increase, 1)
                    if self.training_mode == "joint":
                        if steps % self.replay_frequency == 0:
                            agent.learn_joint(mem, self)
                    elif self.training_mode == "separate":
                        if steps % self.gan_steps == 0:
                            agent.learn_gan(mem, self,
                                            repeat=(self.gan_steps * agent.steps_per_scale) // self.gan_scale_steps)
                        if steps % self.replay_frequency == 0:
                            agent.learn(mem, self)
                    elif self.training_mode == "dqn_only":
                        if steps % self.replay_frequency == 0:
                            agent.learn(mem, self)
                    elif self.training_mode == "gan_only":
                        agent.learn_gan(mem, self, repeat=1)
                    else:
                        raise NotImplementedError
                    if steps % self.target_update == 0:
                        agent.update_target_net()
                observation = next_observation
            episode += 1
            self.ep_rewards.append(ep_reward)
            self.ep_steps.append(steps)
            if episode == 1 or episode % 100 == 0:
                self.print_and_log(f"{datetime.now()}, episode:{episode:5d}, step:{steps:6d}, reward:{ep_reward:4.1f}")
        self.print_and_log(f"{datetime.now()}, end training")

    def plot(self, max_reward=60):
        self.avg_ep_rewards = [np.array(self.ep_rewards[max(0, i - 150):max(1, i)]).mean()
                               for i in range(len(self.ep_rewards))]
        plt.style.use('default')
        sns.set()
        plt.figure(figsize=(10, 6))
        plt.gca().set_ylim([0, max_reward])
        plt.gca().set_xlim([0, self.max_steps])
        plt.yticks(np.arange(0, max_reward + 1, max_reward // 10))
        plt.xticks(np.arange(0, self.max_steps + 1, self.max_steps // 6))
        plt.plot(self.ep_steps, self.ep_rewards, alpha=0.5)
        plt.plot(self.ep_steps, self.avg_ep_rewards, linewidth=3)
        if len(self.eval_ep_steps) > 0:
            plt.plot(self.eval_ep_steps, self.train_ep_rewards, linewidth=3)
            plt.plot(self.eval_ep_steps, self.test_ep_rewards, linewidth=3)
        plt.xlabel('steps')
        plt.ylabel('episode reward')
        plt.savefig(f"{self.model_dir}/training.png")
        plt.close()

    def save(self, agent):
        agent.save(self.model_dir)

        np.save(f"{self.model_dir}/ep_rewards.npy", self.ep_rewards)
        np.save(f"{self.model_dir}/ep_steps.npy", self.ep_steps)
        np.save(f"{self.model_dir}/train_ep_rewards.npy", self.train_ep_rewards)
        np.save(f"{self.model_dir}/test_ep_rewards.npy", self.test_ep_rewards)
        np.save(f"{self.model_dir}/eval_ep_steps.npy", self.eval_ep_steps)

        self.plot()
        plt.show()

    @staticmethod
    def _eval(env, num_levels, agent):
        eval_reward = 0
        for _ in range(num_levels):
            observation, ep_reward, done = env.reset(), 0, False
            while not done:
                action, _ = agent.act(observation)
                next_observation, reward, done, info = env.step(action)
                ep_reward += reward
                observation = next_observation
            eval_reward += ep_reward
        eval_reward /= num_levels
        return eval_reward

    def eval(self, train_env, test_env, agent, steps):
        train_reward = self._eval(train_env, test_env.num_levels * 2, agent)
        self.train_ep_rewards.append(train_reward)
        self.print_and_log(f"{datetime.now()}, train_reward:{train_reward:4.1f}")

        test_reward = self._eval(test_env, test_env.num_levels * 2, agent)
        self.test_ep_rewards.append(test_reward)
        self.print_and_log(f"{datetime.now()}, test_reward:{test_reward:4.1f}")

        self.eval_ep_steps.append(steps)
