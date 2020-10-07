import os
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class Trainer:
    def __init__(self, max_steps, plot_steps):
        self.max_steps = max_steps
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

    def _init_training(self, file):
        training_timestamp = str(int(time.time()))
        self.model_dir = f'trained_models/model_{training_timestamp}/'

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if file:
            shutil.copy2(f'./{file}', self.model_dir)

        self.print_and_log(f"{datetime.now()}, start training")

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
