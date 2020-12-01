import numpy as np
import os

ep_count = 10
for folder in os.listdir("../results/noise"):
    rewards = np.load(f"../results/noise/{folder}/train_ep_rewards.npy")[-ep_count:]
    train_means = rewards.mean()
    train_stds = rewards.std()
    rewards = np.load(f"../results/noise/{folder}/test_ep_rewards.npy")[-ep_count:]
    test_means = rewards.mean()
    test_stds = rewards.std()
    diff = train_means - test_means
    print(f"{folder}, \t{train_means:.2f}, {train_stds:.2f}, {test_means:.2f}, {test_stds:.2f}, {diff:.2f}")
