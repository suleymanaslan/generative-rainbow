import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sns.set()
EP_COUNT = 60


def load_gan_rewards(mode):
    rewards = np.array([np.load(f"results/gan/{folder}/{mode}_ep_rewards.npy")[-EP_COUNT:]
                        for folder in os.listdir("results/gan")])
    means = rewards.mean(axis=1)
    stds = rewards.std(axis=1)
    return means, stds


def load_dqn_results():
    x_vals_dqn = [0]
    train_rews_dqn = np.load(f"results/dqn/train_ep_rewards.npy")
    test_rews_dqn = np.load(f"results/dqn/test_ep_rewards.npy")
    train_means_dqn = np.expand_dims(train_rews_dqn[-EP_COUNT:].mean(), 0)
    train_stds_dqn = np.expand_dims(train_rews_dqn[-EP_COUNT:].std(), 0)
    test_means_dqn = np.expand_dims(test_rews_dqn[-EP_COUNT:].mean(), 0)
    test_stds_dqn = np.expand_dims(test_rews_dqn[-EP_COUNT:].std(), 0)
    return x_vals_dqn, train_means_dqn, train_stds_dqn, test_means_dqn, test_stds_dqn


def read_gan_results():
    x_ixs = np.flip(np.arange(6) - 5)
    x_vals = np.power(10., x_ixs)
    train_means, train_stds = load_gan_rewards("train")
    test_means, test_stds = load_gan_rewards("test")
    return x_ixs, x_vals, train_means, train_stds, test_means, test_stds


def smooth_array(x_ixs, array, x_new, k=2):
    return np.flip(make_interp_spline(np.flip(x_ixs), array, k=k)(x_new))


def smooth_results(x_ixs, arrays, points=51):
    x_new = np.flip(np.linspace(x_ixs.min(), x_ixs.max(), points))
    arrays_smooth = [smooth_array(x_ixs, array, x_new, k=2)
                     for array in arrays]
    x_vals_smooth = np.power(10., x_new)
    return x_vals_smooth, arrays_smooth


x_ixs, x_vals, train_means, train_stds, test_means, test_stds = read_gan_results()
x_vals_smooth, arrays_smooth = smooth_results(x_ixs, [train_means, train_stds, test_means, test_stds])
train_means_smooth, train_stds_smooth, test_means_smooth, test_stds_smooth = arrays_smooth

x_vals_dqn, train_means_dqn, train_stds_dqn, test_means_dqn, test_stds_dqn = load_dqn_results()

fig, (ax1, ax2) = plt.subplots(1, 2,
                               gridspec_kw={'width_ratios': [8, 1]},
                               sharey=True,
                               facecolor='w')

ax1.set_xscale("log")
ax1.set_xlim(10 ** 0.1, 10 ** -5.1)
ax1.set_ylim(0, 75)
ax1.set_ylabel("Episode Reward")
ax1.set_xlabel(r"$\eta_{GAN} (\times \eta_{DQN})$")
ax1.plot(x_vals_smooth,
         train_means_smooth,
         linewidth=2.5,
         color="indigo",
         label="DQN+GAN (train)",
         )
ax1.plot(x_vals_smooth,
         test_means_smooth,
         linewidth=2.5,
         color="mediumorchid",
         label="DQN+GAN (test)",
         )
ax1.errorbar(x_vals,
             train_means,
             train_stds,
             ls="none",
             ecolor="indigo",
             elinewidth=1.5,
             capsize=3,
             capthick=2,
             )
ax1.errorbar(x_vals,
             test_means,
             test_stds,
             ls="none",
             ecolor="mediumorchid",
             elinewidth=1.5,
             capsize=3,
             capthick=2,
             )
ax1.fill_between(x_vals_smooth,
                 train_means_smooth - train_stds_smooth,
                 train_means_smooth + train_stds_smooth,
                 alpha=0.3,
                 color="indigo",
                 )
ax1.fill_between(x_vals_smooth,
                 test_means_smooth - test_stds_smooth,
                 test_means_smooth + test_stds_smooth,
                 alpha=0.3,
                 color="mediumorchid",
                 )
ax2.set_xticks([0])
ax2.plot(x_vals_dqn,
         train_means_dqn,
         linewidth=2.5,
         color="maroon",
         label="DQN (train)",
         )
ax2.plot(x_vals_dqn,
         test_means_dqn,
         linewidth=2.5,
         color="lightcoral",
         label="DQN (test)",
         )
ax2.errorbar(x_vals_dqn,
             train_means_dqn,
             train_stds_dqn,
             ls="none",
             ecolor="maroon",
             elinewidth=1.5,
             capsize=3,
             capthick=2,
             )
ax2.errorbar(x_vals_dqn,
             test_means_dqn,
             test_stds_dqn,
             ls="none",
             ecolor="lightcoral",
             elinewidth=1.5,
             capsize=3,
             capthick=2,
             )
ax2.fill_between([-1] + x_vals_dqn + [1],
                 train_means_dqn - train_stds_dqn,
                 train_means_dqn + train_stds_dqn,
                 alpha=0.3,
                 color="maroon",
                 )
ax2.fill_between([-1] + x_vals_dqn + [1],
                 test_means_dqn - test_stds_dqn,
                 test_means_dqn + test_stds_dqn,
                 alpha=0.3,
                 color="lightcoral",
                 )
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax1.yaxis.tick_left()
handles, labels = ax2.get_legend_handles_labels()
handles, labels = [(a + b) for a, b in zip(ax1.get_legend_handles_labels(), ax2.get_legend_handles_labels())]
ax1.legend(handles,
           labels,
           loc='upper left',
           )
fig.tight_layout()
plt.subplots_adjust(wspace=0.01)
plt.savefig('results/fig1.pdf')
