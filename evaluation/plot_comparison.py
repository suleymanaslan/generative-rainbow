import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

sns.set()
EP_COUNT = 60


def load_rewards(method, mode):
    rewards = np.array([np.load(f"../results/ep/{method}/{folder}/{mode}_ep_rewards.npy")[-EP_COUNT:]
                        for folder in sorted(os.listdir(f"../results/ep/{method}"), key=int)])
    return rewards.mean(axis=1), rewards.std(axis=1)


def read_results(method):
    x_ixs = np.array(list(map(int, sorted(os.listdir(f"../results/ep/{method}"), key=int))))
    train_means, train_stds = load_rewards(method, "train")
    test_means, test_stds = load_rewards(method, "test")
    return x_ixs, train_means, train_stds, test_means, test_stds


def smooth_array(x_ixs, array, x_new, k=2):
    return make_interp_spline(x_ixs, array, k=k)(x_new)


def smooth_results(x_ixs, arrays, points=91):
    x_ixs_smooth = np.linspace(x_ixs.min(), x_ixs.max(), points)
    arrays_smooth = [smooth_array(x_ixs, array, x_ixs_smooth, k=2)
                     for array in arrays]
    return x_ixs_smooth, arrays_smooth


def main():
    fig, ax = plt.subplots(1, 1, facecolor='w')

    x_ixs, train_means, train_stds, test_means, test_stds = read_results("dqn")
    x_ixs_smooth, arrays_smooth = smooth_results(x_ixs, [train_means, train_stds, test_means, test_stds])
    train_means_smooth, train_stds_smooth, test_means_smooth, test_stds_smooth = arrays_smooth

    ax.set_xticks(range(11))
    ax.set_ylim(0, 90)
    ax.set_ylabel("Episode Reward")
    ax.set_xlabel(r"Unique Training Environments")
    ax.plot(x_ixs_smooth, train_means_smooth, linewidth=2.5, color="maroon", label="Rainbow (train)")
    ax.plot(x_ixs_smooth, test_means_smooth, linewidth=2.5, color="lightcoral", label="Rainbow (test)")
    ax.errorbar(x_ixs, train_means, train_stds, ls="none", ecolor="maroon", elinewidth=1.5, capsize=3, capthick=2)
    ax.errorbar(x_ixs, test_means, test_stds, ls="none", ecolor="lightcoral", elinewidth=1.5, capsize=3, capthick=2)
    ax.fill_between(x_ixs_smooth, train_means_smooth - train_stds_smooth, train_means_smooth + train_stds_smooth,
                    alpha=0.3, color="maroon")
    ax.fill_between(x_ixs_smooth, test_means_smooth - test_stds_smooth, test_means_smooth + test_stds_smooth,
                    alpha=0.3, color="lightcoral")

    x_ixs, train_means, train_stds, test_means, test_stds = read_results("gan")
    x_ixs_smooth, arrays_smooth = smooth_results(x_ixs, [train_means, train_stds, test_means, test_stds])
    train_means_smooth, train_stds_smooth, test_means_smooth, test_stds_smooth = arrays_smooth

    ax.plot(x_ixs_smooth, train_means_smooth, linewidth=2.5, color="indigo", label="Rainbow+GAN (train)")
    ax.plot(x_ixs_smooth, test_means_smooth, linewidth=2.5, color="mediumorchid", label="Rainbow+GAN (test)")
    ax.errorbar(x_ixs, train_means, train_stds, ls="none", ecolor="indigo", elinewidth=1.5, capsize=3, capthick=2)
    ax.errorbar(x_ixs, test_means, test_stds, ls="none", ecolor="mediumorchid", elinewidth=1.5, capsize=3, capthick=2)
    ax.fill_between(x_ixs_smooth, train_means_smooth - train_stds_smooth, train_means_smooth + train_stds_smooth,
                    alpha=0.3, color="indigo")
    ax.fill_between(x_ixs_smooth, test_means_smooth - test_stds_smooth, test_means_smooth + test_stds_smooth,
                    alpha=0.3, color="mediumorchid")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', ncol=2, prop={'size': 8})
    fig.tight_layout()
    plt.savefig('../results/comparison.pdf')


main()
