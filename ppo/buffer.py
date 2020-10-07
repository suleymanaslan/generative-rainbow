# adapted from https://github.com/openai/spinningup

import numpy as np
import scipy.signal
import torch


class Buffer:
    def __init__(self, size, obs_shape, gamma=0.99, lam=0.95):
        self.size = size
        self.obs_shape = obs_shape
        self.gamma = gamma
        self.lam = lam
        self.ptr = 0
        self.path_start_idx = 0
        self.device = torch.device("cuda:0")
        self.obs_buf = np.zeros((self.size,) + self.obs_shape, dtype=np.float32)
        self.act_buf = np.zeros(self.size, dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.logp_buf = np.zeros(self.size, dtype=np.float32)

    @staticmethod
    def discount_cumsum(x, discount):
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.size
        self.obs_buf[self.ptr] = obs.cpu().numpy()
        self.act_buf[self.ptr] = act.cpu().numpy()
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val.cpu().numpy()
        self.logp_buf[self.ptr] = logp.cpu().numpy()
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = self.discount_cumsum(deltas, self.gamma * self.lam)

        self.ret_buf[path_slice] = self.discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.size
        self.ptr, self.path_start_idx = 0, 0

        x = np.array(self.adv_buf, dtype=np.float32)
        global_sum, global_n = np.sum(x), len(x)
        adv_mean = global_sum / global_n

        global_sum_sq = np.sum((x - adv_mean) ** 2)
        adv_std = np.sqrt(global_sum_sq / global_n)

        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in data.items()}
