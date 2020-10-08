# adapted from https://github.com/openai/spinningup

import torch
from torch import optim


class Agent:
    def __init__(self, buffer, actor_critic, policy_lr, value_lr, policy_train_iter, value_train_iter):
        self.buffer = buffer
        self.actor_critic = actor_critic

        self.policy_optimizer = optim.Adam(self.actor_critic.actor.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(self.actor_critic.critic.parameters(), lr=value_lr)

        self.policy_train_iter = policy_train_iter
        self.value_train_iter = value_train_iter

        self.target_kl = 0.01
        self.clip_ratio = 0.2

    def compute_loss_pi(self, data_dict):
        observation = data_dict['obs'].view(self.buffer.size, -1, 64, 64)
        action = data_dict['act']
        advantage = data_dict['adv']
        logp_old = data_dict['logp']

        policy, log_prob_action = self.actor_critic.actor(observation, action)
        ratio = torch.exp(log_prob_action - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantage
        loss_pi = -(torch.min(ratio * advantage, clip_adv)).mean()

        approx_kl = (logp_old - log_prob_action).mean().item()
        entropy = policy.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=entropy, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data_dict):
        obs, ret = data_dict['obs'].view(self.buffer.size, -1, 64, 64), data_dict['ret']
        return ((self.actor_critic.critic(obs) - ret) ** 2).mean()

    def learn(self):
        data_dict = self.buffer.get()

        with torch.no_grad():
            loss_pi_old, pi_info_old = self.compute_loss_pi(data_dict)
            loss_pi_old = loss_pi_old.item()
            loss_v_old = self.compute_loss_v(data_dict).item()

        for i in range(self.policy_train_iter):
            self.policy_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data_dict)

            kl = pi_info['kl']
            if kl > 1.5 * self.target_kl:
                break
            loss_pi.backward()
            self.policy_optimizer.step()

        for i in range(self.value_train_iter):
            self.value_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data_dict)
            loss_v.backward()
            self.value_optimizer.step()

        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']

    def save(self, save_dir):
        torch.save(self.actor_critic.state_dict(), f"{save_dir}/actor_critic.pth")
