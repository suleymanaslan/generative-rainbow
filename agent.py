# adapted from https://github.com/Kaixhin/Rainbow

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from network import DQN


class Agent:
    def __init__(self, env, atoms, v_min, v_max, batch_size, multi_step,
                 discount, norm_clip, lr, adam_eps, hidden_size, noisy_std, load_file=None):
        self.device = torch.device("cuda:0")
        self.env = env
        self.action_size = len(self.env.action_space)
        self.hidden_size = hidden_size
        self.atoms = atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.atoms).to(self.device)
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1)
        self.batch_size = batch_size
        self.n = multi_step
        self.discount = discount
        self.norm_clip = norm_clip
        self.noisy_std = noisy_std

        self.online_net, self.target_net = self._get_nets()

        self.online_net.train()
        self.update_target_net()
        self.target_net.train()

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr, eps=adam_eps)

        if load_file:
            self.load(f"trained_models/{load_file}")

    def _get_nets(self):
        online_net = DQN(self.atoms, self.action_size, self.env.window, self.hidden_size, self.noisy_std) \
            .to(self.device)
        target_net = DQN(self.atoms, self.action_size, self.env.window, self.hidden_size, self.noisy_std) \
            .to(self.device)
        return online_net, target_net

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def save(self, save_dir):
        torch.save(self.online_net.state_dict(), f"{save_dir}/online_net.pth")
        torch.save(self.target_net.state_dict(), f"{save_dir}/target_net.pth")

    def load(self, load_dir):
        self.online_net.load_state_dict(torch.load(f"{load_dir}/online_net.pth"))
        self.target_net.load_state_dict(torch.load(f"{load_dir}/target_net.pth"))

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        self.online_net.reset_noise()

    def _act(self, state):
        with torch.no_grad():
            q, x = self.online_net(state.unsqueeze(0))
            return (q * self.support).sum(2).argmax(1).item(), x

    def act(self, state):
        return self._act(state)

    def act_e_greedy(self, state, epsilon=0.001):
        action, features = self.act(state)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size), features
        else:
            return action, features

    def _learn(self, mem, pgan, train_agent, idxs, states, actions, returns, next_states, nonterminals, weights):
        log_ps, features = self.online_net(states, use_log_softmax=True)
        pgan.learn(states, features, actions, next_states)
        if not train_agent:
            return
        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            pns, _ = self.online_net(next_states)
            dns = self.support.expand_as(pns) * pns
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()
            pns, _ = self.target_net(next_states)
            pns_a = pns[range(self.batch_size), argmax_indices_ns]

            tz = returns.unsqueeze(1) + nonterminals * (self.discount ** self.n) * self.support.unsqueeze(0)
            tz = tz.clamp(min=self.v_min, max=self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size) \
                .unsqueeze(1).expand(self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        loss = -torch.sum(m * log_ps_a, 1)
        self.online_net.zero_grad()
        (weights * loss).mean().backward()
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)
        self.optimizer.step()
        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def learn(self, mem, pgan, train_agent):
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        self._learn(mem, pgan, train_agent, idxs, states, actions, returns, next_states, nonterminals, weights)
