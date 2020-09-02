# adapted from https://github.com/Kaixhin/Rainbow

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
    def __init__(self, atoms, action_size, history_length, hidden_size, noisy_std):
        super(DQN, self).__init__()
        self.atoms = atoms
        self.action_size = action_size
        self.history_length = history_length
        self.hidden_size = hidden_size
        self.net, self.feat_size = self._get_net()

        self.fc_h_v = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(self.hidden_size, self.atoms, std_init=noisy_std)
        self.fc_z_a = NoisyLinear(self.hidden_size, self.action_size * self.atoms, std_init=noisy_std)

    def _get_net(self):
        net = nn.Sequential(nn.Conv2d(self.history_length, 32, 8, stride=4, padding=0), nn.ReLU(),
                            nn.Conv2d(32, 64, 4, stride=2, padding=0), nn.ReLU(),
                            nn.Conv2d(64, 64, 3, stride=1, padding=0), nn.ReLU()
                            )
        feat_size = 3136
        return net, feat_size

    def forward(self, x, use_log_softmax=False):
        x = self.net(x)
        x = x.view(-1, self.feat_size)

        v = self.fc_z_v(F.relu(self.fc_h_v(x)))
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_size, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        q = F.log_softmax(q, dim=2) if use_log_softmax else F.softmax(q, dim=2)

        return q

    def reset_noise(self):
        self.fc_h_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_z_a.reset_noise()


class SimpleDQN(DQN):
    def __init__(self, atoms, action_size, history_length, hidden_size, noisy_std):
        super(SimpleDQN, self).__init__(atoms, action_size, history_length, hidden_size, noisy_std)

    def _get_net(self):
        net = nn.Sequential(nn.Linear(8 * self.history_length, self.hidden_size), nn.ReLU(),
                            nn.Linear(self.hidden_size, self.hidden_size), nn.ReLU()
                            )
        return net, self.hidden_size
