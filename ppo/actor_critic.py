# adapted from https://github.com/openai/spinningup

import torch.nn as nn
import torch
from torch.distributions.categorical import Categorical


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.net, self.feat_size = self._get_net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _get_net(self):
        net = nn.Sequential(nn.Conv2d(self.in_channels, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                            nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                            )
        feat_size = 4096
        return net, feat_size

    def forward(self, x):
        return self.net(x).view(-1, self.feat_size)


class Actor(nn.Module):
    def __init__(self, in_channels, action_size):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.logits_net = nn.Sequential(self.encoder,
                                        nn.Linear(self.encoder.feat_size, 256), nn.ReLU(inplace=True),
                                        nn.Linear(256, action_size)
                                        )

    def distribution(self, observation):
        logits = self.logits_net(observation)
        policy = Categorical(logits=logits)
        return policy

    @staticmethod
    def log_prob_from_distribution(policy, action):
        return policy.log_prob(action)

    def forward(self, observation, action=None):
        policy = self.distribution(observation)
        log_prob_action = None if action is None else self.log_prob_from_distribution(policy, action)
        return policy, log_prob_action


class Critic(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.value_net = nn.Sequential(self.encoder,
                                       nn.Linear(self.encoder.feat_size, 256), nn.ReLU(inplace=True),
                                       nn.Linear(256, 1)
                                       )

    def forward(self, observation):
        return torch.squeeze(self.value_net(observation), -1)


class ActorCritic(nn.Module):
    def __init__(self, env, action_size):
        super().__init__()
        device = torch.device("cuda:0")
        self.env = env
        self.in_channels = self.env.window * 3 if self.env.view_mode == "rgb" else self.env.window
        self.actor = Actor(self.in_channels, action_size).to(device)
        self.critic = Critic(self.in_channels).to(device)

    def step(self, observation, skip_value=False):
        if self.env.view_mode == "rgb":
            observation = observation.view(-1, 64, 64)
        with torch.no_grad():
            observation = observation.unsqueeze(0)
            policy = self.actor.distribution(observation)
            action = policy.sample()
            log_prob_action = self.actor.log_prob_from_distribution(policy, action)
            value = None if skip_value else self.critic(observation)
        return action, value, log_prob_action

    def act(self, observation):
        return self.step(observation, skip_value=True)[0]
