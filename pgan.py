# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import time
import torch
import imageio
import numpy as np
from torch import optim

from pgan_network import PGANGenerator, PGANDiscriminator
from pytorch_GAN_zoo.models.loss_criterions import base_loss_criterions
from pytorch_GAN_zoo.models.loss_criterions.gradient_losses import WGANGPGradientPenalty
from pytorch_GAN_zoo.models.utils.utils import finiteCheck


class PGAN:
    def __init__(self, env):
        self.device = torch.device("cuda:0")
        self.env = env
        self.learning_rate = 0.001
        self.discriminator_net = PGANDiscriminator(self.env.window)
        self.generator_net = PGANGenerator()

        for _ in range(5):
            self.discriminator_net.add_scale(depth_new_scale=128)
            self.generator_net.add_scale(depth_new_scale=128)
        self.discriminator_net.to(self.device)
        self.generator_net.to(self.device)

        self.optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, self.discriminator_net.parameters()),
                                      betas=[0, 0.99], lr=self.learning_rate)
        self.optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, self.generator_net.parameters()),
                                      betas=[0, 0.99], lr=self.learning_rate)

        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()

        self.loss_criterion = base_loss_criterions.WGANGP(self.device)
        self.epsilon_d = 0.001

        self.real_state = None
        self.real_next_state = None
        self.generated_state = None

    def learn(self, states, features, actions, next_states):
        self.optimizer_d.zero_grad()

        self.real_state = states[0][-1].detach().cpu()
        self.real_next_state = next_states[0][-1].detach().cpu()

        real_input = torch.cat((states, next_states[:, self.env.window - 1:self.env.window, :, :]), dim=1)
        pred_real_d = self.discriminator_net(real_input, False)
        loss_d = self.loss_criterion.getCriterion(pred_real_d, True)
        all_loss_d = loss_d

        pred_fake_g = self.generator_net(features.detach()).detach()
        fake_input = torch.cat((states, pred_fake_g), dim=1)
        pred_fake_d = self.discriminator_net(fake_input, False)
        loss_d_fake = self.loss_criterion.getCriterion(pred_fake_d, False)
        all_loss_d += loss_d_fake

        WGANGPGradientPenalty(real_input, fake_input, self.discriminator_net, weight=10.0, backward=True)

        loss_epsilon = (pred_real_d[:, 0] ** 2).sum() * self.epsilon_d
        all_loss_d += loss_epsilon

        all_loss_d.backward(retain_graph=True)
        finiteCheck(self.discriminator_net.parameters())
        self.optimizer_d.step()

        self.optimizer_d.zero_grad()
        self.optimizer_g.zero_grad()

        pred_fake_g = self.generator_net(features.detach())
        self.generated_state = pred_fake_g[0][0].detach().cpu()

        pred_fake_d, phi_g_fake = self.discriminator_net(torch.cat((states, pred_fake_g), dim=1), True)
        loss_g_fake = self.loss_criterion.getCriterion(pred_fake_d, True)

        loss_g_fake.backward(retain_graph=True)
        finiteCheck(self.generator_net.parameters())
        self.optimizer_g.step()

    def save(self, model_dir):
        real_state = (self.real_state.numpy() * 255).astype(np.uint8)
        real_next_state = (self.real_next_state.numpy() * 255).astype(np.uint8)
        generated_state = self.generated_state.numpy()
        generated_state = ((generated_state - generated_state.min()) / (
                generated_state.max() - generated_state.min()) * 255).astype(np.uint8)
        pgan_img = np.concatenate((real_state, real_next_state, generated_state), axis=1)
        imageio.imwrite(f"{model_dir}/{int(time.time())}.jpg", pgan_img)
