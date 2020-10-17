# adapted from https://github.com/Kaixhin/Rainbow and https://github.com/facebookresearch/pytorch_GAN_zoo

import time
from datetime import datetime

import torch
import imageio
import cv2 as cv
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from rainbow.layers import mixed_pool2d

from rainbow.network import Encoder, DQN, FullDQN, BranchedDQN, FullGenerator, GeneratorDQN, BranchedGeneratorDQN, \
    Discriminator
from rainbow.network_utils import WGANGP, finite_check, wgangp_gradient_penalty


class Agent:
    def __init__(self, env, atoms, v_min, v_max, batch_size, multi_step, discount,
                 norm_clip, lr, adam_eps, hidden_size, noisy_std, gan_lr_mult, training_mode="joint", load_file=None):
        self.device = torch.device("cuda:0")
        self.env = env
        self.in_channels = self.env.window * 3 if self.env.view_mode == "rgb" else self.env.window
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
        self.lr = lr
        self.adam_eps = adam_eps
        self.gan_lr_mult = gan_lr_mult

        self.training_mode = training_mode
        assert self.training_mode in ["joint", "separate", "gan_feat", "branch", "dqn_only", "gan_only"]
        self._init_nets()
        self._init_optimizers()

        self.dqn_steps = 0
        self.gan_steps = 0
        self.steps_per_scale = int(25e3)
        self.scale = 0
        self.max_scale = 4

        self.model_alpha = 0.0
        self.alpha_update_cons = 0.002

        self.loss_criterion = WGANGP(self.device)
        self.epsilon_d = 0.001

        self.real_state = None
        self.real_next_state = None
        self.generated_state = None

        self.update_target_net()

        if load_file:
            self.load(f"trained_models/{load_file}")

    def _init_nets(self):
        img_dim = 3 if self.env.view_mode == "rgb" else 1
        if self.training_mode == "gan_feat":
            self.generator_net = FullGenerator(self.in_channels, self.action_size, dim_output=img_dim,
                                               residual_network=False).to(self.device)
            self.target_g_net = Encoder(self.in_channels, residual_network=False).to(self.device)
            self.online_net = DQN(self.generator_net.encoder.feat_size, self.hidden_size, self.atoms, self.action_size,
                                  self.noisy_std).to(self.device)
            self.target_net = DQN(self.generator_net.encoder.feat_size, self.hidden_size, self.atoms, self.action_size,
                                  self.noisy_std).to(self.device)
            self.discrm_net = Discriminator(self.action_size, dim_input=img_dim).to(self.device)
            for param in self.target_g_net.parameters():
                param.requires_grad = False
        elif self.training_mode == "branch":
            self.online_net = BranchedGeneratorDQN(self.in_channels, self.hidden_size, self.atoms, self.action_size,
                                                   self.noisy_std, dim_output=img_dim,
                                                   residual_network=False).to(self.device)
            self.target_net = BranchedDQN(self.in_channels, self.hidden_size, self.atoms, self.action_size,
                                          self.noisy_std, residual_network=False).to(self.device)
            self.discrm_net = Discriminator(self.action_size, dim_input=img_dim).to(self.device)
        else:
            self.online_net = GeneratorDQN(self.in_channels, self.hidden_size, self.atoms, self.action_size,
                                           self.noisy_std, dim_output=img_dim, residual_network=False).to(self.device)
            self.target_net = FullDQN(self.in_channels, self.hidden_size, self.atoms, self.action_size, self.noisy_std,
                                      residual_network=False).to(self.device)
            self.discrm_net = Discriminator(self.action_size, dim_input=img_dim).to(self.device)

        self.online_net.train()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

    def _init_optimizers(self):
        if self.training_mode == "gan_feat":
            self.optimizer_g = optim.Adam(filter(lambda p: p.requires_grad, self.generator_net.parameters()),
                                          betas=(0, 0.99), lr=self.lr)
            self.optimizer_o = optim.Adam(filter(lambda p: p.requires_grad, self.online_net.parameters()),
                                          lr=self.lr, eps=self.adam_eps)
            self.optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, self.discrm_net.parameters()),
                                          betas=(0, 0.99), lr=self.lr)
        else:
            self.optimizer_o = optim.Adam(filter(lambda p: p.requires_grad, self.online_net.parameters()),
                                          betas=(0, 0.99), lr=self.lr, eps=self.adam_eps)
            self.optimizer_d = optim.Adam(filter(lambda p: p.requires_grad, self.discrm_net.parameters()),
                                          betas=(0, 0.99), lr=self.lr)
            self.optimizer_o.zero_grad()
            self.optimizer_d.zero_grad()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()

    def save(self, save_dir):
        if self.training_mode == "gan_feat":
            torch.save(self.generator_net.state_dict(), f"{save_dir}/generator_net.pth")
            torch.save(self.target_g_net.state_dict(), f"{save_dir}/target_g_net.pth")
        torch.save(self.online_net.state_dict(), f"{save_dir}/online_net.pth")
        torch.save(self.target_net.state_dict(), f"{save_dir}/target_net.pth")
        torch.save(self.discrm_net.state_dict(), f"{save_dir}/discrm_net.pth")

    def load(self, load_dir):
        if self.training_mode == "gan_feat":
            self.generator_net.load_state_dict(torch.load(f"{load_dir}/generator_net.pth"))
            self.target_g_net.load_state_dict(torch.load(f"{load_dir}/target_g_net.pth"))
        self.online_net.load_state_dict(torch.load(f"{load_dir}/online_net.pth"))
        self.target_net.load_state_dict(torch.load(f"{load_dir}/target_net.pth"))
        self.discrm_net.load_state_dict(torch.load(f"{load_dir}/discrm_net.pth"))

    def save_generated(self, save_dir):
        if self.real_state is None:
            return
        if self.env.view_mode == "rgb":
            real_state = (self.real_state.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            real_next_state = (self.real_next_state.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            generated_state = (self.generated_state.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
        else:
            real_state = (self.real_state.numpy() * 255).astype(np.uint8)
            real_next_state = (self.real_next_state.numpy() * 255).astype(np.uint8)
            generated_state = (self.generated_state.numpy().clip(0, 1) * 255).astype(np.uint8)

        resize = False
        if resize:
            real_state = cv.resize(real_state, dsize=(512, 512), interpolation=cv.INTER_NEAREST)
            real_next_state = cv.resize(real_next_state, dsize=(512, 512), interpolation=cv.INTER_NEAREST)
            generated_state = cv.resize(generated_state, dsize=(512, 512), interpolation=cv.INTER_NEAREST)

        pgan_img = np.concatenate((real_state, real_next_state, generated_state), axis=1)
        imageio.imwrite(f"{save_dir}/{int(time.time())}.jpg", pgan_img)

    def update_target_net(self):
        if self.training_mode == "gan_feat":
            self.target_g_net.load_state_dict(self.generator_net.encoder.state_dict())
            self.target_net.load_state_dict(self.online_net.state_dict())
        else:
            self.target_net.encoder.load_state_dict(self.online_net.encoder.state_dict())
            self.target_net.dqn.load_state_dict(self.online_net.dqn.state_dict())

    def reset_noise(self):
        self.online_net.reset_noise()

    def _act(self, state):
        with torch.no_grad():
            if self.training_mode == "gan_feat":
                q = self.online_net(self.generator_net(state.unsqueeze(0), skip_gan=True))
            else:
                q, _ = self.online_net(state.unsqueeze(0), skip_gan=True)
            return (q * self.support).sum(2).argmax(1).item()

    def act(self, state):
        if self.env.view_mode == "rgb":
            state = state.view(-1, 64, 64)
        return self._act(state)

    def act_e_greedy(self, state, epsilon=0.001):
        action, features = self.act(state)
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_size), features
        else:
            return action, features

    def _dqn_loss(self, log_ps, states, actions, returns, next_states, nonterminals, gan_feat=False):
        self.dqn_steps += 1

        log_ps_a = log_ps[range(self.batch_size), actions]

        with torch.no_grad():
            if gan_feat:
                pns = self.online_net(self.target_g_net(next_states))
            else:
                pns, _ = self.online_net(next_states, skip_gan=True)
            dns = self.support.expand_as(pns) * pns
            argmax_indices_ns = dns.sum(2).argmax(1)
            self.target_net.reset_noise()
            if gan_feat:
                pns = self.target_net(self.target_g_net(next_states))
            else:
                pns = self.target_net(next_states)
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

        return loss

    def _get_sample(self, mem):
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)

        if self.env.view_mode == "rgb":
            states = states.view(self.batch_size, -1, 64, 64)
            next_states = next_states.view(self.batch_size, -1, 64, 64)

        return idxs, states, actions, returns, next_states, nonterminals, weights

    def _dqn_check(self, trainer, mem, idxs, loss, weights):
        clip_grad_norm_(self.online_net.parameters(), self.norm_clip)

        self.optimizer_o.step()

        if self.dqn_steps == 1 or self.dqn_steps % 2000 == 0:
            trainer.print_and_log(f"{datetime.now()} Loss:{(weights * loss).mean():.2f}")

        mem.update_priorities(idxs, loss.detach().cpu().numpy())

    def _learn(self, mem, trainer, idxs, states, actions, returns, next_states, nonterminals, weights):
        log_ps, _ = self.online_net(states, skip_gan=True, use_log_softmax=True)
        loss = self._dqn_loss(log_ps, states, actions, returns, next_states, nonterminals)

        self.optimizer_o.zero_grad()

        (weights * loss).mean().backward()
        self._dqn_check(trainer, mem, idxs, loss, weights)

    def learn(self, mem, trainer):
        idxs, states, actions, returns, next_states, nonterminals, weights = self._get_sample(mem)
        self._learn(mem, trainer, idxs, states, actions, returns, next_states, nonterminals, weights)

    def learn_joint(self, mem, trainer):
        idxs, states, actions, returns, next_states, nonterminals, weights = self._get_sample(mem)

        log_ps, pgan_states, pgan_next_states, pred_fake_g, loss_dict = self._gan_loss(states, actions, next_states)

        all_loss_o = loss_dict["g_fake"] * self.gan_lr_mult

        loss = self._dqn_loss(log_ps, states, actions, returns, next_states, nonterminals)

        all_loss_o += (weights * loss).mean()
        all_loss_o.backward(retain_graph=True)
        finite_check(self.online_net.parameters())
        self._dqn_check(trainer, mem, idxs, loss, weights)
        self._gan_check(trainer, pgan_states, pgan_next_states, pred_fake_g, loss_dict)

    def learn_gan_feat(self, mem, trainer):
        idxs, states, actions, returns, next_states, nonterminals, weights = self._get_sample(mem)

        _, pgan_states, pgan_next_states, pred_fake_g, loss_dict = self._gan_loss(states, actions, next_states,
                                                                                  gan_lr_mult=1, gan_feat=True)

        loss_dict["g_fake"].backward(retain_graph=True)
        finite_check(self.generator_net.parameters())

        self.optimizer_g.step()

        self._gan_check(trainer, pgan_states, pgan_next_states, pred_fake_g, loss_dict, gan_feat=True)

        log_ps = self.online_net(self.target_g_net(states), use_log_softmax=True)
        loss = self._dqn_loss(log_ps, states, actions, returns, next_states, nonterminals, gan_feat=True)

        self.optimizer_o.zero_grad()
        (weights * loss).mean().backward()
        self._dqn_check(trainer, mem, idxs, loss, weights)

    def learn_branch(self, mem, trainer):
        self.learn_joint(mem, trainer)

    def _gan_loss(self, states, actions, next_states, gan_lr_mult=None, gan_feat=False):
        if gan_lr_mult is None:
            gan_lr_mult = self.gan_lr_mult

        if gan_feat:
            generator_net = self.generator_net
            generator_optimizer = self.optimizer_g
        else:
            generator_net = self.online_net
            generator_optimizer = self.optimizer_o

        self.gan_steps += 1
        actions_one_hot = torch.eye(self.action_size)[actions].to(self.device)

        if (self.gan_steps % (self.steps_per_scale // 1000) == 0) and self.model_alpha > 0:
            self.model_alpha = max(0.0, self.model_alpha - self.alpha_update_cons)

        gan_frames = 2
        gan_channels = gan_frames * 3 if self.env.view_mode == "rgb" else gan_frames
        gan_next_frames = 1
        gan_next_channels = gan_next_frames * 3 if self.env.view_mode == "rgb" else gan_next_frames

        if self.scale < self.max_scale:
            pgan_states = mixed_pool2d(states[:, -gan_channels:, :, :])
            pgan_next_states = mixed_pool2d(next_states[:, -gan_next_channels:, :, :])
            for _ in range(1, self.max_scale - self.scale):
                pgan_states = mixed_pool2d(pgan_states)
                pgan_next_states = mixed_pool2d(pgan_next_states)
        else:
            pgan_states = states[:, -gan_channels:, :, :]
            pgan_next_states = next_states[:, -gan_next_channels:, :, :]

        if self.model_alpha > 0:
            low_res_real = mixed_pool2d(pgan_states)
            low_res_real = F.interpolate(low_res_real, scale_factor=2, mode='nearest')
            pgan_states = self.model_alpha * low_res_real + (1 - self.model_alpha) * pgan_states
            low_res_real = mixed_pool2d(pgan_next_states)
            low_res_real = F.interpolate(low_res_real, scale_factor=2, mode='nearest')
            pgan_next_states = self.model_alpha * low_res_real + (1 - self.model_alpha) * pgan_next_states

        if self.env.view_mode == "rgb":
            img_size = 2 ** (self.scale + 2)
            pgan_states = pgan_states.view(self.batch_size, gan_frames, 3, img_size, img_size)
            pgan_states = pgan_states.permute(0, 2, 1, 3, 4)
            pgan_next_states = pgan_next_states.view(self.batch_size, gan_next_frames, 3, img_size, img_size)
            pgan_next_states = pgan_next_states.permute(0, 2, 1, 3, 4)
        else:
            pgan_states = pgan_states.unsqueeze(1)
            pgan_next_states = pgan_next_states.unsqueeze(1)

        self.discrm_net.set_alpha(self.model_alpha)
        generator_net.set_alpha(self.model_alpha)

        self.optimizer_d.zero_grad()

        real_input = torch.cat((pgan_states, pgan_next_states), dim=2)
        pred_real_d = self.discrm_net(real_input, actions_one_hot, False)
        loss_d = self.loss_criterion.get_criterion(pred_real_d, True)
        all_loss_d = loss_d

        if gan_feat:
            pred_fake_g = generator_net(states, actions=actions_one_hot)
            log_ps = None
        else:
            log_ps, pred_fake_g = generator_net(states, actions=actions_one_hot, use_log_softmax=True)

        if self.env.view_mode == "rgb":
            img_size = 2 ** (self.scale + 2)
            pred_fake_g = pred_fake_g.view(self.batch_size, gan_next_frames, 3, img_size, img_size)
            pred_fake_g = pred_fake_g.permute(0, 2, 1, 3, 4)
        else:
            pred_fake_g = pred_fake_g.unsqueeze(1)

        fake_input = torch.cat((pgan_states, pred_fake_g.detach()), dim=2)
        pred_fake_d = self.discrm_net(fake_input, actions_one_hot, False)
        loss_d_fake = self.loss_criterion.get_criterion(pred_fake_d, False)
        all_loss_d += loss_d_fake

        loss_d_grad = wgangp_gradient_penalty(real_input, fake_input, actions_one_hot, self.discrm_net, weight=10.0,
                                              backward=False)
        (loss_d_grad * gan_lr_mult).backward(retain_graph=True)

        loss_epsilon = (pred_real_d[:, 0] ** 2).sum() * self.epsilon_d
        all_loss_d += loss_epsilon

        (all_loss_d * gan_lr_mult).backward(retain_graph=True)
        finite_check(self.discrm_net.parameters())
        self.optimizer_d.step()

        self.optimizer_d.zero_grad()
        generator_optimizer.zero_grad()

        pred_fake_d, phi_g_fake = self.discrm_net(
            torch.cat((pgan_states, pred_fake_g), dim=2), actions_one_hot, True)
        loss_g_fake = self.loss_criterion.get_criterion(pred_fake_d, True)

        loss_dict = {"g_fake": loss_g_fake,
                     "d_real": loss_d,
                     "d_fake": loss_d_fake,
                     "d_grad": loss_d_grad,
                     "epsilon": loss_epsilon}

        return log_ps, pgan_states, pgan_next_states, pred_fake_g, loss_dict

    def _gan_check(self, trainer, pgan_states, pgan_next_states, pred_fake_g, loss_dict, gan_feat=False):
        generator_net = self.generator_net if gan_feat else self.online_net

        if self.gan_steps == 1 or self.gan_steps % (self.steps_per_scale // 10) == 0:
            trainer.print_and_log(f"{datetime.now()} "
                                  f"[{self.scale}/{self.max_scale}][{self.gan_steps:05d}/{self.steps_per_scale}], "
                                  f"A:{self.model_alpha:.1f}, L_G:{loss_dict['g_fake'].item():.2f}, "
                                  f"L_DR:{loss_dict['d_real'].item():.2f}, L_DF:{loss_dict['d_fake'].item():.2f}, "
                                  f"L_DG:{loss_dict['d_grad']:.2f}, L_DE:{loss_dict['epsilon'].item():.2f}")

            if self.env.view_mode == "rgb":
                self.real_state = pgan_states[0, :, -1].detach().cpu()
                self.real_next_state = pgan_next_states[0, :, -1].detach().cpu()
                self.generated_state = pred_fake_g[0, :, 0].detach().cpu()
            else:
                self.real_state = pgan_states[0][0][-1].detach().cpu()
                self.real_next_state = pgan_next_states[0][0][-1].detach().cpu()
                self.generated_state = pred_fake_g[0][0][0].detach().cpu()
            self.save_generated(trainer.model_dir)

        if self.gan_steps >= self.steps_per_scale:
            self.gan_steps = 0
            if self.scale < self.max_scale:
                self.discrm_net.add_scale(depth_new_scale=128)
                generator_net.add_scale(depth_new_scale=128)

                self.discrm_net.to(self.device)
                generator_net.to(self.device)

                self._init_optimizers()

                self.scale += 1
                self.model_alpha = 1.0

    def learn_gan(self, mem, trainer, repeat=None):
        if repeat is None:
            repeat = self.steps_per_scale
        for ix in range(1, repeat + 1):
            idxs, states, actions, returns, next_states, nonterminals, weights = self._get_sample(mem)

            log_ps, pgan_states, pgan_next_states, pred_fake_g, loss_dict = self._gan_loss(states, actions, next_states,
                                                                                           gan_lr_mult=1)

            loss_dict["g_fake"].backward(retain_graph=True)
            finite_check(self.online_net.parameters())

            self.optimizer_o.step()

            self._gan_check(trainer, pgan_states, pgan_next_states, pred_fake_g, loss_dict)
