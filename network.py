# adapted from https://github.com/Kaixhin/Rainbow and https://github.com/facebookresearch/pytorch_GAN_zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import flatten, upscale2d, EqualizedLinear, EqualizedConv2d, EqualizedConv3d, NormalizationLayer, \
    SqueezeLayer, NoisyLinear, BasicBlock
from network_utils import mini_batch_std_dev


class Encoder(nn.Module):
    def __init__(self, history_length, residual_network):
        super(Encoder, self).__init__()
        self.history_length = history_length
        self.residual_network = residual_network

        if self.residual_network:
            self.layer1 = BasicBlock(self.history_length, 64, 2)
            self.layer2 = BasicBlock(64, 64, 2)
            self.layer3 = BasicBlock(64, 64, 2)

        self.net, self.feat_size = self._get_net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _get_net(self):
        if self.residual_network:
            net = nn.Sequential(self.layer1, self.layer2, self.layer3)
        else:
            net = nn.Sequential(nn.Conv2d(self.history_length, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                nn.Conv2d(64, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                                )
        feat_size = 4096
        return net, feat_size

    def forward(self, x):
        return self.net(x).view(-1, self.feat_size)


class DQN(nn.Module):
    def __init__(self, feat_size, hidden_size, atoms, action_size, noisy_std):
        super(DQN, self).__init__()
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.atoms = atoms
        self.action_size = action_size

        self.fc_h_v = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(self.hidden_size, self.atoms, std_init=noisy_std)
        self.fc_z_a = NoisyLinear(self.hidden_size, self.action_size * self.atoms, std_init=noisy_std)

    def reset_noise(self):
        self.fc_h_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_z_a.reset_noise()

    def forward(self, x, use_log_softmax=False):
        v = self.fc_z_v(F.relu(self.fc_h_v(x), inplace=True))
        a = self.fc_z_a(F.relu(self.fc_h_a(x), inplace=True))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_size, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        q = F.log_softmax(q, dim=2) if use_log_softmax else F.softmax(q, dim=2)
        return q


class FullDQN(nn.Module):
    def __init__(self, history_length, hidden_size, atoms, action_size, noisy_std, residual_network=False):
        super(FullDQN, self).__init__()
        self.encoder = Encoder(history_length, residual_network)
        self.dqn = DQN(self.encoder.feat_size, hidden_size, atoms, action_size, noisy_std)

    def reset_noise(self):
        self.dqn.reset_noise()

    def forward(self, x, use_log_softmax=False):
        x = self.encoder(x)
        q = self.dqn(x, use_log_softmax)
        return q


class Generator(nn.Module):
    def __init__(self, feat_size, action_size, dim_output=1):
        super(Generator, self).__init__()
        self.feat_size = feat_size
        self.action_size = action_size
        self.dim_output = dim_output
        self.depth_scale0 = 128
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_latent = self.feat_size + self.action_size
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.to_rgb_layers = nn.ModuleList()
        self.to_rgb_layers.append(EqualizedConv2d(self.depth_scale0, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

        self.format_layer = EqualizedLinear(self.dim_latent, 16 * self.scales_depth[0], equalized=self.equalized_lr,
                                            init_bias_to_zero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = None

    def add_scale(self, depth_new_scale):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_last_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.to_rgb_layers.append(EqualizedConv2d(depth_new_scale, self.dim_output, 1, equalized=self.equalized_lr,
                                                  init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, actions):
        x = self.normalization_layer(x)
        x = flatten(x)
        x = torch.cat((x, actions), dim=1)
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            y = upscale2d(y)

        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = upscale2d(x)
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = upscale2d(y)

        x = self.to_rgb_layers[-1](x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        return x


class GeneratorDQN(nn.Module):
    def __init__(self, history_length, hidden_size, atoms, action_size, noisy_std, dim_output=1,
                 residual_network=False):
        super(GeneratorDQN, self).__init__()
        self.encoder = Encoder(history_length, residual_network)
        self.dqn = DQN(self.encoder.feat_size, hidden_size, atoms, action_size, noisy_std)
        self.generator = Generator(self.encoder.feat_size, action_size, dim_output)

    def add_scale(self, depth_new_scale):
        self.generator.add_scale(depth_new_scale)

    def set_alpha(self, alpha):
        self.generator.set_alpha(alpha)

    def reset_noise(self):
        self.dqn.reset_noise()

    def forward(self, x, skip_gan=False, actions=None, use_log_softmax=False):
        x = self.encoder(x)
        q = self.dqn(x, use_log_softmax)

        if skip_gan:
            return q, x

        assert actions is not None
        x = self.generator(x, actions)

        return q, x


class Discriminator(nn.Module):
    def __init__(self, action_size, dim_input=1):
        super(Discriminator, self).__init__()
        self.action_size = action_size
        self.depth_scale0 = 128
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_input = dim_input
        self.size_decision_layer = 1
        self.mini_batch_normalization = True
        self.dim_entry_scale0 = self.depth_scale0 + 1
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.from_rgb_layers = nn.ModuleList()
        self.from_rgb_layers.append(
            EqualizedConv3d(self.dim_input, self.depth_scale0, kernel_size=3, padding=[0, 1, 1],
                            equalized=self.equalized_lr, init_bias_to_zero=self.init_bias_to_zero))

        self.merge_layers = nn.ModuleList()

        self.decision_layer = EqualizedLinear(self.scales_depth[0], self.size_decision_layer,
                                              equalized=self.equalized_lr, init_bias_to_zero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.dim_entry_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.group_scale0.append(
            EqualizedLinear(self.depth_scale0 * 16 + self.action_size, self.depth_scale0, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2, inplace=True)
        self.squeeze = SqueezeLayer(2)

    def add_scale(self, depth_new_scale):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_last_scale, 3, padding=1, equalized=self.equalized_lr,
                            init_bias_to_zero=self.init_bias_to_zero))

        self.from_rgb_layers.append(
            EqualizedConv3d(self.dim_input, depth_new_scale, kernel_size=3, padding=[0, 1, 1],
                            equalized=self.equalized_lr, init_bias_to_zero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, actions, get_feature=False):
        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            y = F.avg_pool3d(x, (1, 2, 2))
            y = self.leaky_relu(self.from_rgb_layers[- 2](y))
            y = self.squeeze(y)

        x = self.leaky_relu(self.from_rgb_layers[-1](x))
        x = self.squeeze(x)

        merge_layer = self.alpha > 0 and len(self.scale_layers) > 1

        shift = len(self.from_rgb_layers) - 2

        for group_layer in reversed(self.scale_layers):
            for layer in group_layer:
                x = self.leaky_relu(layer(x))

            x = nn.AvgPool2d((2, 2))(x)

            if merge_layer:
                merge_layer = False
                x = self.alpha * y + (1 - self.alpha) * x

            shift -= 1

        if self.mini_batch_normalization:
            x = mini_batch_std_dev(x)

        x = self.leaky_relu(self.group_scale0[0](x))

        x = flatten(x)
        x = torch.cat((x, actions), dim=1)

        x = self.leaky_relu(self.group_scale0[1](x))

        out = self.decision_layer(x)

        if not get_feature:
            return out

        return out, x
