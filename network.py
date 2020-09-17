# adapted from https://github.com/Kaixhin/Rainbow and https://github.com/facebookresearch/pytorch_GAN_zoo

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_GAN_zoo.models.networks.custom_layers import EqualizedConv2d, EqualizedLinear, NormalizationLayer, \
    Upscale2d
from pytorch_GAN_zoo.models.utils.utils import num_flat_features
from pytorch_GAN_zoo.models.networks.mini_batch_stddev_module import miniBatchStdDev


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


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.downsample = None if stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                             bias=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class DQN(nn.Module):
    def __init__(self, atoms, action_size, history_length, hidden_size, noisy_std):
        super(DQN, self).__init__()
        self.atoms = atoms
        self.action_size = action_size
        self.history_length = history_length
        self.hidden_size = hidden_size

        self.conv1 = nn.Conv2d(self.history_length, 64, kernel_size=5, stride=2, padding=2, bias=False)
        self.layer1 = BasicBlock(64, 64, 1)
        self.layer2 = BasicBlock(64, 64, 1)
        self.layer3 = BasicBlock(64, 128, 2)
        self.layer4 = BasicBlock(128, 128, 1)
        self.layer5 = BasicBlock(128, 256, 2)
        self.layer6 = BasicBlock(256, 256, 1)
        self.layer7 = BasicBlock(256, 256, 2)
        self.layer8 = BasicBlock(256, 256, 1)

        self.net, self.feat_size = self._get_net()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.fc_h_v = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_h_a = NoisyLinear(self.feat_size, self.hidden_size, std_init=noisy_std)
        self.fc_z_v = NoisyLinear(self.hidden_size, self.atoms, std_init=noisy_std)
        self.fc_z_a = NoisyLinear(self.hidden_size, self.action_size * self.atoms, std_init=noisy_std)

    def _get_net(self):
        net = nn.Sequential(self.conv1, nn.ReLU(inplace=True),
                            self.layer1,
                            self.layer2,
                            self.layer3,
                            self.layer4,
                            self.layer5,
                            self.layer6,
                            self.layer7,
                            self.layer8,
                            )

        feat_size = 4096
        return net, feat_size

    def forward(self, x, use_log_softmax=False):
        x = self.net(x)
        x = x.view(-1, self.feat_size)

        v = self.fc_z_v(F.relu(self.fc_h_v(x)))
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_size, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        q = F.log_softmax(q, dim=2) if use_log_softmax else F.softmax(q, dim=2)

        return q, x

    def reset_noise(self):
        self.fc_h_v.reset_noise()
        self.fc_h_a.reset_noise()
        self.fc_z_v.reset_noise()
        self.fc_z_a.reset_noise()


class GeneratorDQN(DQN):
    def __init__(self, atoms, action_size, history_length, hidden_size, noisy_std):
        super(GeneratorDQN, self).__init__(atoms, action_size, history_length, hidden_size, noisy_std)
        self.depth_scale0 = 128
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_output = 1
        self.dim_latent = self.feat_size + self.action_size
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.to_rgb_layers = nn.ModuleList()

        self.format_layer = EqualizedLinear(self.dim_latent, 16 * self.scales_depth[0], equalized=self.equalized_lr,
                                            initBiasToZero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = None

    def add_scale(self, depth_new_scale, final_scale=False):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_last_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        if final_scale:
            self.to_rgb_layers.append(EqualizedConv2d(depth_new_scale, self.dim_output, 1, equalized=self.equalized_lr,
                                                      initBiasToZero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, skip_gan=False, actions=None, use_log_softmax=False):
        x = self.net(x)
        x = x.view(-1, self.feat_size)

        v = self.fc_z_v(F.relu(self.fc_h_v(x)))
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))
        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_size, self.atoms)
        q = v + a - a.mean(1, keepdim=True)
        q = F.log_softmax(q, dim=2) if use_log_softmax else F.softmax(q, dim=2)

        if skip_gan:
            return q, x

        x = self.normalization_layer(x)
        x = x.view(-1, num_flat_features(x))
        x = torch.cat((x, actions), dim=1)
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 4, 4)
        x = self.normalization_layer(x)

        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            y = Upscale2d(y)

        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = Upscale2d(x)
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                y = Upscale2d(y)

        x = self.to_rgb_layers[-1](x)

        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x

        if self.generation_activation is not None:
            x = self.generation_activation(x)

        return q, x


class PGANDiscriminator(nn.Module):
    def __init__(self, history_length):
        super(PGANDiscriminator, self).__init__()
        self.depth_scale0 = 128
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_input = history_length + 1
        self.size_decision_layer = 1
        self.mini_batch_normalization = True
        self.dim_entry_scale0 = self.depth_scale0 + 1
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.from_rgb_layers = nn.ModuleList()

        self.merge_layers = nn.ModuleList()

        self.decision_layer = EqualizedLinear(self.scales_depth[0], self.size_decision_layer,
                                              equalized=self.equalized_lr, initBiasToZero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.dim_entry_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.group_scale0.append(EqualizedLinear(self.depth_scale0 * 16, self.depth_scale0, equalized=self.equalized_lr,
                                                 initBiasToZero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    def add_scale(self, depth_new_scale, final_scale=False):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_last_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        if final_scale:
            self.from_rgb_layers.append(EqualizedConv2d(self.dim_input, depth_new_scale, 1, equalized=self.equalized_lr,
                                                        initBiasToZero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, get_feature=False):
        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            y = self.leaky_relu(self.from_rgb_layers[- 2](y))

        x = self.leaky_relu(self.from_rgb_layers[-1](x))

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
            x = miniBatchStdDev(x)

        x = self.leaky_relu(self.group_scale0[0](x))

        x = x.view(-1, num_flat_features(x))

        x = self.leaky_relu(self.group_scale0[1](x))

        out = self.decision_layer(x)

        if not get_feature:
            return out

        return out, x
