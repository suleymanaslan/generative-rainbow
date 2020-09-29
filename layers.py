# adapted from https://github.com/Kaixhin/Rainbow and https://github.com/facebookresearch/pytorch_GAN_zoo

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod


def flatten(x):
    size = x.size()[1:]
    num_features = 1
    for s in size:
        num_features *= s
    return x.view(-1, num_features)


def upscale2d(x, factor=2):
    assert isinstance(factor, int) and factor >= 1
    if factor == 1:
        return x
    s = x.size()
    x = x.view(-1, s[1], s[2], 1, s[3], 1)
    x = x.expand(-1, s[1], s[2], factor, s[3], factor)
    x = x.contiguous().view(-1, s[1], s[2] * factor, s[3] * factor)
    return x


def mixed_pool2d(x, kernel_size=2, max_alpha=0.5):
    return F.max_pool2d(x, kernel_size) * max_alpha + F.avg_pool2d(x, kernel_size) * (1 - max_alpha)


class ConstrainedLayer(nn.Module):
    def __init__(self, module, equalized=True, lr_mul=1.0, init_bias_to_zero=True):
        super(ConstrainedLayer, self).__init__()

        self.module = module
        self.equalized = equalized

        if init_bias_to_zero:
            self.module.bias.data.fill_(0)
        if self.equalized:
            self.module.weight.data.normal_(0, 1)
            self.module.weight.data /= lr_mul
            self.weight = self.get_layer_normalization_factor(self.module) * lr_mul

    @staticmethod
    def get_layer_normalization_factor(x):
        size = x.weight.size()
        fan_in = prod(size[1:])
        return math.sqrt(2.0 / fan_in)

    def forward(self, x):

        x = self.module(x)
        if self.equalized:
            x *= self.weight
        return x


class EqualizedConv3d(ConstrainedLayer):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                                  **kwargs)


class EqualizedConv2d(ConstrainedLayer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias),
                                  **kwargs)


class EqualizedLinear(ConstrainedLayer):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        ConstrainedLayer.__init__(self, nn.Linear(in_features, out_features, bias=bias), **kwargs)


class NormalizationLayer(nn.Module):
    def __init__(self):
        super(NormalizationLayer, self).__init__()

    @staticmethod
    def forward(x, epsilon=1e-8):
        return x * (((x ** 2).mean(dim=1, keepdim=True) + epsilon).rsqrt())


class SqueezeLayer(nn.Module):
    def __init__(self, dim):
        super(SqueezeLayer, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)


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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None if stride == 1 else nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                                                             bias=False)
        self.downsample_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            identity = self.downsample_bn(identity)
        out += identity
        out = self.relu(out)
        return out
