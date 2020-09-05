# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_GAN_zoo.models.networks.custom_layers import EqualizedConv2d, EqualizedLinear, NormalizationLayer, \
    Upscale2d
from pytorch_GAN_zoo.models.utils.utils import num_flat_features
from pytorch_GAN_zoo.models.networks.mini_batch_stddev_module import miniBatchStdDev


class PGANGenerator(nn.Module):
    def __init__(self):
        super(PGANGenerator, self).__init__()
        self.depth_scale0 = 128
        self.equalized_lr = True
        self.init_bias_to_zero = True
        self.dim_output = 1
        self.dim_latent = 4096
        self.scales_depth = [self.depth_scale0]

        self.scale_layers = nn.ModuleList()

        self.to_rgb_layers = nn.ModuleList()
        self.to_rgb_layers.append(EqualizedConv2d(self.depth_scale0, self.dim_output, 1, equalized=self.equalized_lr,
                                                  initBiasToZero=self.init_bias_to_zero))

        self.format_layer = EqualizedLinear(self.dim_latent, 9 * self.scales_depth[0], equalized=self.equalized_lr,
                                            initBiasToZero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.depth_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2)

        self.normalization_layer = NormalizationLayer()

        self.generation_activation = None

    def add_scale(self, depth_new_scale):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_last_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        self.to_rgb_layers.append(EqualizedConv2d(depth_new_scale, self.dim_output, 1, equalized=self.equalized_lr,
                                                  initBiasToZero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        # print(f"input, x:{x.shape}")

        x = self.normalization_layer(x)
        x = x.view(-1, num_flat_features(x))
        x = self.leaky_relu(self.format_layer(x))
        x = x.view(x.size()[0], -1, 3, 3)
        x = self.normalization_layer(x)
        # print(f"format, x:{x.shape}")

        for conv_layer in self.group_scale0:
            x = self.leaky_relu(conv_layer(x))
            # print(f"conv, x:{x.shape}")
            x = self.normalization_layer(x)

        if self.alpha > 0 and len(self.scale_layers) == 1:
            y = self.to_rgb_layers[-2](x)
            # print(f"to rgb, y:{y.shape}")
            y = Upscale2d(y)
            # print(f"upscale, y:{y.shape}")

        for scale, layer_group in enumerate(self.scale_layers, 0):
            x = Upscale2d(x)
            # print(f"upscale, x:{x.shape}")
            for conv_layer in layer_group:
                x = self.leaky_relu(conv_layer(x))
                # print(f"conv, x:{x.shape}")
                x = self.normalization_layer(x)
            if self.alpha > 0 and scale == (len(self.scale_layers) - 2):
                y = self.to_rgb_layers[-2](x)
                # print(f"to rgb, y:{y.shape}")
                y = Upscale2d(y)
                # print(f"upscale, y:{y.shape}")

        x = self.to_rgb_layers[-1](x)
        # print(f"to rgb, x:{x.shape}")

        if self.alpha > 0:
            x = self.alpha * y + (1.0 - self.alpha) * x
            # print(f"blend, x:{x.shape}, alpha:{self.alpha}")

        if self.generation_activation is not None:
            x = self.generation_activation(x)
            # print(f"g act, x:{x.shape}")

        return x


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
        self.from_rgb_layers.append(EqualizedConv2d(self.dim_input, self.depth_scale0, 1, equalized=self.equalized_lr,
                                                    initBiasToZero=self.init_bias_to_zero))

        self.merge_layers = nn.ModuleList()

        self.decision_layer = EqualizedLinear(self.scales_depth[0], self.size_decision_layer,
                                              equalized=self.equalized_lr, initBiasToZero=self.init_bias_to_zero)

        self.group_scale0 = nn.ModuleList()
        self.group_scale0.append(
            EqualizedConv2d(self.dim_entry_scale0, self.depth_scale0, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.group_scale0.append(EqualizedLinear(self.depth_scale0 * 9, self.depth_scale0, equalized=self.equalized_lr,
                                                 initBiasToZero=self.init_bias_to_zero))

        self.alpha = 0

        self.leaky_relu = torch.nn.LeakyReLU(0.2)

    def add_scale(self, depth_new_scale):
        depth_last_scale = self.scales_depth[-1]
        self.scales_depth.append(depth_new_scale)

        self.scale_layers.append(nn.ModuleList())
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_new_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))
        self.scale_layers[-1].append(
            EqualizedConv2d(depth_new_scale, depth_last_scale, 3, padding=1, equalized=self.equalized_lr,
                            initBiasToZero=self.init_bias_to_zero))

        self.from_rgb_layers.append(EqualizedConv2d(self.dim_input, depth_new_scale, 1, equalized=self.equalized_lr,
                                                    initBiasToZero=self.init_bias_to_zero))

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x, get_feature=False):
        # print(f"input, x:{x.shape}")

        if self.alpha > 0 and len(self.from_rgb_layers) > 1:
            y = F.avg_pool2d(x, (2, 2))
            # print(f"pool, y:{y.shape}")
            y = self.leaky_relu(self.from_rgb_layers[- 2](y))
            # print(f"from rgb, y:{y.shape}")

        x = self.leaky_relu(self.from_rgb_layers[-1](x))
        # print(f"from rgb, x:{x.shape}")

        merge_layer = self.alpha > 0 and len(self.scale_layers) > 1

        shift = len(self.from_rgb_layers) - 2

        for group_layer in reversed(self.scale_layers):
            for layer in group_layer:
                x = self.leaky_relu(layer(x))
                # print(f"conv, x:{x.shape}")

            x = nn.AvgPool2d((2, 2))(x)
            # print(f"pool, x:{x.shape}")

            if merge_layer:
                merge_layer = False
                x = self.alpha * y + (1 - self.alpha) * x
                # print(f"merge, x:{x.shape}, alpha:{self.alpha}")

            shift -= 1

        if self.mini_batch_normalization:
            x = miniBatchStdDev(x)
            # print(f"batchnorm, x:{x.shape}")

        x = self.leaky_relu(self.group_scale0[0](x))
        # print(f"conv, x:{x.shape}")

        x = x.view(-1, num_flat_features(x))
        # print(f"flat, x:{x.shape}")

        x = self.leaky_relu(self.group_scale0[1](x))
        # print(f"linear, x:{x.shape}")

        out = self.decision_layer(x)
        # print(f"out:{out.shape}")

        if not get_feature:
            return out

        return out, x
