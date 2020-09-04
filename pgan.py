# adapted from https://github.com/facebookresearch/pytorch_GAN_zoo

import torch

from pgan_network import PGANGenerator, PGANDiscriminator


class PGAN:
    def __init__(self, env):
        self.device = torch.device("cuda:0")
        self.learning_rate = 0.001
        self.discriminator_net = PGANDiscriminator(env.window).to(self.device)
        self.generator_net = PGANGenerator().to(self.device)
