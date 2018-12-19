import torch
import torch.nn as nn

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if gan_type == 'wgan_gp':
            self.loss = nn.MSELoss()
        elif gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_type == 'vanilla':
            self.loss = nn.BCELoss()
        #######################################################################
        ###  Relativistic GAN - https://github.com/AlexiaJM/RelativisticGAN ###
        #######################################################################
        # When Using `BCEWithLogitsLoss()`, remove the sigmoid layer in D.
        elif gan_type == 're_s_gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 're_avg_gan':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_type == 're_avg_hinGan':
            self.loss = nn.BCELoss()

        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

