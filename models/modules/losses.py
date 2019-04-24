import torch
import torch.nn as nn
import numpy as np

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan_gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
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
        else:
            raise ValueError("GAN type [%s] not recognized." % gan_type)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        if self.gan_type == 'wgan_gp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss

################# Discounting loss #########################
######################################################
class Discounted_L1(nn.Module):
    def __init__(self, opt):
        super(Discounted_L1, self).__init__()
        # Register discounting template as a buffer
        self.register_buffer('discounting_mask', torch.tensor(spatial_discounting_mask(opt.fineSize//2 - opt.overlap * 2, opt.fineSize//2 - opt.overlap * 2, 0.9, opt.discounting)))
        self.L1 = nn.L1Loss()

    def forward(self, input, target):
        self._assert_no_grad(target)
        input_tmp = input * self.discounting_mask
        target_tmp = target * self.discounting_mask
        return self.L1(input_tmp, target_tmp)


    def _assert_no_grad(self, variable):
        assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"


def spatial_discounting_mask(mask_width, mask_height, discounting_gamma, discounting=1):
    """Generate spatial discounting mask constant.
    Spatial discounting mask is first introduced in publication:
        Generative Image Inpainting with Contextual Attention, Yu et al.
    Returns:
        tf.Tensor: spatial discounting mask
    """
    gamma = discounting_gamma
    shape = [1, 1, mask_width, mask_height]
    if discounting:
        print('Use spatial discounting l1 loss.')
        mask_values = np.ones((mask_width, mask_height), dtype='float32')
        for i in range(mask_width):
            for j in range(mask_height):
                mask_values[i, j] = max(
                    gamma**min(i, mask_width-i),
                    gamma**min(j, mask_height-j))
        mask_values = np.expand_dims(mask_values, 0)
        mask_values = np.expand_dims(mask_values, 1)
        mask_values = mask_values
    else:
        mask_values = np.ones(shape, dtype='float32')

    return mask_values
