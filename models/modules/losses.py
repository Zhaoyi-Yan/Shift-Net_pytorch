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
            self.loss = nn.BCEWithLogitsLoss()
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

def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

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
