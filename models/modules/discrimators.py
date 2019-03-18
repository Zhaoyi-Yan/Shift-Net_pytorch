import functools
import torch.nn as nn
from .denset_net import *
import util.util as util
import torch.nn.functional as F

from .modules import *
################################### This is for D ###################################
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_spectral_norm)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

# D2: weight mask
class D_WM(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=True, weight_mask=0.9):
        super(D_WM, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.weight_mask = weight_mask
        kw = 4
        padw = 1
        sequence = [
            spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), use_spectral_norm),
            nn.LeakyReLU(0.2)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias), use_spectral_norm),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias), use_spectral_norm),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]
        sequence += [spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw), use_spectral_norm)]

        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()
        
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        output = self.model(input)
        self.mask_binary = util.cal_flag_given_mask_thred(self.mask.squeeze(), 1, 1, 1).type_as(input).view(30, 30)
        output *= self.mask_binary
        output = self.sigmoid(output) if self.use_sigmoid else output

        return output

    def set_mask(self, mask_global):
        self.mask = F.interpolate(mask_global, (30, 30), mode='nearest')

# Defines a densetnet inspired discriminator (Should improve its ability to create stronger representation)
class DenseNetDiscrimator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, use_spectral_norm=True):
        super(DenseNetDiscrimator, self).__init__()
        self.model = densenet121(pretrained=True, use_spectral_norm=use_spectral_norm)
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if self.use_sigmoid:
            return self.sigmoid(self.model(input))
        else:
            return self.model(input)