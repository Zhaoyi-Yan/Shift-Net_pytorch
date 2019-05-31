import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter


class Self_Attn (nn.Module):
	""" Self attention Layer"""

	'''
	https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
	'''

	def __init__(self, in_dim, activation, with_attention=False):
		super (Self_Attn, self).__init__ ()
		self.chanel_in = in_dim
		self.activation = activation
		self.with_attention = with_attention

		self.query_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.key_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
		self.value_conv = nn.Conv2d (in_channels=in_dim, out_channels=in_dim, kernel_size=1)
		self.gamma = nn.Parameter (torch.zeros (1))

		self.softmax = nn.Softmax (dim=-1)  #

	def forward(self, x):
		"""
			inputs :
				x : input feature maps( B X C X W X H)
			returns :
				out : self attention value + input feature
				attention: B X N X N (N is Width*Height)
		"""
		m_batchsize, C, width, height = x.size ()
		proj_query = self.query_conv (x).view (m_batchsize, -1, width * height).permute (0, 2, 1)  # B X CX(N)
		proj_key = self.key_conv (x).view (m_batchsize, -1, width * height)  # B X C x (*W*H)
		energy = torch.bmm (proj_query, proj_key)  # transpose check
		attention = self.softmax (energy)  # BX (N) X (N)
		proj_value = self.value_conv (x).view (m_batchsize, -1, width * height)  # B X C X N

		out = torch.bmm (proj_value, attention.permute (0, 2, 1))
		out = out.view (m_batchsize, C, width, height)

		out = self.gamma * out + x

		if self.with_attention:
			return out, attention
		else:
			return out

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias


class PartialConv(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, bias=True):
		super(PartialConv).__init__()
		self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
									stride, padding, dilation, groups, bias)
		self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
								   stride, padding, dilation, groups, False)

		#self.input_conv.apply(weights_init('kaiming'))

		torch.nn.init.constant_(self.mask_conv.weight, 1.0)

		# mask is not updated
		for param in self.mask_conv.parameters():
			param.requires_grad = False

	def forward(self, input, mask):

		output = self.input_conv(input * mask)
		if self.input_conv.bias is not None:
			output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
				output)
		else:
			output_bias = torch.zeros_like(output)

		with torch.no_grad():
			output_mask = self.mask_conv(mask)

		no_update_holes = output_mask == 0
		mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

		output_pre = (output - output_bias) / mask_sum + output_bias
		output = output_pre.masked_fill_(no_update_holes, 0.0)

		new_mask = torch.ones_like(output)
		new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

		return output, new_mask


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
