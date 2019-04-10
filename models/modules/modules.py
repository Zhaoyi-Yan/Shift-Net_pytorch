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
		with torch.no_grad():
			output_mask = self.mask_conv(mask)

		output = self.input_conv(input * mask)
		if self.input_conv.bias is not None:
			output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
				output)
		else:
			output_bias = torch.zeros_like(output)

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