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

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)



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

class InceptionDown(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, intermediate=None, norm_layer=nn.BatchNorm2d, is_norm=False):
        super(InceptionDown, self).__init__()

        self.is_norm = is_norm

        self.in_channels = in_channels
        self.out_channels = out_channels

        if intermediate is None:
            intermediate = np.max([out_channels // 8, 16])
            if intermediate % 2 != 0:
                intermediate = out_channels
        out_channels = out_channels // 4

        # ACTIVATION
        self.act = nn.LeakyReLU(0.2, True)

        ## LEVEL 0
        self.conv0_1x1_0 = nn.Conv2d(in_channels, intermediate, 1,
                                    stride, 0, dilation, groups, bias)
        self.bconv0_1x1_0 = norm_layer(intermediate)

        self.conv0_1x1_1 = nn.Conv2d(in_channels, intermediate, 1,
                                    stride, 0, dilation, groups, bias)
        self.bconv0_1x1_1 = norm_layer(intermediate)

        self.max_pool0 = nn.MaxPool2d(2, 2)

        self.conv0_1x1_2 = nn.Conv2d(in_channels, out_channels, 1,
                                    stride, 0, dilation, groups, bias)
        self.bconv0_1x1_2 = norm_layer(out_channels)

        ## LEVEL 2
        self.conv1_3x3 = nn.Conv2d(intermediate, intermediate, 3,
                                    stride, 1, dilation, groups, bias)
        self.bconv1_3x3 = norm_layer(intermediate)

        self.conv1_1x3 = nn.Conv2d(intermediate, out_channels, (1, 3),
                                    stride, (0, 1), dilation, groups, bias)
        self.bconv1_1x3 = norm_layer(out_channels)

        self.conv1_3x1 = nn.Conv2d(intermediate, out_channels, (3, 1),
                                    stride, (1, 0), dilation, groups, bias)
        self.bconv1_3x1 = norm_layer(out_channels)

        self.conv1_1x1 = nn.Conv2d(in_channels, out_channels, 1,
                                    stride, 0, dilation, groups, bias)
        self.bconv1_1x1 = norm_layer(out_channels)

        ## LEVEL 3
        self.conv2_1x3 = nn.Conv2d(intermediate, out_channels, (1, 3),
                                   stride, (0, 1), dilation, groups, bias)
        self.bconv2_1x3 = norm_layer(out_channels)

        self.conv2_3x1 = nn.Conv2d(intermediate, out_channels, (3, 1),
                                   stride, (1, 0), dilation, groups, bias)
        self.bconv2_3x1 = norm_layer(out_channels)

    def _forward(self, input, conv, normalization):
        #print(self.in_channels, self.out_channels, input.shape)
        input = conv(input)
        input = normalization(input)
        return self.act(input)

    def forward(self, input):

        if self.is_norm:
            #LEVEL 0
            conv0_1x1_0 = self._forward(input, self.conv0_1x1_0, self.bconv0_1x1_0)

            conv0_1x1_1 = self._forward(input, self.conv0_1x1_1, self.bconv0_1x1_1) #self.conv0_1x1_1(input)

            max_pool0 = self.max_pool0(input)

            conv0_1x1_2 = self._forward(input, self.conv0_1x1_2, self.bconv0_1x1_2)#self.conv0_1x1_2(input)

            # LEVEL 1
            conv1_3x3 = self._forward(conv0_1x1_0, self.conv1_3x3, self.bconv1_3x3)#self.conv1_3x3(conv0_1x1_0)

            conv1_1x3 = self._forward(conv0_1x1_1, self.conv1_1x3, self.bconv1_1x3)#self.conv1_1x3(conv0_1x1_1)

            conv1_3x1 = self._forward(conv0_1x1_1, self.conv1_3x1, self.bconv1_3x1)#self.conv1_3x1(conv0_1x1_1)

            conv1_1x1 = self._forward(max_pool0, self.conv1_1x1, self.bconv1_1x1)#self.conv1_1x1(max_pool0)

            # LEVEL 2
            conv2_1x3 = self._forward(conv1_3x3, self.conv2_1x3, self.bconv2_1x3)#self.conv2_1x3(conv1_3x3)

            conv2_3x1 = self._forward(conv1_3x3, self.conv2_3x1, self.bconv2_3x1)#self.conv2_3x1(conv1_3x3)
        else:
            # LEVEL 0
            conv0_1x1_0 = self.conv0_1x1_0(input)

            conv0_1x1_1 = self.conv0_1x1_1(input)

            max_pool0 = self.max_pool0(input)

            conv0_1x1_2 = self.conv0_1x1_2(input)

            # LEVEL 1
            conv1_3x3 = self.conv1_3x3(conv0_1x1_0)

            conv1_1x3 = self.conv1_1x3(conv0_1x1_1)

            conv1_3x1 = self.conv1_3x1(conv0_1x1_1)

            conv1_1x1 = self.conv1_1x1(max_pool0)

            # LEVEL 2
            conv2_1x3 = self.conv2_1x3(conv1_3x3)

            conv2_3x1 = self.conv2_3x1(conv1_3x3)

        conv2_1x3_3x1 = conv2_1x3 + conv2_3x1

        conv1_1x3_3x1 = conv1_3x1 + conv1_1x3

        holder = [conv2_1x3_3x1, conv1_1x3_3x1, conv0_1x1_2]
        out = []
        for conv in holder:
            conv = self.max_pool0(conv)
            out.append(conv)

        out.append(conv1_1x1)

        return torch.cat(out, 1)

class InceptionUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, intermediate=None, norm_layer=nn.BatchNorm2d, is_norm=False):
        super(InceptionUp, self).__init__()

        self.is_norm = is_norm

        self.in_channels = in_channels
        self.out_channels = out_channels

        if intermediate is None:
            intermediate = np.max([out_channels // 8, 16])
            #print(in_channels, out_channels, intermediate)
        out_channels = out_channels // 4

        # ACTIVATION
        self.act = nn.LeakyReLU(0.2, True)

        ## LEVEL 0
        self.conv0_1x1_0 = nn.ConvTranspose2d(in_channels, out_channels, 1,
                                              stride=2, padding=0, output_padding=1,
                                              groups=1, bias=True, dilation=1)
        self.bconv0_1x1_0 = norm_layer(out_channels)

        self.conv0_1x1_1 = nn.ConvTranspose2d(in_channels, intermediate, 1,
                                              stride=2, padding=0, output_padding=1,
                                              groups=1, bias=True, dilation=1)
        self.bconv0_1x1_1 = norm_layer(intermediate)

        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv0_1x1_2 = nn.ConvTranspose2d(in_channels, intermediate, 1,
                                              stride=2, padding=0, output_padding=1,
                                              groups=1, bias=True, dilation=1)
        self.bconv0_1x1_2 = norm_layer(intermediate)

        ## LEVEL 2
        self.conv1_3x3 = nn.ConvTranspose2d(intermediate, out_channels, 3,
                                              stride=1, padding=1, output_padding=0,
                                              groups=1, bias=True, dilation=1)
        self.bconv1_3x3 = norm_layer(out_channels)

        self.conv1_5x5 = nn.ConvTranspose2d(intermediate, out_channels, 5,
                                              stride=1, padding=2, output_padding=0,
                                              groups=1, bias=True, dilation=1)
        self.bconv1_5x5 = norm_layer(out_channels)

        self.conv1_1x1 = nn.ConvTranspose2d(in_channels, out_channels, 1,
                                              stride=1, padding=0, output_padding=0,
                                              groups=1, bias=True, dilation=1)
        self.bconv1_1x1 = norm_layer(out_channels)

    def _forward(self, input, conv, normalization):
        #print(self.in_channels, self.out_channels, input.shape)
        input = conv(input)
        input = normalization(input)
        return self.act(input)

    def forward(self, input):

        if self.is_norm:
            #LEVEL 1
            conv0_1x1_0 = self._forward(input, self.conv0_1x1_0, self.bconv0_1x1_0)

            conv0_1x1_1 = self._forward(input, self.conv0_1x1_1, self.bconv0_1x1_1)#self.conv0_1x1_1(input)

            upsample0 = self.upsample0(input)

            conv0_1x1_2 = self._forward(input, self.conv0_1x1_2, self.bconv0_1x1_2)#self.conv0_1x1_2(input)

            # LEVEL 2
            conv1_3x3 = self._forward(conv0_1x1_1, self.conv1_3x3, self.bconv1_3x3)#self.conv1_3x3(conv0_1x1_1)

            conv1_5x5 = self._forward(conv0_1x1_2, self.conv1_5x5, self.bconv1_5x5)#self.conv1_5x5(conv0_1x1_2)

            conv1_1x1 = self._forward(upsample0, self.conv1_1x1, self.bconv1_1x1)#self.conv1_1x1(upsample0)

            holder = [conv0_1x1_0, conv1_3x3, conv1_5x5, conv1_1x1]
        else:
            # LEVEL 1
            conv0_1x1_0 = self.conv0_1x1_0(input)

            conv0_1x1_1 = self.conv0_1x1_1(input)

            upsample0 = self.upsample0(input)

            conv0_1x1_2 = self.conv0_1x1_2(input)

            # LEVEL 2
            conv1_3x3 = self.conv1_3x3(conv0_1x1_1)

            conv1_5x5 = self.conv1_5x5(conv0_1x1_2)

            conv1_1x1 = self.conv1_1x1(upsample0)

            holder = [conv0_1x1_0, conv1_3x3, conv1_5x5, conv1_1x1]

        return torch.cat(holder, 1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
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
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

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