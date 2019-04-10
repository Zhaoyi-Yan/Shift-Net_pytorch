import torch
import torch.nn as nn
import torch.nn.functional as F

# For original shift
from models.shift_net.InnerShiftTriple import InnerShiftTriple
from models.shift_net.InnerCos import InnerCos

# For res shift
from models.res_shift_net.innerResShiftTriple import InnerResShiftTriple

# For pixel soft shift
from models.soft_shift_net.innerSoftShiftTriple import InnerSoftShiftTriple

# For patch patch shift
from models.patch_soft_shift.innerPatchSoftShiftTriple import InnerPatchSoftShiftTriple 

# For res patch patch shift
from models.res_patch_soft_shift.innerResPatchSoftShiftTriple import InnerResPatchSoftShiftTriple 

from .unet import UnetSkipConnectionBlock
from .modules import *

################################### ***************************  #####################################
###################################       partial Shift_net            #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class PartialConvLayer (nn.Module):

	def __init__(self, in_channels, out_channels, bn=True, bias=False, sample="none-3", activation="relu"):
		super().__init__()
		self.bn = bn

		if sample == "down-7":
			# Kernel Size = 7, Stride = 2, Padding = 3
			self.input_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 7, 2, 3, bias=False)

		elif sample == "down-5":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 5, 2, 2, bias=False)

		elif sample == "down-3":
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=False)

		else:
			self.input_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)
			self.mask_conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)

		nn.init.constant_(self.mask_conv.weight, 1.0)

		# "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
		# negative slope of leaky_relu set to 0, same as relu
		# "fan_in" preserved variance from forward pass
		nn.init.kaiming_normal_(self.input_conv.weight, a=0, mode="fan_in")

		for param in self.mask_conv.parameters():
			param.requires_grad = False

		if bn:
			# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
			# Applying BatchNorm2d layer after Conv will remove the channel mean
			self.batch_normalization = nn.BatchNorm2d(out_channels)

		if activation == "relu":
			# Used between all encoding layers
			self.activation = nn.ReLU()
		elif activation == "leaky_relu":
			# Used between all decoding layers (Leaky RELU with alpha = 0.2)
			self.activation = nn.LeakyReLU(negative_slope=0.2)

	def forward(self, input_x, mask):
		# output = W^T dot (X .* M) + b
		output = self.input_conv(input_x * mask)

		# requires_grad = False
		with torch.no_grad():
			# mask = (1 dot M) + 0 = M
			output_mask = self.mask_conv(mask)

		if self.input_conv.bias is not None:
			# spreads existing bias values out along 2nd dimension (channels) and then expands to output size
			output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
		else:
			output_bias = torch.zeros_like(output)

		# mask_sum is the sum of the binary mask at every partial convolution location
		mask_is_zero = (output_mask == 0)
		# temporarily sets zero values to one to ease output calculation 
		mask_sum = output_mask.masked_fill_(mask_is_zero, 1.0)

		# output at each location as follows:
		# output = (W^T dot (X .* M) + b - b) / M_sum + b ; if M_sum > 0
		# output = 0 ; if M_sum == 0
		output = (output - output_bias) / mask_sum + output_bias
		output = output.masked_fill_(mask_is_zero, 0.0)

		# mask is updated at each location
		new_mask = torch.ones_like(output)
		new_mask = new_mask.masked_fill_(mask_is_zero, 0.0)

		if self.bn:
			output = self.batch_normalization(output)

		if hasattr(self, 'activation'):
			output = self.activation(output)

		return output, new_mask


class PartialConvUNet(nn.Module):

	# 256 x 256 image input, 256 = 2^8
	def __init__(self, input_size=256, layers=7):
		if 2 ** (layers + 1) != input_size:
			raise AssertionError

		super().__init__()
		self.freeze_enc_bn = False
		self.layers = layers

		# ======================= ENCODING LAYERS =======================
		# 3x256x256 --> 64x128x128
		self.encoder_1 = PartialConvLayer(3, 64, bn=False, sample="down-7")

		# 64x128x128 --> 128x64x64
		self.encoder_2 = PartialConvLayer(64, 128, sample="down-5")

		# 128x64x64 --> 256x32x32
		self.encoder_3 = PartialConvLayer(128, 256, sample="down-3")

		# 256x32x32 --> 512x16x16
		self.encoder_4 = PartialConvLayer(256, 512, sample="down-3")

		# 512x16x16 --> 512x8x8 --> 512x4x4 --> 512x2x2
		for i in range(5, layers + 1):
			name = "encoder_{:d}".format(i)
			setattr(self, name, PartialConvLayer(512, 512, sample="down-3"))

		# ======================= DECODING LAYERS =======================
		# dec_7: UP(512x2x2) + 512x4x4(enc_6 output) = 1024x4x4 --> 512x4x4
		# dec_6: UP(512x4x4) + 512x8x8(enc_5 output) = 1024x8x8 --> 512x8x8
		# dec_5: UP(512x8x8) + 512x16x16(enc_4 output) = 1024x16x16 --> 512x16x16
		for i in range(5, layers + 1):
			name = "decoder_{:d}".format(i)
			setattr(self, name, PartialConvLayer(512 + 512, 512, activation="leaky_relu"))

		# UP(512x16x16) + 256x32x32(enc_3 output) = 768x32x32 --> 256x32x32
		self.decoder_4 = PartialConvLayer(512 + 256, 256, activation="leaky_relu")

		# UP(256x32x32) + 128x64x64(enc_2 output) = 384x64x64 --> 128x64x64
		self.decoder_3 = PartialConvLayer(256 + 128, 128, activation="leaky_relu")

		# UP(128x64x64) + 64x128x128(enc_1 output) = 192x128x128 --> 64x128x128
		self.decoder_2 = PartialConvLayer(128 + 64, 64, activation="leaky_relu")

		# UP(64x128x128) + 3x256x256(original image) = 67x256x256 --> 3x256x256(final output)
		self.decoder_1 = PartialConvLayer(64 + 3, 3, bn=False, activation="", bias=True)
	
	def forward(self, input_x, mask):
		encoder_dict = {}
		mask_dict = {}

		key_prev = "h_0"
		encoder_dict[key_prev], mask_dict[key_prev] = input_x, mask

		for i in range(1, self.layers + 1):
			encoder_key = "encoder_{:d}".format(i)
			key = "h_{:d}".format(i)
			# Passes input and mask through encoding layer
			encoder_dict[key], mask_dict[key] = getattr(self, encoder_key)(encoder_dict[key_prev], mask_dict[key_prev])
			key_prev = key

		# Gets the final output data and mask from the encoding layers
		# 512 x 2 x 2
		out_key = "h_{:d}".format(self.layers)
		out_data, out_mask = encoder_dict[out_key], mask_dict[out_key]

		for i in range(self.layers, 0, -1):
			encoder_key = "h_{:d}".format(i - 1)
			decoder_key = "decoder_{:d}".format(i)

			# Upsample to 2 times scale, matching dimensions of previous encoding layer output
			out_data = F.interpolate(out_data, scale_factor=2)
			out_mask = F.interpolate(out_mask, scale_factor=2)

			# concatenate upsampled decoder output with encoder output of same H x W dimensions
			# s.t. final decoding layer input will contain the original image
			out_data = torch.cat([out_data, encoder_dict[encoder_key]], dim=1)
			# also concatenate the masks
			out_mask = torch.cat([out_mask, mask_dict[encoder_key]], dim=1)
			
			# feed through decoder layers
			out_data, out_mask = getattr(self, decoder_key)(out_data, out_mask)

		return out_data

	def train(self, mode=True):
		super().train(mode)
		if self.freeze_enc_bn:
			for name, module in self.named_modules():
				if isinstance(module, nn.BatchNorm2d) and "enc" in name:
					# Sets batch normalization layers to evaluation mode
                    module.eval()


################################### ***************************  #####################################
###################################         Shift_net            #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_spectral_norm=use_spectral_norm)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        unet_shift_block = UnetSkipConnectionShiftBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, layer_to_last=3)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class UnetSkipConnectionShiftBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_spectral_norm=False, layer_to_last=3):
        super(UnetSkipConnectionShiftBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=layer_to_last)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last)
        innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel

################################### ***************************  #####################################
###################################         Res Shift_net            #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class ResUnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(ResUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_spectral_norm=use_spectral_norm)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        unet_shift_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, layer_to_last=3)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_spectral_norm=False, layer_to_last=3):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=layer_to_last)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last)
        innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # Res shift differs with other shift here. It is `*2` not `*3`.
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel

################################### ***************************  #####################################
###################################        Soft pixel shift      #####################################
################################### ***************************  #####################################
class SoftUnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(SoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_spectral_norm=use_spectral_norm)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        unet_shift_block = SoftUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                 mask_global, input_nc=None, \
                                                                 submodule=unet_block,
                                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, layer_to_last=3)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class SoftUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_spectral_norm=False, layer_to_last=3):
        super(SoftUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerSoftShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred, opt.triple_weight, layer_to_last=layer_to_last)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last)
        innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel


################################### ***************************  #####################################
###################################      patch soft shift_net    #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class PatchSoftUnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(PatchSoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_spectral_norm=use_spectral_norm)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        unet_shift_block = PatchSoftUnetSkipConnectionShiftTriple(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, layer_to_last=3)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class PatchSoftUnetSkipConnectionShiftTriple(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_spectral_norm=False, layer_to_last=3):
        super(PatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerPatchSoftShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse, layer_to_last=layer_to_last)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last)
        innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel


################################### ***************************  #####################################
###################################  Res patch soft shift_net    #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class ResPatchSoftUnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(ResPatchSoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True, use_spectral_norm=use_spectral_norm)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        unet_shift_block = ResPatchSoftUnetSkipConnectionShiftTriple(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, layer_to_last=3)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class ResPatchSoftUnetSkipConnectionShiftTriple(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_spectral_norm=False, layer_to_last=3):
        super(ResPatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResPatchSoftShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse, layer_to_last=layer_to_last)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last)
        innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # Res shift differs with other shift here. It is `*2` not `*3`.
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1). use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:  # if it is the outermost, directly pass the input in.
            return self.model(x)
        else:
            x_latter = self.model(x)
            _, _, h, w = x.size()
            if h != x_latter.size(2) or w != x_latter.size(3):
                x_latter = F.interpolate(x_latter, (h, w), mode='bilinear')
            return torch.cat([x_latter, x], 1)  # cat in the C channel
