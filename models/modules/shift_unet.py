import torch
import torch.nn as nn
import functools
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
###################################      UNet dilated shift_net  #####################################
################################### ***************************  #####################################
'''
Add 5 ResNet blocks layers of dilated convs to replace the inner 4 unet architecture.
And add more stages for each resolution.
'''
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetResNetGeneratorShiftTriple_1(nn.Module):
    def __init__(self, input_nc, output_nc, innerCos_list, shift_list, mask_global, opt, ngf=64,
                norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetResNetGeneratorShiftTriple_1, self).__init__()

        # Encoder layers
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for one stage
        self.e2_cc = spectral_norm(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e2_norm1 = norm_layer(ngf*2)
        self.e2_norm2 = norm_layer(ngf*2)

        self.e3_c = spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for one stage
        self.e3_cc = spectral_norm(nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e3_norm1 = norm_layer(ngf*4)
        self.e3_norm2 = norm_layer(ngf*4)

        self.e4_c = spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for two stages
        self.e4_cc1 = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e4_cc2 = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e4_norm1 = norm_layer(ngf*8)
        self.e4_norm2 = norm_layer(ngf*8)
        self.e4_norm3 = norm_layer(ngf*8)

        # ResNet blocks
        self.res1 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res2 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res3 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res4 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res5 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)

        # Then stay for another two stages
        self.e5_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e5_cc = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e5_norm1 = norm_layer(ngf*8)
        self.e5_norm2 = norm_layer(ngf*8)

        # decoder
        self.d5_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_dcc = spectral_norm(nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d5_norm1 = norm_layer(ngf*4)
        self.d5_norm2 = norm_layer(ngf*4)

        self.d6_dc = spectral_norm(nn.ConvTranspose2d(ngf*4*3, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_dcc = spectral_norm(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d6_norm1 = norm_layer(ngf*2)
        self.d6_norm2 = norm_layer(ngf*2)

        self.d7_dc = spectral_norm(nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_dcc = spectral_norm(nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d7_norm1 = norm_layer(ngf)
        self.d7_norm2 = norm_layer(ngf)

        self.d8_dc = spectral_norm(nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        # construct shift and innerCos
        self.shift = InnerShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=3)
        self.shift.set_mask(mask_global)
        shift_list.append(self.shift)

        self.innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=3)
        self.innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(self.innerCos)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2_1 = self.e2_norm1(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e2_2 = self.e2_norm2(self.e2_cc(F.leaky_relu_(e2_1, negative_slope=0.2)))

        e3_1 = self.e3_norm1(self.e3_c(F.leaky_relu_(e2_2, negative_slope=0.2)))
        e3_2 = self.e3_norm2(self.e3_cc(F.leaky_relu_(e3_1, negative_slope=0.2)))

        e4_1 = self.e4_norm1(self.e4_c(F.leaky_relu_(e3_2, negative_slope=0.2)))
        e4_2 = self.e4_norm2(self.e4_cc1(F.leaky_relu_(e4_1, negative_slope=0.2)))
        e4_3 = self.e4_norm3(self.e4_cc2(F.leaky_relu_(e4_2, negative_slope=0.2)))
        # dilated convs
        res1 = self.res1(F.relu_(e4_3))
        res2 = self.res2(F.relu_(res1))
        res3 = self.res3(F.relu_(res2))
        res4 = self.res4(F.relu_(res3))
        res5 = self.res5(F.relu_(res4))

        e5_1 = self.e5_norm1(self.e5_c(F.relu_(res5)))
        e5_2 = self.e5_norm2(self.e5_cc(F.relu_(e5_1)))

        # Decoder
        d5_1 = self.d5_norm1(self.d5_dc(F.relu_(torch.cat([e5_2, e4_3], dim=1))))
        d5_2 = self.d5_norm2(self.d5_dcc(F.relu_(d5_1)))
        d6_1 = self.d6_norm1(self.d6_dc(self.shift(self.innerCos(F.relu_(torch.cat([d5_2, e3_2], dim=1))))))
        d6_2 = self.d6_norm2(self.d6_dcc(F.relu_(d6_1)))
        d7_1 = self.d7_norm1(self.d7_dc(F.relu_(torch.cat([d6_2, e2_2], dim=1))))
        d7_2 = self.d7_norm2(self.d7_dcc(F.relu_(d7_1)))
        # No norm on the last layer
        d8 = self.d8_dc(F.relu_(torch.cat([d7_2, e1], 1)))

        d8 = torch.tanh(d8)

        return d8

################################### ***************************  #####################################
###################################      UNet dilated shift_net  #####################################
################################### ***************************  #####################################
'''
Add 5 ResNet blocks layers of dilated convs to replace the inner 4 unet architecture.
And add more stages for each resolution.
Add another `res from the frist resblock to the last block`
'''
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetResNetGeneratorShiftTriple_2(nn.Module):
    def __init__(self, input_nc, output_nc, innerCos_list, shift_list, mask_global, opt, ngf=64,
                norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetResNetGeneratorShiftTriple_2, self).__init__()

        # Encoder layers
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for one stage
        self.e2_cc = spectral_norm(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e2_norm1 = norm_layer(ngf*2)
        self.e2_norm2 = norm_layer(ngf*2)

        self.e3_c = spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for one stage
        self.e3_cc = spectral_norm(nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e3_norm1 = norm_layer(ngf*4)
        self.e3_norm2 = norm_layer(ngf*4)

        self.e4_c = spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        # stay for two stages
        self.e4_cc1 = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e4_cc2 = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e4_norm1 = norm_layer(ngf*8)
        self.e4_norm2 = norm_layer(ngf*8)
        self.e4_norm3 = norm_layer(ngf*8)

        # ResNet blocks
        self.res1 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res2 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res3 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res4 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)
        self.res5 = ResnetBlock(ngf*8, kernel_size=3, padding_type='reflect', norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=True)

        # Then stay for another two stages
        self.e5_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e5_cc = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.e5_norm1 = norm_layer(ngf*8)
        self.e5_norm2 = norm_layer(ngf*8)

        # decoder
        self.d5_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_dcc = spectral_norm(nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d5_norm1 = norm_layer(ngf*4)
        self.d5_norm2 = norm_layer(ngf*4)

        self.d6_dc = spectral_norm(nn.ConvTranspose2d(ngf*4*3, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_dcc = spectral_norm(nn.Conv2d(ngf*2, ngf*2, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d6_norm1 = norm_layer(ngf*2)
        self.d6_norm2 = norm_layer(ngf*2)

        self.d7_dc = spectral_norm(nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_dcc = spectral_norm(nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1), use_spectral_norm)
        self.d7_norm1 = norm_layer(ngf)
        self.d7_norm2 = norm_layer(ngf)

        self.d8_dc = spectral_norm(nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        # construct shift and innerCos
        self.shift = InnerShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=3)
        self.shift.set_mask(mask_global)
        shift_list.append(self.shift)

        self.innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=3)
        self.innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(self.innerCos)

    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2_1 = self.e2_norm1(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e2_2 = self.e2_norm2(self.e2_cc(F.leaky_relu_(e2_1, negative_slope=0.2)))

        e3_1 = self.e3_norm1(self.e3_c(F.leaky_relu_(e2_2, negative_slope=0.2)))
        e3_2 = self.e3_norm2(self.e3_cc(F.leaky_relu_(e3_1, negative_slope=0.2)))

        e4_1 = self.e4_norm1(self.e4_c(F.leaky_relu_(e3_2, negative_slope=0.2)))
        e4_2 = self.e4_norm2(self.e4_cc1(F.leaky_relu_(e4_1, negative_slope=0.2)))
        e4_3 = self.e4_norm3(self.e4_cc2(F.leaky_relu_(e4_2, negative_slope=0.2)))

        e4_3_tmp = e4_3
        # dilated convs
        res1 = self.res1(F.relu_(e4_3))
        res2 = self.res2(F.relu_(res1))
        res3 = self.res3(F.relu_(res2))
        res4 = self.res4(F.relu_(res3))
        res5 = self.res5(F.relu_(res4))

        res5_f = res5 + e4_3_tmp

        e5_1 = self.e5_norm1(self.e5_c(F.relu_(res5_f)))
        e5_2 = self.e5_norm2(self.e5_cc(F.relu_(e5_1)))

        # Decoder
        d5_1 = self.d5_norm1(self.d5_dc(F.relu_(torch.cat([e5_2, e4_3], dim=1))))
        d5_2 = self.d5_norm2(self.d5_dcc(F.relu_(d5_1)))
        d6_1 = self.d6_norm1(self.d6_dc(self.shift(self.innerCos(F.relu_(torch.cat([d5_2, e3_2], dim=1))))))
        d6_2 = self.d6_norm2(self.d6_dcc(F.relu_(d6_1)))
        d7_1 = self.d7_norm1(self.d7_dc(F.relu_(torch.cat([d6_2, e2_2], dim=1))))
        d7_2 = self.d7_norm2(self.d7_dcc(F.relu_(d7_1)))
        # No norm on the last layer
        d8 = self.d8_dc(F.relu_(torch.cat([d7_2, e1], 1)))

        d8 = torch.tanh(d8)

        return d8

class ResnetGenerator_shift(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, padding_type='reflect', use_spectral_norm=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_shift, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, kernel_size=3, padding_type=padding_type, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

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
