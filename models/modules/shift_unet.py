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
###################################         Shift_net            #####################################
################################### ***************************  #####################################
# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = UnetSkipConnectionShiftBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class UnetSkipConnectionShiftBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(UnetSkipConnectionShiftBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerShiftTriple(opt.fixed_mask, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight)

        shift.set_mask(mask_global, 3)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = ResUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class ResUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(ResUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResShiftTriple(opt.fixed_mask, inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight)

        shift.set_mask(mask_global, 3)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # Res shift differs with other shift here. It is `*2` not `*3`.
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = SoftUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                 mask_global, input_nc=None, \
                                                                 submodule=unet_block,
                                                                 norm_layer=norm_layer, shift_layer=True)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class SoftUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SoftUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerSoftShiftTriple(opt.fixed_mask, opt.shift_sz, opt.stride, opt.mask_thred, opt.triple_weight)

        shift.set_mask(mask_global, 3)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCosBefore = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCosBefore.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCosBefore)

        # Different position only has differences in `upconv`
            # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCosBefore, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(PatchSoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = PatchSoftUnetSkipConnectionShiftTriple(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class PatchSoftUnetSkipConnectionShiftTriple(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(PatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerPatchSoftShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse)

        shift.set_mask(mask_global, 3)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # shift triple differs in here. It is `*3` not `*2`.
            upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ResPatchSoftUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        print(unet_block)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = ResPatchSoftUnetSkipConnectionShiftTriple(ngf * 2, ngf * 4, opt, innerCos_list, shift_list,
                                                                    mask_global, input_nc=None, \
                                                                    submodule=unet_block,
                                                                    norm_layer=norm_layer)  # passing in unet_shift_block
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Mention: the TripleBlock differs in `upconv` defination.
# 'cos' means that we add a `innerCos` layer in the block.
class ResPatchSoftUnetSkipConnectionShiftTriple(nn.Module):
    def __init__(self, outer_nc, inner_nc, opt, innerCos_list, shift_list, mask_global, input_nc, \
                 submodule=None, shift_layer=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False):
        super(ResPatchSoftUnetSkipConnectionShiftTriple, self).__init__()
        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResPatchSoftShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse)

        shift.set_mask(mask_global, 3)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip)
        innerCos.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(innerCos)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            # Res shift differs with other shift here. It is `*2` not `*3`.
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            # shift should be placed after uprelu
            # NB: innerCos are placed before shift. So need to add the latent gredient to
            # to former part.
            up = [uprelu, innerCos, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
###################################       InceptionUnet          #####################################
################################### ***************************  #####################################
class InceptionUnetGeneratorShiftTriple(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, opt, innerCos_list, shift_list, mask_global, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(InceptionUnetGeneratorShiftTriple, self).__init__()

        # construct unet structure
        unet_block = InceptionUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer,
                                             innermost=True)
        for i in range(num_downs - 5):  # The innner layers number is 3 (sptial size:512*512), if unet_256.
            unet_block = InceptionUnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                 norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = InceptionUnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                             norm_layer=norm_layer)

        unet_shift_block = InceptionShiftUnetSkipConnectionBlock(ngf * 2, ngf * 4, opt=opt, innerCos_list=innerCos_list, shift_list=shift_list,
                                                                 mask_global=mask_global, input_nc=None, \
                                                                 submodule=unet_block,
                                                                 norm_layer=norm_layer, shift_layer=True)  # passing in unet_shift_block # innerCos_list=None, shift_list=None, mask_global=None, input_nc=None, opt=None,\submodule=None, shift_layer=False, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False

        unet_block = InceptionUnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_shift_block,
                                             norm_layer=norm_layer)
        unet_block = InceptionUnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True,
                                             norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class InceptionUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(InceptionUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = InceptionDown(input_nc, inner_nc) # nn.Conv2d(input_nc, inner_nc, kernel_size=4,stride=2, padding=1)


        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1)
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1)

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = InceptionUp(inner_nc, outer_nc) #nn.ConvTranspose2d(inner_nc, outer_nc,kernel_size=4, stride=2,padding=1)

            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            upconv = InceptionUp(inner_nc * 2, outer_nc) #nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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


# construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class InceptionShiftUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, innerCos_list=None, shift_list=None, mask_global=None, input_nc=None, opt=None,\
                 submodule=None, shift_layer=False, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(InceptionShiftUnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost
        if input_nc is None:
            input_nc = outer_nc

        if shift_layer:
            # As the downconv layer is outer_nc in and inner_nc out.
            # So the shift define like this:
            shift = InnerShiftTriple(opt.fixed_mask, opt.shift_sz, opt.stride, opt.mask_thred, opt.triple_weight)

            shift.set_mask(mask_global, 3)
            shift_list.append(shift)

            # Add latent constraint
            # Then add the constraint to the constrain layer list!
            innerCosBefore = InnerCos(strength=opt.strength, skip=opt.skip)
            innerCosBefore.set_mask(mask_global, 3)  # Here we need to set mask for innerCos layer too.
            innerCos_list.append(innerCosBefore)

        downconv = InceptionDown(input_nc, inner_nc) # nn.Conv2d(input_nc, inner_nc, kernel_size=4,stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        # Different position only has differences in `upconv`
        # for the outermost, the special is `tanh`
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1)
            downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                                 stride=2, padding=1)

            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
            # for the innermost, the special is `inner_nc` instead of `inner_nc*2`
        elif innermost:
            upconv = InceptionUp(inner_nc, outer_nc) #nn.ConvTranspose2d(inner_nc, outer_nc,kernel_size=4, stride=2,padding=1)

            down = [downrelu, downconv]  # for the innermost, no submodule, and delete the bn
            up = [uprelu, upconv, upnorm]
            model = down + up
            # else, the normal
        else:
            upconv = InceptionUp(inner_nc * 3, outer_nc) #nn.ConvTranspose2d(inner_nc * 2, outer_nc,kernel_size=4, stride=2,padding=1)

            down = [downrelu, downconv, downnorm]
            up = [uprelu, innerCosBefore, shift, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
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
            return torch.cat([x_latter, x], 1) # cat in the C channel