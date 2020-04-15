import torch
import torch.nn as nn
import torch.nn.functional as F

# For original shift
from models.shift_net.InnerShiftTriple import InnerShiftTriple
from models.shift_net.InnerCos import InnerCos

# for face shift
from models.face_shift_net.InnerFaceShiftTriple import InnerFaceShiftTriple

# For res shift
from models.res_shift_net.innerResShiftTriple import InnerResShiftTriple

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

        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=layer_to_last, device=device)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last, device=device)
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
###################################         Face Shift_net       #####################################
################################### ***************************  #####################################
class FaceUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, innerCos_list, shift_list, mask_global, opt, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(FaceUnetGenerator, self).__init__()

        # Encoder layers
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf*2)

        self.e3_c = spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=6, stride=2, padding=2), use_spectral_norm)
        self.e3_norm = norm_layer(ngf*4)

        self.e4_c = spectral_norm(nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e4_norm = norm_layer(ngf*8)

        self.e5_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e5_norm = norm_layer(ngf*8)

        self.e6_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e6_norm = norm_layer(ngf*8)

        self.e7_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e7_norm = norm_layer(ngf*8)

        self.e8_c = spectral_norm(nn.Conv2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)


        # Deocder layers
        self.d1_dc = spectral_norm(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d1_norm = norm_layer(ngf*8)

        self.d2_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2 , ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d2_norm = norm_layer(ngf*8)

        self.d3_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d3_norm = norm_layer(ngf*8)

        self.d4_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d4_norm = norm_layer(ngf*8)

        self.d5_dc = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_norm = norm_layer(ngf*4)

        # shift before this layer
        self.d6_dc = spectral_norm(nn.ConvTranspose2d(ngf*4*3, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_norm = norm_layer(ngf*2)

        self.d7_dc = spectral_norm(nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_norm = norm_layer(ngf)

        self.d8_dc = spectral_norm(nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        # construct shift and innerCos
        self.shift = InnerFaceShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=3)
        self.shift.set_mask(mask_global)
        shift_list.append(self.shift)

        self.innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=3)
        self.innerCos.set_mask(mask_global)  # Here we need to set mask for innerCos layer too.
        innerCos_list.append(self.innerCos)
        
    # In this case, we have very flexible unet construction mode.
    def forward(self, input, flip_feat=None):
        # Encoder
        # No norm on the first layer
        e1 = self.e1_c(input)
        e2 = self.e2_norm(self.e2_c(F.leaky_relu_(e1, negative_slope=0.2)))
        e3 = self.e3_norm(self.e3_c(F.leaky_relu_(e2, negative_slope=0.2)))
        e4 = self.e4_norm(self.e4_c(F.leaky_relu_(e3, negative_slope=0.2)))
        e5 = self.e5_norm(self.e5_c(F.leaky_relu_(e4, negative_slope=0.2)))
        e6 = self.e6_norm(self.e6_c(F.leaky_relu_(e5, negative_slope=0.2)))

        e7 = self.e7_norm(self.e7_c(F.leaky_relu_(e6, negative_slope=0.2)))
        # No norm on the inner_most layer
        e8 = self.e8_c(F.leaky_relu_(e7, negative_slope=0.2))

        # Decoder
        d1 = self.d1_norm(self.d1_dc(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_dc(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_dc(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_dc(F.relu_(torch.cat([d3, e5], dim=1))))
        d5 = self.d5_norm(self.d5_dc(F.relu_(torch.cat([d4, e4], dim=1))))
        tmp, innerFeat = self.shift(self.innerCos(F.relu_(torch.cat([d5, e3], dim=1))), flip_feat)
        d6 = self.d6_norm(self.d6_dc(tmp))
        d7 = self.d7_norm(self.d7_dc(F.relu_(torch.cat([d6, e2], dim=1))))
        # No norm on the last layer
        d8 = self.d8_dc(F.relu_(torch.cat([d7, e1], 1)))

        d8 = torch.tanh(d8)

        return d8, innerFeat

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

        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, layer_to_last=layer_to_last, device=device)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last, device=device)
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

        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerPatchSoftShiftTriple(opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse, layer_to_last=layer_to_last, device=device)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last, device=device)
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

        device = 'cpu' if len(opt.gpu_ids) == 0 else 'gpu'
        # As the downconv layer is outer_nc in and inner_nc out.
        # So the shift define like this:
        shift = InnerResPatchSoftShiftTriple(inner_nc, opt.shift_sz, opt.stride, opt.mask_thred,
                                            opt.triple_weight, opt.fuse, layer_to_last=layer_to_last, device=device)

        shift.set_mask(mask_global)
        shift_list.append(shift)

        # Add latent constraint
        # Then add the constraint to the constrain layer list!
        innerCos = InnerCos(strength=opt.strength, skip=opt.skip, layer_to_last=layer_to_last, device=device)
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
