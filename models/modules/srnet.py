import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import spectral_norm, ResnetBlock

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, use_spectral_norm=use_spectral_norm)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)

# construct network from the inside to the outside.
# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        if input_nc is None:
            input_nc = outer_nc

        downconv = spectral_norm(nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1), use_spectral_norm)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

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
            upconv = spectral_norm(nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), use_spectral_norm)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

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


class SR_C(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d):
        super(SR_C, self).__init__()
        self.c1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(16):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]

        self.res_bs = nn.Sequential(*res_bs)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # 4x upscale
        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)
        self.c4 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up2 = nn.PixelShuffle(2)

        self.c_f = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=1)
        self.out_act = nn.Tanh()

    def forward(self, input):
        out = F.relu_(self.c1(input))
        out = self.res_bs(out)
        out = self.c2(out) # no relu here
        out = self.c2_norm(out)

        content_feat = out
        out = F.relu_(self.pixel_up1(self.c3(out)))
        out = F.relu_(self.pixel_up2(self.c4(out)))
        out_f = self.out_act(self.c_f(out))

        return out_f, content_feat


# Texture block
#
#
class SR_T(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(SR_T, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.c1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(self.num_res_blocks):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]
        self.res_bs = nn.Sequential(*res_bs)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # upscale 2
        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)

    def forward(self, input):
        map_all = torch.cat((map_in, map_ref), dim=1)
        out = self.c1(map_all)
        out = self.res_bs(out)

        out = self.c2_norm(self.c2(out))
        out = out + map_in
        # upscale 2
        out = F.relu_(self.pixel_up1(self.c3(input)))

# Fusion block: Fuse content and texture maps
class SR_F(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(SR_F, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.c1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(self.num_res_blocks):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]
        self.res_bs = nn.Sequential(*res_bs)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # upscale 2
        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)

    def forward(self, input):
        map_all = torch.cat((map_in, map_ref), dim=1)
        out = self.c1(map_all)
        out = self.res_bs(out)

        out = self.c2_norm(self.c2(out))
        out = out + map_in
        # upscale 2
        out = F.relu_(self.pixel_up1(self.c3(input)))


# It is an easy type of UNet, intead of constructing UNet with UnetSkipConnectionBlocks.
# In this way, every thing is much clear and more flexible for extension.
class EasyUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_spectral_norm=False):
        super(EasyUnetGenerator, self).__init__()

        # Encoder layers
        self.e1_c = spectral_norm(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)

        self.e2_c = spectral_norm(nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.e2_norm = norm_layer(ngf*2)

        self.e3_c = spectral_norm(nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
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
        self.d1_c = spectral_norm(nn.ConvTranspose2d(ngf*8, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d1_norm = norm_layer(ngf*8)

        self.d2_c = spectral_norm(nn.ConvTranspose2d(ngf*8*2 , ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d2_norm = norm_layer(ngf*8)

        self.d3_c = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d3_norm = norm_layer(ngf*8)

        self.d4_c = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*8, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d4_norm = norm_layer(ngf*8)

        self.d5_c = spectral_norm(nn.ConvTranspose2d(ngf*8*2, ngf*4, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d5_norm = norm_layer(ngf*4)

        self.d6_c = spectral_norm(nn.ConvTranspose2d(ngf*4*2, ngf*2, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d6_norm = norm_layer(ngf*2)

        self.d7_c = spectral_norm(nn.ConvTranspose2d(ngf*2*2, ngf, kernel_size=4, stride=2, padding=1), use_spectral_norm)
        self.d7_norm = norm_layer(ngf)

        self.d8_c = spectral_norm(nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1), use_spectral_norm)


    # In this case, we have very flexible unet construction mode.
    def forward(self, input):
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
        d1 = self.d1_norm(self.d1_c(F.relu_(e8)))
        d2 = self.d2_norm(self.d2_c(F.relu_(torch.cat([d1, e7], dim=1))))
        d3 = self.d3_norm(self.d3_c(F.relu_(torch.cat([d2, e6], dim=1))))
        d4 = self.d4_norm(self.d4_c(F.relu_(torch.cat([d3, e5], dim=1))))
        d5 = self.d5_norm(self.d5_c(F.relu_(torch.cat([d4, e4], dim=1))))
        d6 = self.d6_norm(self.d6_c(F.relu_(torch.cat([d5, e3], dim=1))))
        d7 = self.d7_norm(self.d7_c(F.relu_(torch.cat([d6, e2], dim=1))))
        # No norm on the last layer
        d8 = self.d8_c(F.relu_(torch.cat([d7, e1], 1)))

        d8 = torch.tanh(d8)

        return d8
