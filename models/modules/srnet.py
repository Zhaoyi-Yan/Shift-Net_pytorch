import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import spectral_norm, ResnetBlock



# Out is the the output of 2x/4x RGB images.
# content_feat is the content feature of original size.
class SR_C(nn.Module):
    def __init__(self, input_nc=3, upsacle=4, norm_layer=nn.BatchNorm2d):
        super(SR_C, self).__init__()
        self.c1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1)
        self.upscale = upsacle

        res_bs = []
        for i in range(16):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]

        self.res_bs = nn.Sequential(*res_bs)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # 4x upscale
        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)

        if upsacle == 4:
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

        if self.upscale:
            out = F.relu_(self.pixel_up2(self.c4(out)))
        out_f = self.out_act(self.c_f(out))

        return out_f, content_feat


# Texture block
# Input: the content_feat of SR_C, channel=64
#   maps: the shift feature (in relu3_1), channels=64
class SR_T(nn.Module):
    def __init__(self, input_nc=64, ref_nc=256, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(SR_T, self).__init__()
        self.num_res_blocks = num_res_blocks

        self.c1 = nn.Conv2d(input_nc + ref_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(self.num_res_blocks):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]
        self.res_bs = nn.Sequential(*res_bs)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # upscale 2
        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)

    # The input should be features coming from SR_C
    def forward(self, input, maps):
        # change me later
        map_ref = maps[0]
        map_all = torch.cat((input, map_ref), dim=1)
        # constructing residual info
        out = self.c1(map_all)
        out = self.res_bs(out)
        out = self.c2_norm(self.c2(out))

        # add residual info
        out = out + input
        # upscale 2
        out = F.relu_(self.pixel_up1(self.c3(out)))

# Fusion block: Fuse content and texture maps
# Input should be the features of SR_T
#  maps: the shift feature (in relu2_1)
class SR_F_1(nn.Module):
    def __init__(self, input_nc=64, ref_nc=128, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(SR_F_1, self).__init__()
        self.num_res_blocks = num_res_blocks

        self.c1 = nn.Conv2d(input_nc + ref_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(self.num_res_blocks // 2):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]
        self.res_bs = nn.Sequential(*res_bs)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        self.c3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_up1 = nn.PixelShuffle(2)

        
    # using maps[1]
    def forward(self, input, maps):
        map_ref = maps[0]
        map_all = torch.cat((input, map_ref), dim=1)

        # constructing residual info
        out = self.c1(map_all)
        out = self.res_bs(out)
        out = self.c2_norm(self.c2(out))

        out = out + input
        # upscale 2
        out = F.relu_(self.pixel_up1(self.c3(out)))

        return out

# Fusion block: Fuse content and texture maps
# Input should be the features of SR_T
#  maps: the shift feature (in relu1_1)
class SR_F_2(nn.Module):
    def __init__(self, input_nc=64, ref_nc=64, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(SR_F_2, self).__init__()
        self.num_res_blocks = num_res_blocks
        self.c1 = nn.Conv2d(input_nc + ref_nc, 64, kernel_size=3, stride=1, padding=1)
        res_bs = []
        for i in range(self.num_res_blocks // 4):
            res_bs += [ResnetBlock(64, 3, 'zero', norm_layer, use_spectral_norm=False, use_bias=True)]
        self.res_bs = nn.Sequential(*res_bs)

        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c2_norm = norm_layer

        # Directly reduce the dim
        self.c3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.c_f = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=1)
        self.out_act = nn.Tanh()
        
    # using maps[1]
    def forward(self, input, maps):
        map_ref = maps[0]
        map_all = torch.cat((input, map_ref), dim=1)

        # constructing residual info
        out = self.c1(map_all)
        out = self.res_bs(out)
        out = self.c2_norm(self.c2(out))

        out = out + input
        # upscale 2
        out = self.out_act(self.c_f(F.relu_(self.c3(out))))

        return out

# Combine models
# shift_maps: offline shift features.
#  shift_maps[0]: relu3_1
#  shift_maps[1]: relu2_1
#  shift_maps[2]: relu1_1
class m64_UP_1(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, num_res_blocks=16):
        super(m64_UP_1, self).__init__()
        self.sr_c = SR_C(input_nc=3, upsacle=4)
        self.sr_t = SR_T(input_nc=64, ref_nc=256)
        self.sr_f_1 = SR_F_1(input_nc=64, ref_nc=128)
        self.sr_f_2 = SR_F_2(input_nc=64, ref_nc=64)

    def forward(self, input, shift_maps):
        direct_content_up, content_feat = self.sr_c(input)
        out_feat = self.sr_t(content_feat, shift_maps[0])
        out_fusion_1 = self.sr_f_1(out_feat, shift_maps[1])
        out_fusion_2 = self.sr_f_2(out_fusion_1, shift_maps[2])

        return out_fusion_2, direct_content_up

