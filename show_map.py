# -*- coding: UTF-8 -*-
import torch
import util.util as util
from util.NonparametricShift import Modified_NonparametricShift
from torch.nn import functional as F
import numpy as numpy
import matplotlib.pyplot as plt

bz = 1
c = 2 # at least 2
w = 4
h = 4

feature_size = [bz, c, w, h]

former = torch.rand(c*h*w).mul_(50).reshape(c, h, w).int().float()
latter = torch.rand(c*h*w).mul_(50).reshape(c, h, w).int().float()


flag = torch.zeros(h,w).byte()
flag[h//4:h//2+1, h//4:h//2+1] = 1
flag = flag.view(h*w)

ind_lst = torch.FloatTensor(h*w, h*w).zero_()
shift_offsets = []

Nonparm = Modified_NonparametricShift()
cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former, latter, 1, 1, flag)
## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY

_, indexes = torch.max(cosine, dim=1)


# SET  TRANSITION MATRIX
mask_indexes = (flag == 1).nonzero()
non_mask_indexes = (flag == 0).nonzero()[indexes]
ind_lst[mask_indexes, non_mask_indexes] = 1


# GET FINAL SHIFT FEATURE
shift_masked_all = Nonparm._paste(latter_windows, ind_lst, i_2, i_3, i_1, i_4)

print('flag')
print(flag.reshape(h,w))
print('ind_lst')
print(ind_lst)
print('out')
print(shift_masked_all)

# get shift offset ()
shift_offset = torch.stack([non_mask_indexes.squeeze() // w, torch.fmod(non_mask_indexes.squeeze(), w)], dim=-1)


shift_offsets.append(shift_offset)
shift_offsets = torch.cat(shift_offsets, dim=0).float()
print('shift_offset')
print(shift_offset)
print(shift_offset.size())  # (5*5)*2 (masked points)

shift_offsets_cl = shift_offsets.clone()


#visualize which pixels are attended
print(flag.size()) # 256, (16*16)


# global and N*C*H*W
# put shift_offsets_cl back to the global map.
shift_offsets_map = flag.clone().view(1, h,w, 1).expand(bz,h,w,2).float()
print(shift_offsets_map.size()) # 1*16*16

# mask_indexes 是对应的mask区域的点的位置。
# shift_offsets是对应的要shift到mask区域的外部点的位置。
shift_offsets_map[:, mask_indexes.squeeze() // w, mask_indexes.squeeze() % w, :] = shift_offsets_cl.unsqueeze(0)
# 至此，shift_offsets_map是完整的，而且只有mask内部有值，代表着该点将被外面的某点替换。“某点”的坐标就是该点的值（2个通道）
print('global shift_offsets_map')
print(shift_offsets_map)
print(shift_offsets_map.size())
print(shift_offsets_map.type())

flow2 = torch.from_numpy(util.highlight_flow(shift_offsets_map, flag.unsqueeze(0)))
print('flow2 size')
print(flow2.size())

flow2 = flow2.permute(0, 3, 1, 2)
# upflow = F.interpolate(flow, scale_factor=4, mode='nearest')
upflow2 = F.interpolate(flow2, scale_factor=1, mode='nearest')

print('**After upsample flow2 size**')
print(upflow2.size())

# upflow = upflow.squeeze().permute(1,2,0)
upflow2 = upflow2.squeeze().permute(1,2,0)
print(upflow2.size())

# print('flow 1')
# print(upflow)
# print(upflow.size())

# print('flow 2')
# print(upflow2)
# print(upflow2.size())
plt.imshow(upflow2/255.)
# # axs[0].imshow(upflow)
# axs[1].imshow(upflow2)

plt.show()
