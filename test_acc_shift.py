import torch
import util.util as util
from util.NonparametricShift import Modified_NonparametricShift
from torch.nn import functional as F
import numpy as numpy
import matplotlib.pyplot as plt

bz = 1
c = 2 # at least 2
w = 24*2
h = 24*2

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

print('cosine')
print(cosine)
_, indexes = torch.max(cosine, dim=1)


# SET  TRANSITION MATRIX
mask_indexes = (flag == 1).nonzero()
non_mask_indexes = (flag == 0).nonzero()[indexes]
ind_lst[mask_indexes, non_mask_indexes] = 1

for mi, nmi in zip(mask_indexes, non_mask_indexes):
    print('The %d\t-th pixel shifts to %d\t-th coordinate' %(nmi, mi))

# GET FINAL SHIFT FEATURE
shift_masked_all = Nonparm._paste(latter_windows, ind_lst, i_2, i_3, i_1, i_4)

print('former')
print(former)
print('latter')
print(latter)
print('flag')
print(flag.reshape(h,w))
print('ind_lst')
print(ind_lst)
print('out')
print(shift_masked_all)

# get shift offset
shift_offset = torch.stack([non_mask_indexes.squeeze() // w, torch.fmod(non_mask_indexes.squeeze(), w)], dim=-1)
print()
print('shift_offset')
print(shift_offset)
print(shift_offset.size())

shift_offsets.append(shift_offset)
shift_offsets = torch.cat(shift_offsets, dim=0).float()
print(shift_offsets.size())
print(shift_offsets)

lt = (flag==1).nonzero()[0]
rb = (flag==1).nonzero()[-1]

mask_h = rb//w+1 - lt//w
mask_w = rb%w+1 - lt%w

shift_offsets =  shift_offsets.view([bz] + [2] + [mask_h, mask_w]) # So only appropriate for square mask.
print(shift_offsets.size())
print(shift_offsets)

h_add = torch.arange(0, float(h)).view([1, 1, h, 1]).float()
h_add = h_add.expand(bz, 1, h, w)
w_add = torch.arange(0, float(w)).view([1, 1, 1, w]).float()
w_add = w_add.expand(bz, 1, h, w)

com_map = torch.cat([h_add, w_add], dim=1)
print('com_map')
print(com_map)

com_map_crop = com_map[:, :, lt//w:rb//w+1, lt%w:rb%w+1]
print('com_map crop')
print(com_map_crop)

shift_offsets = shift_offsets - com_map_crop
print('final shift_offsets')
print(shift_offsets)


# to flow image
flow = torch.from_numpy(util.flow_to_image(shift_offsets.permute(0,2,3,1).cpu().data.numpy()))
flow = flow.permute(0,3,1,2)

#visualize which pixels are attended
print(flag.size())
print(shift_offsets.size())
flow2 = torch.from_numpy(util.highlight_flow((shift_offsets * flag.int()).numpy()))

upflow = F.interpolate(flow, scale_factor=4, mode='nearest')
upflow2 = F.interpolate(flow2, scale_factor=4, mode='nearest')


upflow = upflow.squeeze().permute(1,2,0)
upflow2 = upflow2.squeeze().permute(1,2,0)

print('flow 1')
print(upflow)
print(upflow.size())

print('flow 2')
print(upflow2)
print(upflow2.size())

fig, axs = plot.subplots(ncols=2)
axs[0].imshow(upflow)
axs[1].imshow(upflow2)

plt.show()
