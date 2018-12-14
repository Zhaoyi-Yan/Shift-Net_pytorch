import torch
import util.util as util
from util.NonparametricShift import Modified_NonparametricShift

former = torch.rand(16).mul_(50).reshape(1, 4, 4).int().float()
latter = torch.rand(16).mul_(50).reshape(1, 4, 4).int().float()

print(former.size())
print(former.type())

flag = torch.zeros(4,4).byte()
flag[1:3, 1:3] = 1
flag = flag.view(16)

ind_lst = torch.FloatTensor(16, 16).zero_()

Nonparm = Modified_NonparametricShift()
cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former, latter, 1, 1, flag, True)
## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY

print('cosine')
print(cosine)
_, indexes = torch.max(cosine, dim=1)
print('index')
print(indexes)

# SET  TRANSITION MATRIX
mask_indexes = (flag == 1).nonzero()
non_mask_indexes = (flag == 0).nonzero()[indexes]
ind_lst[mask_indexes, non_mask_indexes.t()] = 1

# GET FINAL SHIFT FEATURE
shift_masked_all = Nonparm._paste(latter_windows, ind_lst, i_2, i_3, i_1, i_4)

print('former')
print(former)
print('latter')
print(latter)
print('flag')
print(flag.reshape(4,4))
print('ind_lst')
print(ind_lst)
print(shift_masked_all)

