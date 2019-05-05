import numpy as np
from .NonparametricShift import Modified_NonparametricShift
import torch
import util as util
import time



def shift_offline(input, shift_sz, stride, flag):
    flag = flag

    bz, c_real, h, w = input.size()

    ind_lst = torch.Tensor(bz, h * w, h * w).zero_().to(input)

    # former and latter are all tensors
    former_all = input.clone()
    latter_all = input.clone()
    shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all).zero_() # addition feature

    flag = flag.to(input).long()

    # None batch version
    Nonparm = Modified_NonparametricShift()

    for idx in range(bz):
        flag_cur = flag[idx]
        latter = latter_all.narrow(0, idx, 1) ### encoder feature
        former = former_all.narrow(0, idx, 1) ### decoder feature

        #GET COSINE, RESHAPED LATTER AND ITS INDEXES
        cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag_cur)

        ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
        _, indexes = torch.max(cosine, dim=1)

        # SET  TRANSITION MATRIX
        mask_indexes = (flag_cur == 1).nonzero()
        non_mask_indexes = (flag_cur == 0).nonzero()[indexes]
        ind_lst[idx][mask_indexes, non_mask_indexes] = 1

        # GET FINAL SHIFT FEATURE
        shift_masked_all[idx] = Nonparm._paste(latter_windows, ind_lst[idx], i_2, i_3, i_1, i_4)

    return shift_masked_all

        
