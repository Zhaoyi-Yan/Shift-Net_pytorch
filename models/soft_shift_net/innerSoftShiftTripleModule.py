from util.NonparametricShift import Modified_NonparametricShift
from torch.nn import functional as F
import torch.nn as nn
import torch


class InnerSoftShiftTripleModule(nn.Module):
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag):
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real

        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        ctx.ind_lst = ctx.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_()

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### decoder feature
        latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all) # addition feature

        assert mask.dim() == 2, "Mask dimension must be 2"

        if torch.cuda.is_available:
            flag = flag.cuda()

        # None batch version
        Nonparm = Modified_NonparametricShift()

        for idx in range(ctx.bz):
            latter = latter_all.narrow(0, idx, 1) ### encoder feature
            former = former_all.narrow(0, idx, 1) ### decoder feature

            #GET COSINE, RESHAPED LATTER AND ITS INDEXES
            cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag)

            ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
            cosine_softmax = F.softmax(cosine, dim=1)

            mask_indexes = (flag == 1).nonzero()
            non_mask_indexes = (flag == 0).nonzero()
            ctx.ind_lst[idx][mask_indexes, non_mask_indexes.t()] = cosine_softmax

            # GET FINAL SHIFT FEATURE
            shift_masked_all[idx] = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1, i_4)

        return torch.cat((former_all, latter_all, shift_masked_all), 1)
