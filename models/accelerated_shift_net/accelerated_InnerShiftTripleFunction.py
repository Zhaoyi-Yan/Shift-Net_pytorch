import numpy as np
from util.NonparametricShift import Modified_NonparametricShift
import torch
from time import time


class AcceleratedInnerShiftTripleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag):
        #print('[INFO] GET INTO FORWARD')

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
            _, indexes = torch.max(cosine, dim=1)

            # SET  TRANSITION MATRIX
            mask_indexes = (flag == 1).nonzero()
            non_mask_indexes = (flag == 0).nonzero()[indexes]
            ctx.ind_lst[idx][mask_indexes, non_mask_indexes.t()] = 1

            # GET FINAL SHIFT FEATURE
            shift_masked_all[idx] = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1, i_4)


        return torch.cat((former_all, latter_all, shift_masked_all), 1)



    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :].clone()
        grad_shifted_all = grad_output[:, c*2//3:c, :, :].clone()

        for idx in range(ctx.bz):

            # C: content, pixels in masked region of the former part.
            # S: style, pixels in the non-masked region of the latter part.
            # N: the shifted feature, the new feature that will be used as the third part of features maps.
            # W_mat: ind_lst[idx], shift matrix.
            # Note: **only the masked region in N has values**.

            # The gradient of shift feature should be added back to the latter part(to be precise: S).
            # `ind_lst[idx][i,j] = 1` means that the i_th pixel will **be replaced** by j_th pixel in the forward.
            # When applying `S mm W_mat`, then S will be transfer to N. 
            # (pixels in non-masked region of the latter part will be shift to the masked region in the third part.)
            # However, we need to transfer back the gradient of the third part to S.
            # This means the graident in S will **`be replaced`(to be precise, enhanced)** by N.
            
            # So we need to transpose `W_mat`
            W_mat_t = ind_lst[idx].t()

            grad = grad_shifted_all[idx].view(c//3, -1).t()

            grad_shifted_weighted = torch.mm(W_mat_t, grad)

            # Then transpose it back
            grad_shifted_weighted = grad_shifted_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
            grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_shifted_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None
