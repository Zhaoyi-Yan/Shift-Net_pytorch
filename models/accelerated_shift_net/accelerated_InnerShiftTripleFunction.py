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

        ctx.ind_lst = torch.cuda.FloatTensor(ctx.bz, ctx.h*ctx.w, ctx.h*ctx.w).fill_(0)

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### UPCONV
        latter_all = input.narrow(1, c//2, c//2) ### UNET ADD

        assert mask.dim() == 2, "Mask dimension must be 2"

        if torch.cuda.is_available:
            ctx.ind_lst = ctx.ind_lst.cuda()
            flag = flag.cuda()

        # None batch version
        Nonparm = Modified_NonparametricShift()
        shift_masked = None
        for idx in range(ctx.bz):
            latter = latter_all.narrow(0, idx, 1) ### UNET ADD
            former = former_all.narrow(0, idx, 1) ### UPCONV

            #GET COSINE, RESHAPED LATTER AND ITS INDEXES
            cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag)

            ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
            _, indexes = torch.max(cosine, dim=1)

            # SET  TRANSITION MATRIX
            mask_indexes = (flag == 1).nonzero()
            non_mask_indexes = (flag == 0).nonzero()[indexes]
            ctx.ind_lst[idx][tuple((mask_indexes, non_mask_indexes))] = 1

            # TRANSITION MATRIX
            former_masked = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1, i_4)

            former_masked = former_masked.detach()
            if idx == 0:
                shift_masked = former_masked
            else:
                shift_masked = torch.cat((shift_masked, former_masked), 0)

        return torch.cat((former_all, latter_all, shift_masked), 1)


    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :]#.cuda()
        grad_swapped_all = grad_output[:, c*2//3:c, :, :]#.cuda()

        for idx in range(ctx.bz):

            ## MASK TO NON-MASK (NOT NEEDED TO TRANSPOSE)
            W_mat_t = ind_lst[idx].cuda().t()

            grad = grad_swapped_all[idx].view(c//3, -1).t()

            grad_swapped_weighted = torch.mm(W_mat_t, grad)

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
            grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_swapped_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None, None, None, None, None, None
    '''
        # None batch version
        for idx in range(self.bz):
            latter = latter_all.narrow(0, idx, 1) ### UNET ADD
            former = former_all.narrow(0, idx, 1) ### UPCONV

            #print(latter.shape)
            #print(former.shape)
            Nonparm = Modified_NonparametricShift()

            ## EXTRACT MASK PATCHES FROM FORMER
            patches_former = Nonparm._extract_patches_from_flag(former.clone().squeeze(), 1, stride, flag, 1)

            ## EXTRACT NON-MASK PATCHES FROM FORMER
            patches_latter = Nonparm._extract_patches_from_flag(latter.clone().squeeze(), 1, stride, flag, 0)

            ## CALCULATE ABSOLUTE COSINE SIMILARITY
            multi = torch.abs(torch.matmul(patches_former, patches_latter.permute(1,0)))

            ## CALCULATE SOFTMAX OVER NON-MASK DIMENSION
            multi = F.softmax(multi, dim=1) ###

            ## CALCULATE softmax_weight * X(NON-MASK)
            patches_former = torch.matmul(multi, patches_latter)

            ## ADD PREVIOUS MASK PATCH + NEW ATTENTION COMBINAISON OF OTHERS FEATURES

            former_masked = Nonparm._paste(former.clone().squeeze(), 1, stride, flag, patches_former)

            former_masked.masked_fill_(inv_ex_mask, 0)

            former_masked = former_masked.detach().clone()

            #former_masked = Nonparm._paste(former_masked.squeeze(), 1, stride, flag, patches_former)

        return torch.cat((former, latter, former_masked), 1)

    '''