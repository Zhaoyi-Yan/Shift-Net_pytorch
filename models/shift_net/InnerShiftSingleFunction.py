import numpy as np
from util.NonparametricShift import Modified_NonparametricShift
import torch
import util.util as util
import time


class InnerShiftSingleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag, show_flow):
        #print('[INFO] GET INTO FORWARD')
        InnerShiftSingleFunction.ctx = ctx
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.show_flow = show_flow

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        ctx.c = c_real

        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_().to(input)

        # former and latter are all tensors
        # former_all = input.narrow(1, 0, c//2) ### decoder feature
        # latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(input.size()).type_as(input).zero_() # addition feature

        assert mask.dim() == 2, "Mask dimension must be 2"
        ctx.mask = mask

        ctx.flag = ctx.flag.to(input).long()

        # None batch version
        Nonparm = Modified_NonparametricShift()
        ctx.shift_offsets = []

        for idx in range(ctx.bz):
            # latter = latter_all.narrow(0, idx, 1) ### encoder feature
            # former = former_all.narrow(0, idx, 1) ### decoder feature
            input_i = input.narrow(0, idx, 1)

            #GET COSINE, RESHAPED LATTER AND ITS INDEXES
            cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(input_i.clone().squeeze(), input_i.clone().squeeze(), 1, stride, flag)

            ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
            _, indexes = torch.max(cosine, dim=1)

            # SET  TRANSITION MATRIX
            mask_indexes = (ctx.flag == 1).nonzero()
            non_mask_indexes = (ctx.flag == 0).nonzero()[indexes]
            ctx.ind_lst[idx][mask_indexes, non_mask_indexes] = 1

            # GET FINAL SHIFT FEATURE
            shift_masked_all[idx] = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1, i_4)

            if ctx.show_flow:
                shift_offset = torch.stack([non_mask_indexes.squeeze() // ctx.w, non_mask_indexes.squeeze() % ctx.w], dim=-1)
                ctx.shift_offsets.append(shift_offset)

        if ctx.show_flow:
            # Note: Here we assume that each mask is the same for the same batch image.
            ctx.shift_offsets = torch.cat(ctx.shift_offsets, dim=0).float() # make it cudaFloatTensor
            # Assume mask is the same for each image in a batch.
            mask_nums = ctx.shift_offsets.size(0)//ctx.bz
            ctx.flow_srcs = torch.zeros(ctx.bz, 3, ctx.h, ctx.w).type_as(input)

            for idx in range(ctx.bz):
                shift_offset = ctx.shift_offsets.narrow(0, idx*mask_nums, mask_nums)
                # reconstruct the original shift_map.
                shift_offsets_map = torch.zeros(1, ctx.h, ctx.w, 2).type_as(input)
                shift_offsets_map[:, (ctx.flag == 1).nonzero().squeeze() // ctx.w, (ctx.flag == 1).nonzero().squeeze() % ctx.w, :] = \
                                                                                                shift_offset.unsqueeze(0)
                # It is indicating the pixels(non-masked) that will shift the the masked region.
                flow_src = util.highlight_flow(shift_offsets_map, ctx.flag.unsqueeze(0))
                ctx.flow_srcs[idx] = flow_src


        return shift_masked_all + input.masked_fill_(mask.view(1, 1, ctx.h, ctx.w).expand(ctx.bz, ctx.c, ctx.h, ctx.w), 0)

        
    @staticmethod
    def get_flow_src():
        return InnerShiftSingleFunction.ctx.flow_srcs

    # Seems not quite suitable for encoder-decoder.
    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        grad_shifted_all = grad_output.clone().zero_()

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
            grad_shifted_all[idx] = grad_shifted_weighted.mul(ctx.triple_w)


        grad_input = grad_shifted_all + grad_output.masked_fill_(ctx.mask.view(1, 1, ctx.h, ctx.w).expand(ctx.bz, ctx.c, ctx.h, ctx.w), 0)

        return grad_input, None, None, None, None, None, None
