import numpy as np
from util.NonparametricShift import Modified_NonparametricShift, Batch_NonShift
import torch
import util.util as util
import time


class InnerShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, shift_sz, stride, triple_w, flag, show_flow):
        InnerShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.show_flow = show_flow

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real

        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_().to(input)

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### decoder feature
        latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all).zero_() # addition feature

        ctx.flag = ctx.flag.to(input).long()

        # None batch version
        bNonparm = Batch_NonShift()
        ctx.shift_offsets = []

        # batch version
        cosine, latter_windows, i_2, i_3, i_1 = bNonparm.cosine_similarity(former_all.clone(), latter_all.clone(), 1, stride, flag)

        _, indexes = torch.max(cosine, dim=2)

        mask_indexes = (flag==1).nonzero(as_tuple=False)[:, 1].view(ctx.bz, -1)

        non_mask_indexes = (flag==0).nonzero(as_tuple=False)[:, 1].view(ctx.bz, -1).gather(1, indexes)

        idx_b = torch.arange(ctx.bz).long().unsqueeze(1).expand(ctx.bz, mask_indexes.size(1))
        # set the elemnets of indexed by [mask_indexes, non_mask_indexes] to 1.
        # It is a batch version
        ctx.ind_lst[(idx_b, mask_indexes, non_mask_indexes)] = 1

        shift_masked_all = bNonparm._paste(latter_windows, ctx.ind_lst, i_2, i_3, i_1)


        # --- Non-batch version ----
        #for idx in range(ctx.bz):
        #    flag_cur = ctx.flag[idx]
        #    latter = latter_all.narrow(0, idx, 1) ### encoder feature
        #    former = former_all.narrow(0, idx, 1) ### decoder feature

        #    #GET COSINE, RESHAPED LATTER AND ITS INDEXES
        #    cosine, latter_windows, i_2, i_3, i_1 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag_cur)

        #   ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
        #    _, indexes = torch.max(cosine, dim=1)

        #    # SET  TRANSITION MATRIX
        #    mask_indexes = (flag_cur == 1).nonzero()
        #    non_mask_indexes = (flag_cur == 0).nonzero()[indexes]
        #    ctx.ind_lst[idx][mask_indexes, non_mask_indexes] = 1

        #    # GET FINAL SHIFT FEATURE
        #    shift_masked_all[idx] = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1)

        #    if ctx.show_flow:
        #        shift_offset = torch.stack([non_mask_indexes.squeeze() // ctx.w, non_mask_indexes.squeeze() % ctx.w], dim=-1)
        #        ctx.shift_offsets.append(shift_offset)

        if ctx.show_flow:
            assert 1==2, "I do not want maintance the functionality of `show flow`... ^_^"
            ctx.shift_offsets = torch.cat(ctx.shift_offsets, dim=0).float() # make it cudaFloatTensor
            # Assume mask is the same for each image in a batch.
            mask_nums = ctx.shift_offsets.size(0)//ctx.bz
            ctx.flow_srcs = torch.zeros(ctx.bz, 3, ctx.h, ctx.w).type_as(input)

            for idx in range(ctx.bz):
                shift_offset = ctx.shift_offsets.narrow(0, idx*mask_nums, mask_nums)
                # reconstruct the original shift_map.
                shift_offsets_map = torch.zeros(1, ctx.h, ctx.w, 2).type_as(input)
                shift_offsets_map[:, (flag_cur == 1).nonzero(as_tuple=False).squeeze() // ctx.w, (flag_cur == 1).nonzero(as_tuple=False).squeeze() % ctx.w, :] = \
                                                                                                shift_offset.unsqueeze(0)
                # It is indicating the pixels(non-masked) that will shift the the masked region.
                flow_src = util.highlight_flow(shift_offsets_map, flag_cur.unsqueeze(0))
                ctx.flow_srcs[idx] = flow_src

        return torch.cat((former_all, latter_all, shift_masked_all), 1)


    @staticmethod
    def get_flow_src():
        return InnerShiftTripleFunction.ctx.flow_srcs

    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        # C: content, pixels in masked region of the former part.
        # S: style, pixels in the non-masked region of the latter part.
        # N: the shifted feature, the new feature that will be used as the third part of features maps.
        # W_mat: ind_lst[idx], shift matrix.
        # Note: **only the masked region in N has values**.

        # The gradient of shift feature should be added back to the latter part(to be precise: S).
        # `ind_lst[idx][i,j] = 1` means that the i_th pixel will **be replaced** by j_th pixel in the forward.
        #  When applying `S mm W_mat`, then S will be transfer to N.
        #  (pixels in non-masked region of the latter part will be shift to the masked region in the third part.)
        #  However, we need to transfer back the gradient of the third part to S.
        #  This means the graident in S will **`be replaced`(to be precise, enhanced)** by N.
        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :].clone()
        grad_shifted_all = grad_output[:, c*2//3:c, :, :].clone()

        W_mat_t = ind_lst.permute(0, 2, 1).contiguous()
        grad = grad_shifted_all.view(ctx.bz, c//3, -1).permute(0, 2, 1)
        grad_shifted_weighted = torch.bmm(W_mat_t, grad)
        grad_shifted_weighted = grad_shifted_weighted.permute(0, 2, 1).contiguous().view(ctx.bz, c//3, ctx.h, ctx.w)
        grad_latter_all = torch.add(grad_latter_all, grad_shifted_weighted.mul(ctx.triple_w))

       # ----- 'Non_batch version here' --------------------
       # for idx in range(ctx.bz):
       #     # So we need to transpose `W_mat`
       #     W_mat_t = ind_lst[idx].t()

       #     grad = grad_shifted_all[idx].view(c//3, -1).t()

       #     grad_shifted_weighted = torch.mm(W_mat_t, grad)

       #     # Then transpose it back
       #     grad_shifted_weighted = grad_shifted_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
       #     grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_shifted_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None, None
