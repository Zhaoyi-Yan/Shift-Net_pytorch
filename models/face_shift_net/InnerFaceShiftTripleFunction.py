import numpy as np
from util.NonparametricShift import Modified_NonparametricShift
import torch
import util.util as util
import time

# This script offers a version of shift from multi-references.
class InnerFaceShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, shift_sz, stride, triple_w, flag, flag_flip, show_flow, flip_feat=None):
        InnerFaceShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flag_flip = flag_flip
        ctx.show_flow = show_flow

        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real

        ctx.ind_lst = torch.Tensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_().to(input)
        ctx.ind_lst_flip = ctx.ind_lst.clone()

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### decoder feature
        latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all).zero_() # addition feature

        if not flip_feat is None:
            assert flip_feat.size() == former_all.size(), "flip_feat size should be equal to former size"

            ctx.flag = ctx.flag.to(input).long()
            ctx.flag_flip = ctx.flag_flip.to(input).long()

            # None batch version
            Nonparm = Modified_NonparametricShift()
            ctx.shift_offsets = []

            for idx in range(ctx.bz):
                flag_cur = ctx.flag[idx]
                flag_cur_flip = ctx.flag_flip[idx]
                latter = latter_all.narrow(0, idx, 1) ### encoder feature
                former = former_all.narrow(0, idx, 1) ### decoder feature

                #GET COSINE, RESHAPED LATTER AND ITS INDEXES
                cosine, latter_windows, i_2, i_3, i_1 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag_cur)
                cosine_flip, latter_windows_flip, _, _, _ = Nonparm.cosine_similarity(former.clone().squeeze(), flip_feat.clone().squeeze(), 1, stride, flag_cur_flip)

                # compare which is the bigger one.
                cosine_con = torch.cat([cosine, cosine_flip], dim=1)
                _, indexes_con = torch.max(cosine_con, dim=1)
                # then ori_larger is (non_mask_count*1),
                # 1:indicating the original feat is better for shift.
                # 0:indicating the flippled feat is a better one.
                ori_larger = (indexes_con < cosine.size(1)).long().view(-1,1)


                ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
                _, indexes = torch.max(cosine, dim=1)
                _, indexes_flip = torch.max(cosine_flip, dim=1)

                # SET  TRANSITION MATRIX
                mask_indexes = (flag_cur == 1).nonzero()
                non_mask_indexes = (flag_cur == 0).nonzero()[indexes]
                # then remove some indexes where we should select flip feat according to ori_larger
                mask_indexes_select_index = (mask_indexes.squeeze() * ori_larger.squeeze()).nonzero()
                mask_indexes_select = mask_indexes[mask_indexes_select_index].squeeze()
                ctx.ind_lst[idx][mask_indexes_select, non_mask_indexes] = 1



                non_mask_indexes_flip = (flag_cur_flip == 0).nonzero()[indexes_flip]
                # then remove some indexes where we should select ori feat according to 1-ori_larger
                mask_indexes_flip_select_index = (mask_indexes.squeeze() * (1 - ori_larger.squeeze())).nonzero()
                mask_indexes_flip_select = mask_indexes[mask_indexes_flip_select_index].squeeze()
                ctx.ind_lst_flip[idx][mask_indexes_flip_select, non_mask_indexes_flip] = 1


                # GET FINAL SHIFT FEATURE
                ori_tmp = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1)
                ori_tmp_flip = Nonparm._paste(latter_windows_flip, ctx.ind_lst_flip[idx], i_2, i_3, i_1)

                # combine the two features by directly adding, it is ok.
                shift_masked_all[idx] = ori_tmp + ori_tmp_flip

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
                shift_offsets_map[:, (flag_cur == 1).nonzero().squeeze() // ctx.w, (flag_cur == 1).nonzero().squeeze() % ctx.w, :] = \
                                                        shift_offset.unsqueeze(0)
                # It is indicating the pixels(non-masked) that will shift the the masked region.
                flow_src = util.highlight_flow(shift_offsets_map, flag_cur.unsqueeze(0))
                ctx.flow_srcs[idx] = flow_src

        return torch.cat((former_all, latter_all, shift_masked_all), 1)


    @staticmethod
    def get_flow_src():
        return InnerFaceShiftTripleFunction.ctx.flow_srcs


    # How it works, the extra grad from feat_flip will be enchaned the grad of the second part of the layer (when input I).
    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst
        ind_lst_flip = ctx.ind_lst_flip

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
            W_mat_t_flip = ind_lst_flip[idx].t()

            grad = grad_shifted_all[idx].view(c//3, -1).t()

            # only the grad of points that locations(non_mask) contribute to is kept
            grad_shifted_weighted = torch.mm(W_mat_t, grad)
            # only the grad of points that locations(non_mask_flip) contribute to is kept
            grad_shifted_weighted_flip = torch.mm(W_mat_t_flip, grad)

            # Then transpose it back
            grad_shifted_weighted = grad_shifted_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
            grad_shifted_weighted_flip = grad_shifted_weighted_flip.t().contiguous().view(1, c//3, ctx.h, ctx.w)

            grad_shifted_weighted_all = grad_shifted_weighted + grad_shifted_weighted_flip

            grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_shifted_weighted_all.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None, None, None
