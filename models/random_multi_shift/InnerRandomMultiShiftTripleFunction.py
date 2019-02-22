import numpy as np
from util.NonparametricShift import Modified_NonparametricShift
import torch
import torch.nn.functional as F
import util.util as util
import time


class InnerRandomMultiShiftTripleFunction(torch.autograd.Function):
    ctx = None

    @staticmethod
    def forward(ctx, input, mask, shift_sz, stride, triple_w, flag, show_flow, previous_neighbor):
        #print('[INFO] GET INTO FORWARD')
        InnerRandomMultiShiftTripleFunction.ctx = ctx
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.show_flow = show_flow
        ctx.previous_neighbor = previous_neighbor

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
            ctx.flag = ctx.flag.cuda()

        # None batch version
        Nonparm = Modified_NonparametricShift()
        ctx.shift_offsets = []

        # We only assume that two shift layers
        # Prepare neighbors for the next layer.
        neighbor_size = 3
        # For multi-shift, we also need to get index_neighbor_ref_unfold.
        # It shuold be recalculated whenever mask changes.
        if ctx.previous_neighbor is None:
            # constructiing index map
            neighbor_size = 3
            index_neighbor_ref = torch.arange(ctx.h*ctx.w).type_as(input).resize_(1, 1, ctx.h, ctx.w)
            # adding boarder with value of '-1', indicating invalid regions.
            index_neighbor_ref_tmp = F.pad(index_neighbor_ref, (neighbor_size//2, neighbor_size//2, neighbor_size//2, neighbor_size//2), 'constant', -1)
            # Also, we need to assigin '-1' to masked region.
            index_neighbor_ref_tmp.view(-1)[(ctx.flag == 1).nonzero()] = -1
            index_neighbor_ref_tmp = index_neighbor_ref_tmp.view(1, 1, ctx.h + 2*(neighbor_size//2), ctx.w + 2*(neighbor_size//2))
            # It is static for each image in a batch.
            # Eg.(16*16)*(3*3)
            ctx.index_neighbor_ref_unfold = index_neighbor_ref_tmp.unfold(2, neighbor_size, 1).unfold(3, neighbor_size, 1).contiguous().view(ctx.h*ctx.w, -1)
        
        # For each position, we need to assign proper nerighbors to it.
        ctx.current_neighbor = torch.zeros(ctx.bz, ctx.h*ctx.w, neighbor_size**2).type_as(input)
        
        for idx in range(ctx.bz):
            latter = latter_all.narrow(0, idx, 1) ### encoder feature
            former = former_all.narrow(0, idx, 1) ### decoder feature
            mask_indexes = (ctx.flag == 1).nonzero()

            # For the first shift layer.
            if ctx.previous_neighbor is None:
                #GET COSINE, RESHAPED LATTER AND ITS INDEXES
                cosine, latter_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag)

                ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
                _, indexes = torch.max(cosine, dim=1)

                # SET  TRANSITION MATRIX
                non_mask_indexes = (ctx.flag == 0).nonzero()[indexes]
                ctx.ind_lst[idx][mask_indexes, non_mask_indexes] = 1

                # GET FINAL SHIFT FEATURE
                shift_masked_all[idx] = Nonparm._paste(latter_windows, ctx.ind_lst[idx], i_2, i_3, i_1, i_4)

                # Construct neighbors for the next shift layer.
                torch.set_printoptions(threshold=10e7)
                ctx.current_neighbor[idx][mask_indexes] = ctx.index_neighbor_ref_unfold[non_mask_indexes]
            # For the second shift layer.
            else:
                # Direct get shift matrix from previous_neighbor
                # TODO: How to map mask_index in this layer to correspoinding point in the previous layer?
                mask_indexes_p = mask_indexes // ctx.w // 2
                mask_indexes_q = mask_indexes % ctx.w // 2
                print('mask_p,q', mask_indexes_p, mask_indexes_q)
                print('cc')
                # print(mask_indexes_p * (ctx.w//2) + mask_indexes_q)
                print(ctx.previous_neighbor.shape)
                print('Now mask_indexes', mask_indexes.size()) # 225*1
                print()
                print(((ctx.previous_neighbor[idx].squeeze().sum(dim=1))==0).nonzero().size())
                print(ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+0)].squeeze().size())
                # Here, I just use for-loop to do it, maybe need optimization.
                # TODO: for some lines, elements are all '-1'. How to deal with it?
                for i in range(mask_indexes.size(0)):
                    # get neighbor
                    i_p = i // ctx.w // 2
                    i_q = i % ctx.w // 2
                    tmp = torch.randint(0, ctx.previous_neighbor.size(-1), (1,)).long()
                    print(ctx.previous_neighbor[idx, (i_p * (ctx.w//2) + i_q+0), tmp])
                    while((ctx.previous_neighbor[idx, (i_p * (ctx.w//2) + i_q+0), tmp]+1) == 0):
                        print('i:',i, ' lala')
                        tmp = torch.randint(0, ctx.previous_neighbor.size(-1), (1,)).long()

                    tmp_neighbor = ctx.previous_neighbor[idx, (i_p * (ctx.w//2) + i_q+0), tmp].long()
                    # mapping neighbor to the current layer, rewrite this line.
                    tmp_neighbor *= 2
                    # clamp tmp_neighbor to the reasonable range.
                    ctx.ind_lst[idx, i, tmp_neighbor] = 1


                    

                print(((ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+0)].squeeze().sum(dim=1)) == 0).nonzero().size())
                print(((ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+1)].squeeze().sum(dim=1)) == 0).nonzero().size())
                print(((ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+2)].squeeze().sum(dim=1)) == 0).nonzero().size())
                print(((ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+3)].squeeze().sum(dim=1)) == 0).nonzero().size())

                # print(ctx.previous_neighbor[idx, (mask_indexes_p * (ctx.w//2) + mask_indexes_q+0)])
                assert 1==2

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

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    @staticmethod
    def get_current_neighbor():
        return InnerRandomMultiShiftTripleFunction.ctx.current_neighbor

    @staticmethod
    def get_flow_src():
        return InnerRandomMultiShiftTripleFunction.ctx.flow_srcs

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

        return grad_input, None, None, None, None, None, None, None
