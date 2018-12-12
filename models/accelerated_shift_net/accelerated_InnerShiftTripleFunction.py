from util.NonparametricShift import Modified_NonparametricShift
import torch


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

        ctx.ind_lst = torch.LongTensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w).zero_() # we need zero it out.

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### UPCONV
        latter_all = input.narrow(1, c//2, c//2) ### UNET ADD

        assert mask.dim() == 2, "Mask dimension must be 2"
        ex_mask = mask.expand(1, c//2, mask.size(0), mask.size(1)) # 1*c*h*w
        inv_ex_mask = torch.add(torch.neg(ex_mask.float()), 1).byte()

        if torch.cuda.is_available:
            ctx.ind_lst = ctx.ind_lst.cuda()
            flag = flag.cuda()

            inv_ex_mask = inv_ex_mask.cuda()

        # None batch version
        for idx in range(ctx.bz):
            latter = latter_all.narrow(0, idx, 1) ### UNET ADD
            former = former_all.narrow(0, idx, 1) ### UPCONV

            Nonparm = Modified_NonparametricShift()

            ## EXTRACT MASK PATCHES FROM FORMER
            patches_former = Nonparm._extract_patches_from_flag(former.clone().squeeze(), 1, stride, flag, 1)

            ## EXTRACT NON-MASK PATCHES FROM FORMER
            patches_latter = Nonparm._extract_patches_from_flag(latter.clone().squeeze(), 1, stride, flag, 0)

            patches_latter_norm = Nonparm._norm(patches_latter.clone())

            ## CALCULATE ABSOLUTE COSINE SIMILARITY
            cosine = torch.mm(patches_former, patches_latter_norm.permute(1,0))

            ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
            _, indexes = torch.max(cosine, dim=1) # indexes: the same size with patches_former(masked part)'s patches.

            ## GET PATCHES CORRESPONDING
            patches_former = patches_latter[indexes] # extract the corresponding (most correlative) patches from patches_latter.


            # CREATE HOLDER
            former_masked = torch.zeros(former.size()).type_as(input)

            # PASTE VALUES INTO HOLDER
            former_masked = Nonparm._paste(former_masked.squeeze(), 1, stride, flag, patches_former)


            # CREATE MAPPING MATRIX
            mask_indexes = (flag == 1).nonzero()
            non_mask_indexes = (flag == 0).nonzero()[indexes]
            ctx.ind_lst[idx][tuple((mask_indexes, non_mask_indexes))] = 1 # advanced indexing

            # It has been checked.
            # For example, when setting 'center=mask', 252 lines will be one-hot in 1024*1024 matrix.
            # Try verify the correctness manually...

            # torch.set_printoptions(threshold=1024*1024)
            # print(flag)
            # print('flag1\n', flag_1)
            # print('flag0\n', flag_0)
            # print('indexes', indexes)
            # print(ctx.ind_lst[idx].sum())

            # print(ctx.ind_lst[idx])


        return torch.cat((former, latter, former_masked), 1)



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
            W_mat_t = ind_lst[idx].t().type_as(grad_output)

            grad = grad_shifted_all[idx].view(c//3, -1).t()

            grad_shifted_weighted = torch.mm(W_mat_t, grad)

            # Then transpose it back
            grad_shifted_weighted = grad_shifted_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
            grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_shifted_weighted.mul(ctx.triple_w))

        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None
