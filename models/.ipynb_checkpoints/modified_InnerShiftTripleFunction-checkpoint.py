import torch
from util.NonparametricShift import NonparametricShift, Modified_NonparametricShift
from util.MaxCoord import MaxCoord
import util.util as util
import torch.nn as nn
from torch.nn import functional as F
import torch


class ModifiedInnerShiftTripleFunction(nn.Module):

    def forward(self, input, mask, shift_sz, stride, triple_w, flag):
        #print('[INFO] GET INTO FORWARD')

        assert input.dim() == 4, "Input Dim has to be 4"
        self.triple_w = triple_w
        self.flag = flag

        self.bz, c_real, self.h, self.w = input.size()
        c = c_real
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### UPCONV
        latter_all = input.narrow(1, c//2, c//2) ### UNET ADD

        assert mask.dim() == 2, "Mask dimension must be 2"
        ex_mask = mask.expand(1, c//2, mask.size(0), mask.size(1)) # 1*c*h*w
        inv_ex_mask = torch.add(torch.neg(ex_mask.float()), 1).byte()

        # bz is the batchsize of this GPU
        #output_lst = ctx.Tensor(ctx.bz, int(c//2*3), ctx.h, ctx.w) # it is triple
        #ind_lst = torch.LongTensor(ctx.bz, ctx.h*ctx.w)

        if torch.cuda.is_available:
            #ind_lst = ind_lst.cuda()
            #nonmask_point_idx = nonmask_point_idx.cuda()
            #sp_x = sp_x.cuda()
            #sp_y = sp_y.cuda()
            flag = flag.cuda()

            # If cuda is available, then need to convert ByteTensor to cuda.ByteTensor
            #ex_mask = ex_mask.cuda()
            inv_ex_mask = inv_ex_mask.cuda()

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
            
            patches_latter_norm = Nonparm._norm(patches_latter.clone())

            ## CALCULATE ABSOLUTE COSINE SIMILARITY
            multi = torch.matmul(patches_former, patches_latter_norm.permute(1,0))

            ## CALCULATE MAX OVER 
            _, multi = torch.max(multi, dim=1)#F.softmax(multi, dim=1) ###
            
            #print(multi)

            ## CALCULATE softmax_weight * X(NON-MASK)
            #print(patches_latter.shape)
            patches_former = patches_latter[multi]#torch.matmul(multi, patches_latter)

            ## ADD PREVIOUS MASK PATCH + NEW ATTENTION COMBINAISON OF OTHERS FEATURES

            former_masked = former.clone().masked_fill_(inv_ex_mask, 0)
            
            former_masked = Nonparm._paste(former_masked.squeeze(), 1, stride, flag, patches_former)
                                         
            former_masked = former_masked.detach().clone()

        return torch.cat((former, latter, former_masked), 1)

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
    @staticmethod
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst
        print(backward.shape)
        flag = ctx.flag

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :].clone()
        grad_swapped_all = grad_output[:, c*2//3:c, :, :].clone()

        spatial_size = ctx.h * ctx.w

        W_mat_all = ctx.Tensor(ctx.bz, spatial_size, spatial_size).zero_()
        for idx in range(ctx.bz):
            W_mat = W_mat_all.select(0,idx).clone()
            for cnt in range(spatial_size):
                indS = ind_lst[idx][cnt]   # indS is index of the outer-mask

                # It means this pixel is in the mask, and this line(index: cnt_th)
                # should be one-hot vector, with the `indS_th` be 1.
                if flag[cnt] == 1:
                    W_mat[cnt, indS] = 1

            W_mat_t = W_mat.t()

            # view(c/3,-1):t() makes each line be a gradient of certain position which is c/3 channels.
            grad_swapped_weighted = torch.mm(W_mat_t, grad_swapped_all[idx].view(c//3, -1).t())

            # Then transpose it back
            grad_swapped_weighted = grad_swapped_weighted.t().contiguous().view(1, c//3, ctx.h, ctx.w)
            grad_latter_all[idx] = torch.add(grad_latter_all[idx], grad_swapped_weighted.mul(ctx.triple_w))


        # note the input channel and the output channel are all c, as no mask input for now.
        grad_input = torch.cat([grad_former_all, grad_latter_all], 1)

        return grad_input, None, None, None, None, None, None, None, None, None, None
