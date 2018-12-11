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

        #ctx.ind_lst = torch.LongTensor(ctx.bz, ctx.h * ctx.w, ctx.h * ctx.w)

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### UPCONV
        latter_all = input.narrow(1, c//2, c//2) ### UNET ADD

        assert mask.dim() == 2, "Mask dimension must be 2"
        ex_mask = mask.expand(1, c//2, mask.size(0), mask.size(1)) # 1*c*h*w
        inv_ex_mask = torch.add(torch.neg(ex_mask.float()), 1).byte()

        if torch.cuda.is_available:
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
            attention = F.softmax(cosine, dim=1)

            ## GET PATCHES CORRESPONDING
            patches_former = torch.matmul(attention, patches_latter)

            # CREATE HOLDER
            former_masked = torch.zeros(former.size()).cuda()

            # PASTE VALUES INTO HOLDER
            former_masked = Nonparm._paste(former_masked.squeeze(), 1, stride, flag, patches_former)

            former_masked = former_masked.detach() # DOESN'T WORK WITHOUT DETACHING THE LAYER

        return torch.cat((former, latter, former_masked), 1)
