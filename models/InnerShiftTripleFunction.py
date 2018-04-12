import torch
from util.NonparametricShift import NonparametricShift
from util.MaxCoord import MaxCoord
import util.util as util
import torch.nn as nn
import torch
from torch.autograd.function import once_differentiable


from torch.autograd import Variable
class InnerShiftTripleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, mask, swap_sz, stride, triple_w, flag, nonmask_point_idx, flatten_offsets, sp_x, sp_y):
        assert input.dim() == 4, "Input Dim has to be 4"
        ctx.triple_w = triple_w
        ctx.flag = flag
        ctx.flatten_offsets = flatten_offsets


        ctx.bz, c_real, ctx.h, ctx.w = input.size()
        c = c_real   # As mask is not as a channel passing
        ctx.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2)
        latter_all = input.narrow(1, c//2, c//2)
        
        assert mask.dim() == 2, "Mask dimension must be 2"
        ex_mask = mask.expand(1, c//2, mask.size(0), mask.size(1)) # 1*c*h*w
        inv_ex_mask = torch.add(torch.neg(ex_mask.float()), 1).byte()

        # bz is the batchsize of this GPU
        output_lst = ctx.Tensor(ctx.bz, int(c//2*3), ctx.h, ctx.w) # it is triple
        ind_lst = torch.LongTensor(ctx.bz, ctx.h*ctx.w)

        if torch.cuda.is_available:
            ind_lst = ind_lst.cuda()
            nonmask_point_idx = nonmask_point_idx.cuda()
            sp_x = sp_x.cuda()
            sp_y = sp_y.cuda()

            # If cuda is available, then need to convert ByteTensor to cuda.ByteTensor
            ex_mask = ex_mask.cuda()
            inv_ex_mask = inv_ex_mask.cuda()


        for idx in range(ctx.bz):
            latter = latter_all.narrow(0, idx, 1)
            former = former_all.narrow(0, idx, 1)
            # Mind the params order is inconsistent with Torch version.
            Nonparm = NonparametricShift()
            _, conv_enc, conv_new_dec, _, = Nonparm.buildAutoencoder(latter.squeeze(), False, False, nonmask_point_idx, swap_sz, stride)
            
            former_var = Variable(former, requires_grad = False)
            tmp1 = conv_enc(former_var)
            latter_non_mask = latter.clone()
            
            # print('cuda is: ', torch.cuda.current_device(), ' and bz is: ', ctx.bz)
            latter_non_mask.masked_fill_(ex_mask, 0) # only save non_mask region

            maxcoor = MaxCoord()
            # mention: kbar and ind are all 0-index
            kbar, ind = maxcoor.update_output(tmp1.data, sp_x, sp_y)

            # calculate the real kbar and real self.ind
            real_patches = kbar.size(1) + torch.sum(ctx.flag)
            _, _, kbar_h, kbar_w = kbar.size()
            kbar = ctx.Tensor(1, real_patches, kbar_h, kbar_w).zero_()

            offset = 0
            for i in range(kbar_h):
                for j in range(kbar_w):
                    indx = i*kbar_w + j
                    non_r_ch = ind[indx]
                    offset = ctx.flatten_offsets[non_r_ch]
                    correct_ch = int(non_r_ch + offset)
                    kbar[:,correct_ch,i,j] = 1
                    ind[indx] = correct_ch
            
            kbar_var = Variable(kbar, requires_grad =False)
            result_tmp_var = conv_new_dec(kbar_var)
            result_tmp = result_tmp_var.data
            result_tmp_mask = result_tmp.clone()

            result_tmp_mask.masked_fill_(inv_ex_mask, 0)  # mask part

            # Swapped_latter should be only contain the content of the mask region. More complementarity!
            swapped_latter = result_tmp_mask

            # construct final self.output
            output_lst[idx] = torch.cat((former, latter, swapped_latter), 1)
            ind_lst[idx] = ind

        # shortcut
        output = output_lst


        # For now:
        # cuda.ByteTensor: ex_mask, inv_ex_mask
        # LongTensor: ind, centers_lt, centers_rb  (It is non-cuda tensors, seems no need to transfer to cuda)

        ctx.save_for_backward(input)

        ctx.ind_lst = ind_lst

        return output


    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        ind_lst = ctx.ind_lst
        flag = ctx.flag

        c = grad_output.size(1)

        # # the former and the latter are keep original. Only the thrid part is shifted.
        grad_former_all = grad_output[:, 0:c//3, :, :]
        grad_latter_all = grad_output[:, c//3: c*2//3, :, :]
        grad_swapped_all = grad_output[:, c*2//3:c, :, :]

        spatial_size = ctx.h * ctx.w

        for idx in range(ctx.bz):
            W_mat = ctx.Tensor(spatial_size, spatial_size).zero_()

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


