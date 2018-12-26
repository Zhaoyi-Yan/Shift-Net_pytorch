from util.NonparametricShift import Modified_NonparametricShift
from torch.nn import functional as F
import torch.nn as nn
import torch
import util.util as util


class InnerPatchSoftShiftTripleModule(nn.Module):
    def forward(self, input, stride, triple_w, mask, mask_thred, shift_sz, show_flow, fuse=True):
        assert input.dim() == 4, "Input Dim has to be 4"
        assert mask.dim() == 4, "Mask Dim has to be 4"
        self.triple_w = triple_w
        self.mask = mask
        self.mask_thred = mask_thred
        self.show_flow = show_flow

        self.bz, self.c, self.h, self.w = input.size()

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        self.ind_lst = self.Tensor(self.bz, self.h * self.w, self.h * self.w).zero_()

        # former and latter are all tensors
        former_all = input.narrow(1, 0, self.c//2) ### decoder feature
        latter_all = input.narrow(1, self.c//2, self.c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all) # addition feature

        # TODO: let `cal_feat_mask` return a floatTensor, not a byteTensor.
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()

        # extract patches from latter.
        latter_all_pad = F.pad(latter_all, [shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2], mode='constant')
        latter_all_windows = latter_all_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
        latter_all_windows = latter_all_windows.contiguous().view(self.bz, -1, self.c//2, shift_sz, shift_sz)

        # Extract patches from mask
        # Mention: mask here must be 1*1*H*W
        m_pad = F.pad(self.mask, (shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2), 'constant', 0)
        m = m_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
        m = m.contiguous().view(1, -1, shift_sz, shift_sz)

        # This two line of code can replace `cal_flag_given_mask_thred`
        m = torch.mean(torch.mean(m, dim=2, keepdim=True), dim=3, keepdim=True)
        # mm: the masked reigon is filled with 0, nonmasked region is filled with 1.
        mm = m.le(self.mask_thred/(1.*shift_sz**2)).float() # 1*(32*32)*1*1

        fuse_weight = torch.eye(shift_sz).view(1, 1, shift_sz, shift_sz).type_as(input)

        for idx in range(self.bz):
            # latter_win = latter_all_windows.narrow(0, idx, 1)[0]
            latter_win = latter_all_windows.narrow(0, idx, 1)[0].detach()
            former = former_all.narrow(0, idx, 1)

            # normalize latter for each patch.
            latter_den = torch.sqrt(torch.einsum("bcij,bcij->b", [latter_win, latter_win]))
            latter_win_normed = latter_win/latter_den.view(-1, 1, 1, 1)
            
            y_i = F.conv2d(former, latter_win_normed, stride=1, padding=shift_sz//2)

            # conv implementation for fuse scores to encourage large patches
            if fuse:
                y_i = y_i.view(1, 1, self.h*self.w, self.h*self.w) # make all of depth of spatial resolution.
                y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)

                y_i = y_i.contiguous().view(1, self.h, self.w, self.h, self.w)
                y_i = y_i.permute(0, 2, 1, 4, 3)
                y_i = y_i.contiguous().view(1, 1, self.h*self.w, self.h*self.w)

                y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)
                y_i = y_i.contiguous().view(1, self.w, self.h, self.w, self.h)
                y_i = y_i.permute(0, 2, 1, 4, 3)

            y_i = y_i.contiguous().view(1, self.h*self.w, self.h, self.w) # 1*(32*32)*32*32

            # firstly, wash away the masked reigon.
            # multiply `mm` means (:, index_masked, :, :) will be 0.
            y_i = y_i * mm

            # Then apply softmax to the nonmasked region.
            cosine = F.softmax(y_i, dim=1)

            # Finally, dummy parameters of masked reigon are filtered out.
            cosine = cosine * mm

            # paste
            shift_i = F.conv_transpose2d(cosine, latter_win, stride=1, padding=1)
            shift_masked_all[idx] = shift_i

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    def get_flow_src(self):
        return self.flow_srcs
