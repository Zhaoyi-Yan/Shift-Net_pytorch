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

        self.mask = self.mask.to(input)

        # extract patches from latter.
        latter_all_pad = F.pad(latter_all, [shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2], 'constant', 0)
        latter_all_windows = latter_all_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
        latter_all_windows = latter_all_windows.contiguous().view(self.bz, -1, self.c//2, shift_sz, shift_sz)

        # Extract patches from mask
        # Mention: mask here must be 1*1*H*W
        m_pad = F.pad(self.mask, (shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2), 'constant', 0)
        m = m_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
        m = m.contiguous().view(self.bz, 1, -1, shift_sz, shift_sz)

        # It implements the similar functionality as `cal_flag_given_mask_thred`.
        # However, it differs what `mm` means.
        # Here mm: the masked reigon is filled with 0, nonmasked region is filled with 1.
        # While mm in `cal_flag_given_mask_thred`, it is opposite.
        m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
        mm = m.le(self.mask_thred/(1.*shift_sz**2)).float() # bz*1*(32*32)*1*1

        fuse_weight = torch.eye(shift_sz).view(1, 1, shift_sz, shift_sz).type_as(input)

        self.shift_offsets = []
        for idx in range(self.bz):
            mm_cur = mm[idx]
            # latter_win = latter_all_windows.narrow(0, idx, 1)[0]
            latter_win = latter_all_windows.narrow(0, idx, 1)[0]
            former = former_all.narrow(0, idx, 1)

            # normalize latter for each patch.
            latter_den = torch.sqrt(torch.einsum("bcij,bcij->b", [latter_win, latter_win]))
            latter_den = torch.max(latter_den, self.Tensor([1e-4]))

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
            y_i = y_i * mm_cur

            # Then apply softmax to the nonmasked region.
            cosine = F.softmax(y_i*10, dim=1)

            # Finally, dummy parameters of masked reigon are filtered out.
            cosine = cosine * mm_cur

            # paste
            shift_i = F.conv_transpose2d(cosine, latter_win, stride=1, padding=shift_sz//2)/9.
            shift_masked_all[idx] = shift_i

            # Addition: show shift map
            # TODO: fix me.
            # cosine here is a full size of 32*32, not only the masked region in `shift_net`,
            # which results in non-direct reusing the code.
        #     torch.set_printoptions(threshold=2015)
        #     if self.show_flow:
        #         _, indexes = torch.max(cosine, dim=1)
        #         # calculate self.flag from self.m
        #         self.flag = (1 - mm).view(-1)
        #         torch.set_printoptions(threshold=1025)
        #         print(self.flag)
        #         non_mask_indexes = (self.flag == 0.).nonzero()
        #         non_mask_indexes = non_mask_indexes[indexes]
        #         print('ll')
        #         print(non_mask_indexes.size())
        #         print(non_mask_indexes)
        #         # Here non_mask_index is too large, should be 192.
        #         shift_offset = torch.stack([non_mask_indexes.squeeze() // self.w, non_mask_indexes.squeeze() % self.w], dim=-1)
        #         print(shift_offset.size())
        #         self.shift_offsets.append(shift_offset)

        # print('cc')
        # if self.show_flow:
        #     # Note: Here we assume that each mask is the same for the same batch image.
        #     self.shift_offsets = torch.cat(self.shift_offsets, dim=0).float() # make it cudaFloatTensor
        #     # Assume mask is the same for each image in a batch.
        #     mask_nums = self.shift_offsets.size(0)//self.bz
        #     self.flow_srcs = torch.zeros(self.bz, 3, self.h, self.w).type_as(input)

        #     for idx in range(self.bz):
        #         shift_offset = self.shift_offsets.narrow(0, idx*mask_nums, mask_nums)
        #         # reconstruct the original shift_map.
        #         shift_offsets_map = torch.zeros(1, self.h, self.w, 2).type_as(input)
        #         print(shift_offsets_map.size())
        #         print(shift_offset.unsqueeze(0).size())

        #         print(shift_offsets_map[:, (self.flag == 1).nonzero().squeeze() // self.w, (self.flag == 1).nonzero().squeeze() % self.w, :].size())
        #         shift_offsets_map[:, (self.flag == 1).nonzero().squeeze() // self.w, (self.flag == 1).nonzero().squeeze() % self.w, :] = \
        #                                                                                         shift_offset.unsqueeze(0)
        #         # It is indicating the pixels(non-masked) that will shift the the masked region.
        #         flow_src = util.highlight_flow(shift_offsets_map, self.flag.unsqueeze(0))
        #         self.flow_srcs[idx] = flow_src           

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    def get_flow_src(self):
        return self.flow_srcs
