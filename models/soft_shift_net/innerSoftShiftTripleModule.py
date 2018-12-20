from util.NonparametricShift import Modified_NonparametricShift
from torch.nn import functional as F
import torch.nn as nn
import torch
import util.util as util


class InnerSoftShiftTripleModule(nn.Module):
    def forward(self, input, mask, stride, triple_w, flag, show_flow):
        assert input.dim() == 4, "Input Dim has to be 4"
        self.triple_w = triple_w
        self.flag = flag
        self.show_flow = show_flow

        self.bz, c_real, self.h, self.w = input.size()
        c = c_real

        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available else torch.FloatTensor

        self.ind_lst = self.Tensor(self.bz, self.h * self.w, self.h * self.w).zero_()

        # former and latter are all tensors
        former_all = input.narrow(1, 0, c//2) ### decoder feature
        latter_all = input.narrow(1, c//2, c//2) ### encoder feature
        shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all) # addition feature

        assert mask.dim() == 2, "Mask dimension must be 2"

        if torch.cuda.is_available:
            self.flag = self.flag.cuda()

        # None batch version
        Nonparm = Modified_NonparametricShift()
        self.shift_offsets = []

        for idx in range(self.bz):
            latter = latter_all.narrow(0, idx, 1) ### encoder feature
            former = former_all.narrow(0, idx, 1) ### decoder feature

            #GET COSINE, RESHAPED LATTER AND ITS INDEXES
            cosine, latter_windows, former_windows, i_2, i_3, i_1, i_4 = Nonparm.cosine_similarity(former.clone().squeeze(), latter.clone().squeeze(), 1, stride, flag, with_former=True)

            ## GET INDEXES THAT MAXIMIZE COSINE SIMILARITY
            cosine_softmax = F.softmax(cosine, dim=1)

            mask_indexes = (self.flag == 1).nonzero()
            non_mask_indexes = (self.flag == 0).nonzero()
            self.ind_lst[idx][mask_indexes, non_mask_indexes.t()] = cosine_softmax

            # GET FINAL SHIFT FEATURE
            shift_masked_all[idx] = Nonparm._paste(latter_windows, self.ind_lst[idx], i_2, i_3, i_1, i_4)

            # For pixel soft shift, we just get the max value of similarity.
            if self.show_flow:
                _, indexes = torch.max(cosine, dim=1)
                non_mask_indexes = non_mask_indexes[indexes]
                shift_offset = torch.stack([non_mask_indexes.squeeze() // self.w, non_mask_indexes.squeeze() % self.w], dim=-1)
                self.shift_offsets.append(shift_offset)

        if self.show_flow:
            # Note: Here we assume that each mask is the same for the same batch image.
            self.shift_offsets = torch.cat(self.shift_offsets, dim=0).float() # make it cudaFloatTensor
            # Assume mask is the same for each image in a batch.
            mask_nums = self.shift_offsets.size(0)//self.bz
            self.flow_srcs = torch.zeros(self.bz, 3, self.h, self.w).type_as(input)

            for idx in range(self.bz):
                shift_offset = self.shift_offsets.narrow(0, idx*mask_nums, mask_nums)
                # reconstruct the original shift_map.
                shift_offsets_map = torch.zeros(1, self.h, self.w, 2).type_as(input)
                shift_offsets_map[:, (self.flag == 1).nonzero().squeeze() // self.w, (self.flag == 1).nonzero().squeeze() % self.w, :] = \
                                                                                                shift_offset.unsqueeze(0)
                # It is indicating the pixels(non-masked) that will shift the the masked region.
                flow_src = util.highlight_flow(shift_offsets_map, self.flag.unsqueeze(0))
                self.flow_srcs[idx] = flow_src

        return torch.cat((former_all, latter_all, shift_masked_all), 1)

    def get_flow_src(self):
        return self.flow_srcs
