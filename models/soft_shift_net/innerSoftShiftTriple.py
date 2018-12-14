import torch.nn as nn
import torch
import util.util as util
from .innerSoftShiftTripleModule import InnerSoftShiftTripleModule


class InnerSoftShiftTriple(nn.Module):
    def __init__(self, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(InnerSoftShiftTriple, self).__init__()
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True # whether we need to calculate the temp varaiables this time.

        # these two variables are for accerlating MaxCoord, it is constant tensors,
        # related with the spatialsize, unrelated with mask.
        self.sp_x = None
        self.sp_y = None
        self.softShift = InnerSoftShiftTripleModule()

    def set_mask(self, mask_global, layer_to_last):
        mask = util.cal_feat_mask(mask_global, layer_to_last)
        self.mask = mask.squeeze()
        return self.mask

    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(1, self.c//2, self.c//2).narrow(0,0,1).detach()
            self.flag = util.cal_flag_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, \
                                                                                                   self.stride, self.mask_thred)
            self.cal_fixed_flag = False

        return self.softShift(input, self.mask, self.stride, self.triple_weight, self.flag)

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
