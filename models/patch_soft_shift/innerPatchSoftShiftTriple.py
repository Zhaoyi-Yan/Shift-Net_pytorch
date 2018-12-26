import torch.nn as nn
import torch
import util.util as util
from .innerPatchSoftShiftTripleModule import InnerPatchSoftShiftTripleModule


# TODO: Make it compatible for show_flow.
#
class InnerPatchSoftShiftTriple(nn.Module):
    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, fuse=True):
        super(InnerPatchSoftShiftTriple, self).__init__()

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.show_flow = False # default false. Do not change it to be true, it is computation-heavy.
        self.flow_srcs = None # Indicating the flow src(pixles in non-masked region that will shift into the masked region)
        self.fuse = fuse
        self.softShift = InnerPatchSoftShiftTripleModule()

    def set_mask(self, mask_global, layer_to_last):
        mask = util.cal_feat_mask(mask_global, layer_to_last)
        self.mask = mask # 1*1*H*W (DO NOT Squeeze here!)
        return self.mask

    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):
        _, self.c, self.h, self.w = input.size()

        # Just pass self.mask in, instead of self.flag.
        # Try to making it faster by avoiding `cal_flag_given_mask_thread`.
        final_out = self.softShift(input, self.stride, self.triple_weight, self.mask, self.mask_thred, self.shift_sz, self.show_flow, self.fuse)
        if self.show_flow:
            self.flow_srcs = self.softShift.get_flow_src()
        return final_out

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
