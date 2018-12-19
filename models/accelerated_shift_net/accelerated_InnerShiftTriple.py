import torch.nn as nn
import torch
import util.util as util
from .accelerated_InnerShiftTripleFunction import AcceleratedInnerShiftTripleFunction

class AcceleratedInnerShiftTriple(nn.Module):
    def __init__(self, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(AcceleratedInnerShiftTriple, self).__init__()
        self.fixed_mask = fixed_mask

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.cal_fixed_flag = True # whether we need to calculate the temp varaiables this time.
        self.show_flow = False # default false. Do not change it to be true, it is computation-heavy.
        self.flow_srcs = None # Indicating the flow src(pixles in non-masked region that will shift into the masked region)


    def set_mask(self, mask_global, layer_to_last):
        mask = util.cal_feat_mask(mask_global, layer_to_last)
        self.mask = mask.squeeze()
        return self.mask

    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input):
        #print(input.shape)
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(1, self.c//2, self.c//2).narrow(0,0,1).detach()
            self.flag = util.cal_flag_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, \
                                                                                                   self.stride, self.mask_thred)
            self.cal_fixed_flag = False

        tmp = AcceleratedInnerShiftTripleFunction.apply(input, self.mask, self.shift_sz, self.stride, self.triple_weight, self.flag, self.show_flow)
        if self.show_flow:
            self.flow_srcs = AcceleratedInnerShiftTripleFunction.get_flow_src()
        return tmp

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
