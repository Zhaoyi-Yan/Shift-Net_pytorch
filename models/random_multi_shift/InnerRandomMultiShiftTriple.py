import torch.nn as nn
import torch
import util.util as util
from .InnerRandomMultiShiftTripleFunction import InnerRandomMultiShiftTripleFunction

class InnerRandomMultiShiftTriple(nn.Module):
    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, layer_to_last=3, neighbor_dict={}):
        super(InnerRandomMultiShiftTriple, self).__init__()

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.layer_to_last = layer_to_last
        self.neighbor_dict = neighbor_dict
        self.show_flow = False # default false. Do not change it to be true, it is computation-heavy.
        self.flow_srcs = None # Indicating the flow src(pixles in non-masked region that will shift into the masked region)

        self.previous_neighbor = None
        self.current_neighbor = None
        # Check neighbor information of precvious layer.
        if str(self.layer_to_last + 1) in self.neighbor_dict:
            print('In dict', str(self.layer_to_last))
            self.previous_neighbor = self.neighbor_dict[str(self.layer_to_last+1)]
            


    def set_mask(self, mask_global):
        mask = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask = mask.squeeze()
        return self.mask

    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        self.flag = util.cal_flag_given_mask_thred(self.mask, self.shift_sz, self.stride, self.mask_thred)

        final_out = InnerRandomMultiShiftTripleFunction.apply(input, self.mask, self.shift_sz, self.stride, self.triple_weight, self.flag, self.show_flow, self.previous_neighbor)
        # get current_neighbor through static method
        self.current_neighbor = InnerRandomMultiShiftTripleFunction.get_current_neighbor()
        # then refresh the current neighbor information.
        self.neighbor_dict[str(self.layer_to_last)] = self.current_neighbor

        if self.show_flow:
            self.flow_srcs = InnerRandomMultiShiftTripleFunction.get_flow_src()
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
