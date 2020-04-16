import torch.nn as nn
import torch
import util.util as util
from .InnerFaceShiftTripleFunction import InnerFaceShiftTripleFunction


class InnerFaceShiftTriple(nn.Module):
    def __init__(self, shift_sz=1, stride=1, mask_thred=1, triple_weight=1, layer_to_last=3, device='gpu'):
        super(InnerFaceShiftTriple, self).__init__()

        self.shift_sz = shift_sz
        self.stride = stride
        self.mask_thred = mask_thred
        self.triple_weight = triple_weight
        self.layer_to_last = layer_to_last
        self.device = device
        self.show_flow = False # default false. Do not change it to be true, it is computation-heavy.
        self.flow_srcs = None # Indicating the flow src(pixles in non-masked region that will shift into the masked region)


    def set_mask(self, mask_global):
        self.mask_all = util.cal_feat_mask(mask_global, self.layer_to_last)

    def _split_mask(self, cur_bsize):
        # get the visible indexes of gpus and assign correct mask to set of images
        cur_device = torch.cuda.current_device()
        self.cur_mask = self.mask_all[cur_device*cur_bsize:(cur_device+1)*cur_bsize, :, :, :]


    # If mask changes, then need to set cal_fix_flag true each iteration.
    def forward(self, input, flip_feat=None):
        self.bz, self.c, self.h, self.w = input.size()
        if self.device != 'cpu':
            self._split_mask(self.bz)
        else:
            self.cur_mask = self.mask_all
        self.mask = self.cur_mask
        self.mask_flip = torch.flip(self.mask, [3])

        self.flag = util.cal_flag_given_mask_thred(self.mask, self.shift_sz, self.stride, self.mask_thred)
        self.flag_flip = util.cal_flag_given_mask_thred(self.mask_flip, self.shift_sz, self.stride, self.mask_thred)

        final_out = InnerFaceShiftTripleFunction.apply(input, self.shift_sz, self.stride, self.triple_weight, self.flag, self.flag_flip, self.show_flow, flip_feat)
        if self.show_flow:
            self.flow_srcs = InnerFaceShiftTripleFunction.get_flow_src()

        innerFeat = input.clone().narrow(1, self.c//2, self.c//2)
        return final_out, innerFeat

    def get_flow(self):
        return self.flow_srcs

    def set_flow_true(self):
        self.show_flow = True

    def set_flow_false(self):
        self.show_flow = False

    def __repr__(self):
        return self.__class__.__name__+ '(' \
              + ' ,triple_weight ' + str(self.triple_weight) + ')'
