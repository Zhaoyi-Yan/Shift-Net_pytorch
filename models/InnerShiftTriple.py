import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util
from .InnerShiftTripleFunction import InnerShiftTripleFunction

class InnerShiftTriple(nn.Module):
    def __init__(self, threshold, fixed_mask, shift_sz=1, stride=1, mask_thred=1, triple_weight=1):
        super(InnerShiftTriple, self).__init__()
        self.threshold = threshold
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

    def set_mask(self, mask_global, layer_to_last, threshold):
        mask = util.cal_feat_mask(mask_global, layer_to_last, threshold)
        self.mask = mask.squeeze()
        return self.mask
    
    # If mask changes, then need to cal_fix_flag true each time.
    def forward(self, input):
        _, self.c, self.h, self.w = input.size()
        if self.fixed_mask and self.cal_fixed_flag == False:
            assert torch.is_tensor(self.flag), 'flag must have been figured out and has to be a tensor!'
        else:
            latter = input.narrow(1, self.c//2, self.c//2).narrow(0,0,1).data

            self.flag, self.nonmask_point_idx, self.flatten_offsets = util.cal_mask_given_mask_thred(latter.squeeze(), self.mask, self.shift_sz, \
                                                                                                        self.stride, self.mask_thred)
            self.cal_fixed_flag = False
        
        if not (torch.is_tensor(self.sp_x) or torch.is_tensor(self.sp_y)):
            print('Pre-calculate constant assistant \'sp_x\' and \'sp_y\' for the layer, which channel is:', self.c, ', h is: ', self.h, ', w is ', self.w)
            self.sp_x, self.sp_y = util.cal_sps_for_Advanced_Indexing(self.h, self.w)


        return InnerShiftTripleFunction.apply(input, self.mask, self.shift_sz, self.stride, \
                                                         self.triple_weight, self.flag, self.nonmask_point_idx, self.flatten_offsets,\
                                                        self.sp_x, self.sp_y)



