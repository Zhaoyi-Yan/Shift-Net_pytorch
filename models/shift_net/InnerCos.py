import torch.nn as nn
import torch
import torch.nn.functional as F
import util.util as util
from .InnerCosFunction import InnerCosFunction

class InnerCos(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0, layer_to_last=3):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()
        self.strength = strength
        # To define whether this layer is skipped.
        self.skip = skip
        self.layer_to_last = layer_to_last
        # Init a dummy value is fine.
        self.target = torch.tensor(1.0)

   def set_mask(self, mask_global):
        mask_all = util.cal_feat_mask(mask_global, self.layer_to_last)
        self.mask_all = mask_all.float()

	
    def _split_mask(self, cur_bsize):
        # get the visible indexes of gpus and assign correct mask to set of images
        cur_device = torch.cuda.current_device()
        self.cur_mask = self.mask_all[cur_device*cur_bsize:(cur_device+1)*cur_bsize, :, :, :]

   def forward(self, in_data):
        self.bz, self.c, _, _ = in_data.size()
        self._split_mask(self.bz)
        self.cur_mask = self.cur_mask.to(in_data)
        if not self.skip:
            # It works like this:
            # Each iteration contains 2 forward, In the first forward, we input GT, just to get the target.
            # In the second forward, we input corrupted image, then back-propagate the network, the guidance loss works as expected.
            self.output = InnerCosFunction.apply(in_data, self.criterion, self.strength, self.target, self.cur_mask)
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).detach() # the latter part
        else:
            self.output = in_data
        return self.output


    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip: ' + skip_str \
              + 'layer ' + str(self.layer_to_last) + ' to last' \
              + ' ,strength: ' + str(self.strength) + ')'
