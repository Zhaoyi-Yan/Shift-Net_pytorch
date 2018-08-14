import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util

class InnerCos(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()

        self.strength = strength
        self.target = None
        # To define whether this layer is skipped.
        self.skip = skip

    def set_mask(self, mask_global, opt):
        mask = util.cal_feat_mask(mask_global, 3, opt.threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()
        self.mask = Variable(self.mask, requires_grad=False)

    def set_target(self, targetIn):
        self.target = targetIn

    def get_target(self):
        return self.target

    def forward(self, in_data):
        if not self.skip:
            self.bs, self.c, _, _ = in_data.size()
            self.former = in_data.narrow(1, 0, self.c//2)
            self.former_in_mask = torch.mul(self.former, self.mask)
            current_gpu_id = in_data.get_device()
            if self.target.size() != self.former_in_mask.size():
                self.target = self.target.narrow(0, current_gpu_id * self.bs, (current_gpu_id+1)*self.bs)

            self.loss = self.criterion(self.former_in_mask * self.strength, self.target)

            # I have to put it here!
            # when input is image with mask(the second pass), we
            # Mention only when input is the groundtruth, the target makes sense.
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).clone() # the latter part
            self.target = self.target * self.strength
            self.target = self.target.detach()

            self.output = in_data
        else:
            self.output = in_data
        return self.output


    def backward(self, retain_graph=True):
        if not self.skip:
            self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip: ' + skip_str \
              + ' ,strength: ' + str(self.strength) + ')'
