import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util
import types

class InnerCos(nn.Module):
    def __init__(self, crit='MSE', strength=1, skip=0):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()

        self.strength = strength
        self.target = None
        # To define whether this layer is skipped.
        self.skip = skip
        self.target = None
        #
        # Mention: Quite special way to make it suport multi-gpu training.
        #
        def identity(self):
            return self
        self.loss = torch.cuda.FloatTensor(1)
        self.loss.float = types.MethodType(identity, self.loss)
        self.register_buffer('cos_loss', self.loss)

    def set_mask(self, mask_global, opt):
        mask = util.cal_feat_mask(mask_global, 3, opt.threshold)
        self.mask = mask.squeeze()
        if torch.cuda.is_available:
            self.mask = self.mask.float().cuda()
        self.mask = Variable(self.mask, requires_grad=False)


    def forward(self, in_data):
        self.bs, self.c, _, _ = in_data.size()
        # This the main fix for multi-gpu.
        # Change self.mask to the gpu where the model is.
        self.mask = self.mask.cuda()
        if not self.skip:
            self.former = in_data.narrow(1, 0, self.c//2)
            self.former_in_mask = torch.mul(self.former, self.mask)
            self.target = in_data.narrow(1, self.c // 2, self.c // 2).data.cuda() # the latter part
            self.target = self.target * self.strength
            self.loss = self.criterion(self.former_in_mask * self.strength, Variable(self.target))
        else:
            self.loss = 0
        self.output = in_data
        return self.output

    def __repr__(self):
        skip_str = 'True' if not self.skip else 'False'
        return self.__class__.__name__+ '(' \
              + 'skip: ' + skip_str \
              + ' ,strength: ' + str(self.strength) + ')'
