import torch.nn as nn
import torch
from torch.autograd import Variable
import util.util as util

# In nn.Modules, the input is a variable that requires grad.
# However, in the xxFunctions, the input will be translated
# to tensors. In this case, it can be done more freely,
# Do not thinking about the graph, and just do the processing.
# In xxFunction.backward, usually the input will be stored via
# ctx.saved_forbackward. Then input will be transformed to Variables(requires grad)
class InnerCos(nn.Module):
    def __init__(self, crit='MSE', strength=1):
        super(InnerCos, self).__init__()
        self.crit = crit
        self.criterion = torch.nn.MSELoss() if self.crit == 'MSE' else torch.nn.L1Loss()
        
        self.strength = strength
        self.target = None

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
        self.bs, self.c, _, _ = in_data.size()
        self.former = in_data.narrow(1, 0, self.c//2)
        # it is broadcasting here
        self.former_in_mask = torch.mul(self.former, self.mask)

        # the latent of I in L-l(former) should be similar with GT_latent(l, the latter)
        # for multigpu competibility
        # Still cannot handle multigpu case, need improving.
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
        return self.output


    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
