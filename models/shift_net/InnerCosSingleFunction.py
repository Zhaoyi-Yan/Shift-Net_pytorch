import torch
import torch.nn as nn

class InnerCosSingleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, criterion, strength, target, mask):
        ctx.c = input.size(1)
        ctx.strength = strength
        ctx.criterion = criterion
        if len(target.size()) == 0: # For the first iteration.
            target = target.expand_as(input).type_as(input)

        ctx.save_for_backward(input, target, mask)
        return input


    @staticmethod
    def backward(ctx, grad_output):

        with torch.enable_grad():
            input, target, mask = ctx.saved_tensors
            in_data = input
            in_data_in_mask = torch.mul(in_data, mask)
            if in_data_in_mask.size() != target.size():  # For the last iteration of one epoch
                target = target.narrow(0, 0, 1).expand_as(in_data_in_mask).type_as(in_data_in_mask)
            
            in_data_in_mask_clone = in_data_in_mask.clone().detach().requires_grad_(True)
            ctx.loss = ctx.criterion(in_data_in_mask_clone, target) * ctx.strength
            ctx.loss.backward()

        grad_output += in_data_in_mask_clone.grad  
        
        return grad_output, None, None, None, None