import numpy as np
import torch
import torch.nn as nn
# Input is a tensor
# Input has to be 1*N*H*W
# output and ind: are all 0-index!!

# Additional params: pre-calculated constant tensor. `sp_x` and `px_y`.
# They are just for Advanced Indexing.
# sp_x: [0,0,..,0, 1,1,...,1, ..., 31,31,...,31],   length is 32*32, it is a list
# sp_y: [0,1,2,...,31,  0,1,2,...,31,  0,1,2,...,31]  length is 32*32, it is a LongTensor(cuda.LongTensor)
class MaxCoord():
    def __init__(self):
        pass

    def update_output(self, input, sp_x, sp_y):
        input_dim = input.dim()
        assert input.dim() == 4, "Input must be 3D or 4D(batch)."
        assert input.size(0) == 1, "The first dimension of input has to be 1!"

        output = torch.zeros_like(input)

        _,c_max = torch.max(input, 1)
        
        print(c_max.shape)

        c_max_flatten = c_max.view(-1)
        
        print(c_max_flatten.shape)

        output[:, c_max_flatten, sp_x, sp_y] = 1
        ind = c_max_flatten

        return output, ind
