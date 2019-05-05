import numpy as np
from .NonparametricShift import Modified_NonparametricShift
import torch
import util as util
import time
import torch.nn.functional as F


def shift_offline(input, stride, mask, mask_thred, shift_sz, fuse=True):
    assert input.dim() == 4, "Input Dim has to be 4"
    assert mask.dim() == 4, "Mask Dim has to be 4"
    mask = mask.to(input)
    bz, c, h, w = input.size()

    # former and latter are all tensors
    former_all = input.clone()
    latter_all = input.clone()
    shift_masked_all = torch.Tensor(former_all.size()).type_as(former_all) # addition feature

    # extract patches from latter.
    latter_all_pad = F.pad(latter_all, [shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2], 'constant', 0)
    latter_all_windows = latter_all_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
    latter_all_windows = latter_all_windows.contiguous().view(bz, -1, c, shift_sz, shift_sz)

    # Extract patches from mask
    # Mention: mask here must be 1*1*H*W
    m_pad = F.pad(mask, (shift_sz//2, shift_sz//2, shift_sz//2, shift_sz//2), 'constant', 0)
    m = m_pad.unfold(2, shift_sz, stride).unfold(3, shift_sz, stride)
    m = m.contiguous().view(bz, 1, -1, shift_sz, shift_sz)

    # This two line of code can replace `cal_flag_given_mask_thred`
    m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
    # mm: the masked reigon is filled with 0, nonmasked region is filled with 1.
    mm = m.le(mask_thred/(1.*shift_sz**2)).float() # bz*1*(32*32)*1*1

    fuse_weight = torch.eye(shift_sz).view(1, 1, shift_sz, shift_sz).type_as(input)

    for idx in range(bz):
        mm_cur = mm[idx]
        # latter_win = latter_all_windows.narrow(0, idx, 1)[0]
        latter_win = latter_all_windows.narrow(0, idx, 1)[0]
        former = former_all.narrow(0, idx, 1)

        # normalize latter for each patch.
        latter_den = torch.sqrt(torch.einsum("bcij,bcij->b", [latter_win, latter_win]))
        latter_den = torch.max(latter_den, torch.Tensor([1e-4]).to(input))

        latter_win_normed = latter_win/latter_den.view(-1, 1, 1, 1)
        y_i = F.conv2d(former, latter_win_normed, stride=1, padding=shift_sz//2)

        # conv implementation for fuse scores to encourage large patches
        if fuse:
            y_i = y_i.view(1, 1, h*w, h*w) # make all of depth of spatial resolution.
            y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)

            y_i = y_i.contiguous().view(1, h, w, h, w)
            y_i = y_i.permute(0, 2, 1, 4, 3)
            y_i = y_i.contiguous().view(1, 1, h*w, h*w)

            y_i = F.conv2d(y_i, fuse_weight, stride=1, padding=1)
            y_i = y_i.contiguous().view(1, w, h, w, h)
            y_i = y_i.permute(0, 2, 1, 4, 3)

        y_i = y_i.contiguous().view(1, h*w, h, w) # 1*(32*32)*32*32

        # firstly, wash away the masked reigon.
        # multiply `mm` means (:, index_masked, :, :) will be 0.
        y_i = y_i * mm_cur

        # Then apply softmax to the nonmasked region.
        cosine = F.softmax(y_i*10, dim=1)

        # Finally, dummy parameters of masked reigon are filtered out.
        cosine = cosine * mm_cur

        # paste
        shift_i = F.conv_transpose2d(cosine, latter_win, stride=1, padding=shift_sz//2)/9.
        shift_masked_all[idx] = shift_i

    return shift_masked_all

        
