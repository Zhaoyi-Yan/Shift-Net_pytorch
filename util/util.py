from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import random
import inspect, re
import numpy as np
import os
import collections
import math
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize

def create_masks(opt, N=10):
    masks = []
    masks_resized = []
    for _ in range(N):
        mask = wrapper_gmask (opt).cpu().numpy()
        masks.append(mask)
        
        mask_resized = resize(np.squeeze(mask), (64, 64))
        masks_resized.append(mask_resized)
        
    return np.array(masks_resized), np.array(masks)

''''''
class OptimizerMask:
    '''
    This class is designed to speed up inference time to cover the over all image with the minimun number of generated mask during training.
    It is used in the notebook to create masks covering the entire image.
    '''
    def __init__(self, masks, stop_criteria=0.85):
        self.masks = masks
        self.indexes = []
        self.stop_criteria = stop_criteria

    def get_iou(self):
        intersection = np.matmul(self.masks, self.masks.T)
        diag = np.diag(intersection)
        outer_add = np.add.outer(diag, diag)
        self.iou = intersection / outer_add
        self.shape = self.iou.shape

    def _is_finished(self):
        masks = self.masks[self.indexes]
        masks = np.sum(masks, axis=0)
        masks[masks > 0] = 1
        area_coverage = np.sum(masks) / np.product(masks.shape)
        print(area_coverage)
        if area_coverage < self.stop_criteria:
            return False
        else:
            return True

    def mean(self):
        _mean = np.mean(np.sum(self.masks[self.indexes], axis=-1)) / (64 * 64)
        print(_mean)

    def _get_next_indexes(self):
        ious = self.iou[self.indexes]
        _mean_iou = np.mean(ious, axis=0)
        idx = np.argmin(_mean_iou)
        self.indexes = np.append(self.indexes, np.argmin(_mean_iou))

    def _solve(self):
        self.indexes = list(np.unravel_index(np.argmin(self.iou), self.shape))
        # print(self.indexes)
        while not self._is_finished():
            self._get_next_indexes()

    def get_masks(self):
        masks = self.masks[self.indexes]
        full = np.ones_like(masks[0])
        left = full - (np.mean(masks, axis=0) > 0)
        return left.reshape((64, 64))

    def solve(self):
        self._solve()


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# Remove dummy dim from a tensor.
# Useful when input is 4 dims.
def rm_extra_dim(image):
    if image.dim() == 3:
        return image[:3, :, :]
    elif image.dim() == 4:
        return image[:, :3, :, :]
    else:
        raise NotImplementedError


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)



def wrapper_gmask(opt):
    # batchsize should be 1 for mask_global
    mask_global = torch.ByteTensor(1, 1, \
                                        opt.fineSize, opt.fineSize)

    res = 0.06  # the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
    density = 0.25
    MAX_SIZE = 350
    maxPartition = 30
    low_pattern = torch.rand(1, 1, int(res * MAX_SIZE), int(res * MAX_SIZE)).mul(255)
    pattern = F.interpolate(low_pattern, (MAX_SIZE, MAX_SIZE), mode='bilinear').detach()
    low_pattern = None
    pattern.div_(255)
    pattern = torch.lt(pattern, density).byte()  # 25% 1s and 75% 0s
    pattern = torch.squeeze(pattern).byte()

    gMask_opts = {}
    gMask_opts['pattern'] = pattern
    gMask_opts['MAX_SIZE'] = MAX_SIZE
    gMask_opts['fineSize'] = opt.fineSize
    gMask_opts['maxPartition'] = maxPartition
    gMask_opts['mask_global'] = mask_global
    return create_gMask(gMask_opts)  # create an initial random mask.

def create_gMask(gMask_opts, limit_cnt=1):
    pattern = gMask_opts['pattern']
    mask_global = gMask_opts['mask_global']
    MAX_SIZE = gMask_opts['MAX_SIZE']
    fineSize = gMask_opts['fineSize']
    maxPartition=gMask_opts['maxPartition']
    if pattern is None:
        raise ValueError
    wastedIter = 0
    while wastedIter <= limit_cnt:
        x = random.randint(1, MAX_SIZE-fineSize)
        y = random.randint(1, MAX_SIZE-fineSize)
        mask = pattern[y:y+fineSize, x:x+fineSize] # need check
        area = mask.sum()*100./(fineSize*fineSize)
        if area>20 and area<maxPartition:
            break
        wastedIter += 1
    if mask_global.dim() == 3:
        mask_global = mask.expand(1, mask.size(0), mask.size(1))
    else:
        mask_global = mask.expand(1, 1, mask.size(0), mask.size(1))
    return mask_global

def create_rand_mask(h=256, w=256, mask_size=64, overlap=0.25):
    mask = np.zeros((h, w))
    positions = []
    step = int(overlap * mask_size)
    for y in range(0, h-mask_size+1, step):
        for x in range(0, w-mask_size+1, step):
            positions.append([y, x])
    arr = np.array(range(len(positions)))
    idx = np.random.choice(arr)
    pos = positions[idx]
    y, x = pos
    mask[y:y + mask_size, x:x + mask_size] = 1
    mask = mask[np.newaxis, ...][np.newaxis, ...]
    return torch.ByteTensor(mask).cuda()

action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
def random_walk(canvas, ini_x, ini_y, length):
    x = ini_x
    y = ini_y
    img_size = canvas.shape[-1]
    x_list = []
    y_list = []
    for i in range(length):
        r = random.choice(range(len(action_list)))
        x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size - 1)
        y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size - 1)
        x_list.append(x)
        y_list.append(y)
    canvas[np.array(x_list), np.array(y_list)] = 0
    return canvas

def create_mask():
    canvas = np.ones((256, 256)).astype("i")
    ini_x = random.randint(0, 255)
    ini_y = random.randint(0, 255)
    print(ini_x, ini_y)
    return random_walk(canvas, ini_x, ini_y, 128 ** 2)

# inMask is tensor should be 1*1*256*256 float
# Return: ByteTensor
def cal_feat_mask(inMask, nlayers):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    assert inMask.size(0) == 1, "the first dimension must be 1 for mask"
    inMask = inMask.float()
    ntimes = 2**nlayers
    inMask = F.interpolate(inMask, (inMask.size(2)//ntimes, inMask.size(3)//ntimes), mode='nearest')
    inMask = inMask.detach().byte()

    return inMask

# It is only for patch_size=1 for now.
def cal_flag_given_mask_thred(img, mask, patch_size, stride, mask_thred):
    assert img.dim() == 3, 'img has to be 3 dimenison!'
    assert mask.dim() == 2, 'mask has to be 2 dimenison!'

    mask = mask.float()
    # This line of code is for further development of supporting patch_size > 1.
    # mask = F.pad(mask, (patch_size//2, patch_size//2, patch_size//2, patch_size//2), 'constant', 0)
    m = mask.unfold(0, patch_size, stride).unfold(1, patch_size, stride)
    m = m.contiguous().view(1, -1, patch_size, patch_size)
    m = torch.mean(torch.mean(m, dim=2, keepdim=True), dim=3, keepdim=True)
    # Adding eps=1e-4 is important here.
    mm = m.gt(mask_thred/(1.*patch_size**2 + 1e-4)).long()
    flag = mm.view(-1)

    # Obsolete Method
    # dim = img.dim()
    # _, H, W = img.size(dim - 3), img.size(dim - 2), img.size(dim - 1)
    # nH = int(math.floor((H - patch_size) / stride + 1))
    # nW = int(math.floor((W - patch_size) / stride + 1))
    # N = nH * nW

    # flag = torch.zeros(N).long()
    # for i in range(N):
    #     h = int(math.floor(i / nW))
    #     w = int(math.floor(i % nW))
    #     mask_tmp = mask[h * stride:h * stride + patch_size,
    #                w * stride:w * stride + patch_size]

    #     if torch.sum(mask_tmp) < mask_thred:
    #         pass
    #     else:
    #         flag[i] = 1

    return flag



def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def info(object, spacing=10, collapse=1):
    """Print methods and doc strings.
    Takes module, class, list, dictionary, or string."""
    methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
    processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
    print( "\n".join(["%s %s" %
                     (method.ljust(spacing),
                      processFunc(str(getattr(object, method).__doc__)))
                     for method in methodList]) )

def varname(p):
    for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
        m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
        if m:
            return m.group(1)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

'''
 https://github.com/WonwoongCho/Generative-Inpainting-pytorch/blob/master/util.py#L229-L333
'''
def flow_to_image(flow):
    """Transfer flow map to image.
    Part of code forked from flownet.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]
        v = flow[i, :, :, 1]
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)
        u[idxunknow] = 0
        v[idxunknow] = 0
        maxu = max(maxu, np.max(u))
        minu = min(minu, np.min(u))
        maxv = max(maxv, np.max(v))
        minv = min(minv, np.min(v))
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(maxrad, np.max(rad))
        u = u/(maxrad + np.finfo(float).eps)
        v = v/(maxrad + np.finfo(float).eps)
        img = compute_color(u, v)
        out.append(img)
    return np.float32(np.uint8(out))


"""
   flow: N*h*w*2
        Indicating which pixel will shift to the location.
   mask: N*(h*w)
"""
def highlight_flow(flow, mask):
    """Convert flow into middlebury color code image.
    """
    assert flow.dim() == 4 and mask.dim() == 2
    assert flow.size(0) == mask.size(0)
    assert flow.size(3) == 2
    bz, h, w, _ = flow.shape
    out = torch.zeros(bz, 3, h, w).type_as(flow)
    for idx in range(bz):
        mask_index = (mask[idx] == 1).nonzero()
        img = torch.ones(3, h, w).type_as(flow) * 144.
        u = flow[idx, :, :, 0]
        v = flow[idx, :, :, 1]
        # It is quite slow here.
        for h_i in range(h):
            for w_j in range(w):
                p = h_i*w + w_j
                #If it is a masked pixel, we get which pixel that will replace it.
                # DO NOT USE `if p in mask_index:`, it is slow.
                if torch.sum(mask_index == p).item() != 0:
                    ui = u[h_i,w_j]
                    vi = v[h_i,w_j]
                    img[:, int(ui), int(vi)] = 255.
                    img[:, h_i, w_j] = 200. # Also indicating where the mask is.
        out[idx] = img
    return out


def compute_color(u,v):
    h, w = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0
    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)
    rad = np.sqrt(u**2+v**2)
    a = np.arctan2(-v, -u) / np.pi
    fk = (a+1) / 2 * (ncols - 1) + 1
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0
    for i in range(np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)
        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))
    return img


def make_color_wheel():
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols, 3])
    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255
    return colorwheel