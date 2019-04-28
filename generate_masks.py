import torch
# import numpy as np
from options.train_options import TrainOptions
import util.util as util
import os
from PIL import Image
import glob

mask_folder = 'masks/testing_masks'
test_folder = './datasets/Paris/test'
util.mkdir(mask_folder)

opt = TrainOptions().parse()

f = glob.glob(test_folder+'/*.png')
print(f)

for fl in f:
    mask = torch.zeros(opt.fineSize, opt.fineSize)
    if opt.mask_sub_type == 'fractal':
        assert 1==2, "It is broken now..."
        mask = util.create_walking_mask()  # create an initial random mask.

    elif opt.mask_sub_type == 'rect':
        mask, rand_t, rand_l = util.create_rand_mask(opt)

    elif opt.mask_sub_type == 'island':
        mask = util.wrapper_gmask(opt)
    
    print('Generating mask for test image: '+os.path.basename(fl))
    util.save_image(mask.squeeze().numpy()*255, os.path.join(mask_folder, os.path.splitext(os.path.basename(fl))[0]+'_mask.png'))



