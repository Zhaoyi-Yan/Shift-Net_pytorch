import torch
# import numpy as np
from options.train_options import TrainOptions
import torchvision.transforms as transforms
import util.util as util
import os
from PIL import Image
import glob
from util.NonparametricShift import Modified_NonparametricShift
from models import networks
import torch.nn.functional as F
import util.util as util
import util.shift_op as shift_op




mask_folder = 'masks/testing_masks'
test_folder = './datasets/Paris/test'
util.mkdir(mask_folder)

opt = TrainOptions().parse()

f = glob.glob(test_folder+'/*.png')


# make sure that the sizes
#
opt.overlap = 4
opt.fineSize = 64


# batchsize should be 1 for mask_global
mask_little = torch.ByteTensor(1, 1, \
                            opt.fineSize, opt.fineSize)

# Here we need to set an artificial mask_global(center hole is ok.)
mask_little.zero_()
mask_little[:, :, int(opt.fineSize/4) + opt.overlap : int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap,\
                        int(opt.fineSize/4) + opt.overlap: int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap] = 1

mask_large = torch.ByteTensor(1, 1, \
                            256, 256)

# load original 64*64 model
netG = networks.define_G(4, 3, opt.ngf,
                            opt.which_model_netG, opt, mask_little, opt.norm, opt.use_spectral_norm_G, opt.init_type, '0', opt.init_gain)

if isinstance(netG, torch.nn.DataParallel):
    netG = netG.module

save_dir = os.path.join(opt.checkpoints_dir, opt.name)
load_filename = '%s_net_G.pth' % (opt.which_epoch)
load_path = os.path.join(save_dir, load_filename)
 
state_dict = torch.load(load_path, map_location=str('cuda:0'))
netG.load_state_dict(state_dict)


for fl in f:
    vgg19_extractor = util.VGG19FeatureExtractor().to(opt.gpu_ids[0])
    A = Image.open(fl).convert('RGB')
    A_little = A.resize((64, 64), Image.BICUBIC)

    transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    A_trans = transform(A)
    A_little_trans = transform(A_little)


    fake_B = netG(A_little_trans)
    # Then cut this patch out and upscale x4.
    mask_patch = fake_B[:, :, int(opt.fineSize/4) + opt.overlap : int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap,\
                            int(opt.fineSize/4) + opt.overlap: int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap]

    # then upscale x 4
    # when fineSize is 128, then upscale x 2.
    mask_large = F.interpolate(mask_patch, scale_factor=4, mode='bilinear')

    # then paste it to the image of 256*256
    A_trans[:, :, int(opt.fineSize/4) + opt.overlap : int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap,\
            int(opt.fineSize/4) + opt.overlap: int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap] = mask_large


    A_feat = vgg19_extractor(A_trans)

    # For each feature, get the enhance one.
    # relu1_1, relu2_1, relu3_1 (the same size, 1/2, 1/4)
    for i in range(3):
        # calculate the mask in latent space.
        mask_in_latent = util.cal_feat_mask(mask_large, i)
        flag = util.cal_flag_given_mask_thred(mask_in_latent, patch_size=1, stride=1, mask_thred=1)
        print(flag)
        shifted = shift_op.shift_offline(A_feat[i], shift_sz=1, stride=1, flag=flag)

    
    print('Generating mask for test image: '+os.path.basename(fl))
    util.save_image(mask.squeeze().numpy()*255, os.path.join(mask_folder, os.path.splitext(os.path.basename(fl))[0]+'_mask.png'))
