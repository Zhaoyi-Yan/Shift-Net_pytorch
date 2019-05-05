import torch
import numpy as np
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
import util.shift_op_soft as shift_op_soft


# python offline_shift.py --name='64_64_1_no_overlap' --which_epoch=30

shifted_folder = 'shifted/64_1/train'
img_folder = './datasets/Paris/train'
util.mkdir(shifted_folder)

opt = TrainOptions().parse()

f = glob.glob(img_folder+'/*.JPG')


# make sure that the sizes
#
opt.overlap = 0
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
netG, _, _ = networks.define_G(4, 3, opt.ngf,
                            opt.which_model_netG, opt, mask_little, opt.norm, opt.use_spectral_norm_G, opt.init_type, opt.gpu_ids, opt.init_gain)

if isinstance(netG, torch.nn.DataParallel):
    netG = netG.module

save_dir = os.path.join(opt.checkpoints_dir, opt.name)
load_filename = '%s_net_G.pth' % (opt.which_epoch)
load_path = os.path.join(save_dir, load_filename)
 
state_dict = torch.load(load_path, map_location=str('cuda:0'))
netG.load_state_dict(state_dict, strict=False)


for fl in f:
    vgg19_extractor = util.VGG19FeatureExtractor().to(opt.gpu_ids[0])
    vgg19_extractor_gt = util.VGG19FeatureExtractor().to(opt.gpu_ids[0])
    A = Image.open(fl).convert('RGB')
    A_little = A.resize((opt.fineSize, opt.fineSize), Image.BICUBIC)

    # need resize A and resize back
    A = A.resize((opt.fineSize, opt.fineSize), Image.BICUBIC)
    A = A.resize((256, 256), Image.BICUBIC)

    transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)
    A_trans = transform(A)
    A_little_trans = transform(A_little)

    A_trans = A_trans.view(1, *A_trans.size())
    A_trans_gt = A_trans.clone()
    A_little_trans = A_little_trans.view(1, *A_little_trans.size())

    A_little_trans.narrow(1,0,1).masked_fill_(mask_little, 0.)#2*123.0/255.0 - 1.0
    A_little_trans.narrow(1,1,1).masked_fill_(mask_little, 0.)#2*104.0/255.0 - 1.0
    A_little_trans.narrow(1,2,1).masked_fill_(mask_little, 0.)#2*117.0/255.0 - 1.0

    A_little_trans = torch.cat((A_little_trans, (1 - mask_little).expand(A_little_trans.size(0), 1, A_little_trans.size(2), A_little_trans.size(3)).type_as(A_little_trans)), dim=1)

    A_little_trans = A_little_trans.to(opt.gpu_ids[0])
    A_trans = A_trans.to(opt.gpu_ids[0])
    A_trans_gt = A_trans_gt.to(opt.gpu_ids[0])



    fake_B = netG(A_little_trans)
    # Then cut this patch out and upscale x4.
    mask_patch = fake_B[:, :, int(opt.fineSize/4) + opt.overlap : int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap,\
                            int(opt.fineSize/4) + opt.overlap: int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap]

    # then upscale x 4
    # when fineSize is 128, then upscale x 2.
    # then the mask size is 128*128
    mask_patch_upscaled = F.interpolate(mask_patch, scale_factor=4, mode='bilinear')

    # then paste it to the image of 256*256
    A_trans[:, :, int(256/4) + opt.overlap : int(256/2) + int(256/4) - opt.overlap,\
            int(256/4) + opt.overlap: int(256/2) + int(256/4) - opt.overlap] = mask_patch_upscaled


    # We only need to use the mask_in_latent part in A_feat.
    A_feat = vgg19_extractor(A_trans)
    A_feat_gt = vgg19_extractor_gt(A_trans_gt)

    # For each feature, get the enhance one.
    # relu1_1, relu2_1, relu3_1 (the same size, 1/2, 1/4)
    # only maske_in_latent part are useful
    for i in range(3):
        if i == 0:
            A_feat[i] = A_feat_gt[i]
            continue
        # calculate the mask in latent space.
        mask_in_latent = util.cal_feat_mask(mask_large, i)
        print(i)
        flag = util.cal_flag_given_mask_thred(mask_in_latent, patch_size=1, stride=1, mask_thred=1)
        shifted = shift_op.shift_offline(A_feat[i], shift_sz=1, stride=1, flag=flag)
        A_feat[i] = shifted

    print('Generating feat for images: '+ os.path.basename(fl))
    torch.save(A_feat, os.path.join(shifted_folder, os.path.splitext(os.path.basename(fl))[0]+'.pt'))

    # torch.save(A_feat, 'x.pt')
    # t = torch.load('x.pt')
