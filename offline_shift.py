import torch
import numpy as np
import random
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


# python offline_shift.py --name='64_64_1_no_overlap' --which_epoch=30   --which_model_netG='unet_shift_triple_64_1'

shifted_folder = 'shifted/64_2/test'
resized_folder = 'datasets/resized_paris/test'
fake_B_folder = 'fakeB/64_2/test'
# img_folder = './datasets/Paris/train'
util.mkdir(shifted_folder)
util.mkdir(resized_folder)
util.mkdir(fake_B_folder)


opt = TrainOptions().parse()

f = glob.glob(resized_folder+'/*.png')

# the training datasets should be strictly 256*256
# create a new dataset
opt.loadSize = 256

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
                            256, 256).zero_()
mask_large[:, :, int(256/4) + opt.overlap : int(256/2) + int(256/4) - opt.overlap,\
                        int(256/4) + opt.overlap: int(256/2) + int(256/4) - opt.overlap] = 1

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

count = 0
for fl in f:
    count += 1
    print('Precessing %d-th' %(count))
    vgg19_extractor = util.VGG19FeatureExtractor().to(opt.gpu_ids[0])
    vgg19_extractor_gt = util.VGG19FeatureExtractor().to(opt.gpu_ids[0])
    transform_list = [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]

    transform = transforms.Compose(transform_list)

    A = Image.open(fl).convert('RGB')
    w, h = A.size
    if w < h:
        ht_1 = opt.loadSize * h // w
        wd_1 = opt.loadSize
        A = A.resize((wd_1, ht_1), Image.BICUBIC)
    else:
        wd_1 = opt.loadSize * w // h
        ht_1 = opt.loadSize
        A = A.resize((wd_1, ht_1), Image.BICUBIC)

    A_trans = transform(A)
    h = A_trans.size(1)
    w = A_trans.size(2)
    w_offset = random.randint(0, max(0, w - 256 - 1))
    h_offset = random.randint(0, max(0, h - 256 - 1))


    A_trans = A_trans[:, h_offset:h_offset + 256,
            w_offset:w_offset + 256]

    A_trans = A_trans.view(1, *A_trans.size())
    A_trans_gt = A_trans.clone()

    # saving resized images to resized_folder(For only training images)
    # A_trans_numpy = util.tensor2im(A_trans)
    # print('Saving resized image: '+os.path.splitext(os.path.basename(fl))[0]+'_resized.png')
    # util.save_image(A_trans_numpy, os.path.join(resized_folder, os.path.splitext(os.path.basename(fl))[0]+'_resized.png'))
    

    # get LR and HR(down and up)
    A_little_trans = F.interpolate(A_trans, size=(opt.fineSize, opt.fineSize), mode='bilinear')

    A_tmp = F.interpolate(A_trans, size=(opt.fineSize, opt.fineSize), mode='bilinear')
    A_trans = F.interpolate(A_tmp, size=(256, 256), mode='bilinear')


    A_little_trans.narrow(1,0,1).masked_fill_(mask_little, 0.)#2*123.0/255.0 - 1.0
    A_little_trans.narrow(1,1,1).masked_fill_(mask_little, 0.)#2*104.0/255.0 - 1.0
    A_little_trans.narrow(1,2,1).masked_fill_(mask_little, 0.)#2*117.0/255.0 - 1.0

    A_little_trans = torch.cat((A_little_trans, (1 - mask_little).expand(A_little_trans.size(0), 1, A_little_trans.size(2), A_little_trans.size(3)).type_as(A_little_trans)), dim=1)

    A_little_trans = A_little_trans.to(opt.gpu_ids[0])
    A_trans = A_trans.to(opt.gpu_ids[0])
    A_trans_gt = A_trans_gt.to(opt.gpu_ids[0])

    # fake_B can also be stored offline
    fake_B = netG(A_little_trans)
    print('Generating fake_B for images: '+ os.path.basename(fl))
    torch.save(fake_B, os.path.join(fake_B_folder, os.path.splitext(os.path.basename(fl))[0]+'_fakeB.pt'))


    # # Then cut this patch out and upscale x4.
    # mask_patch = fake_B[:, :, int(opt.fineSize/4) + opt.overlap : int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap,\
    #                         int(opt.fineSize/4) + opt.overlap: int(opt.fineSize/2) + int(opt.fineSize/4) - opt.overlap]

    # # then upscale x 4
    # # when fineSize is 128, then upscale x 2.
    # # then the mask size is 128*128
    # mask_patch_upscaled = F.interpolate(mask_patch, scale_factor=4, mode='bilinear')

    # # then paste it to the image of 256*256
    # A_trans[:, :, int(256/4) + opt.overlap : int(256/2) + int(256/4) - opt.overlap,\
    #         int(256/4) + opt.overlap: int(256/2) + int(256/4) - opt.overlap] = mask_patch_upscaled


    # # We only need to use the mask_in_latent part in A_feat.
    # A_feat = vgg19_extractor(A_trans)
    # A_feat_gt = vgg19_extractor_gt(A_trans_gt)

    # # For each feature, get the enhance one.
    # # relu1_1, relu2_1, relu3_1 (the same size, 1/2, 1/4)
    # # only maske_in_latent part are useful
    # for i in range(3):
    #     if i == 0:
    #         A_feat_gt[i] *= mask_large.float().to(A_feat[i])
    #         A_feat[i] = A_feat_gt[i]
    #         continue
    #     # calculate the mask in latent space.
    #     mask_in_latent = util.cal_feat_mask(mask_large, i)
    #     flag = util.cal_flag_given_mask_thred(mask_in_latent, patch_size=1, stride=1, mask_thred=1)
    #     shifted = shift_op.shift_offline(A_feat[i], shift_sz=1, stride=1, flag=flag)
    #     A_feat[i] = shifted

    # print('Generating feat for images: '+ os.path.basename(fl))
    # torch.save(A_feat, os.path.join(shifted_folder, os.path.splitext(os.path.basename(fl))[0]+'.pt'))

