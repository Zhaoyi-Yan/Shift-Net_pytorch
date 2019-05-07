import torch
from torch.nn import functional as F
import util.util as util
from models import networks
from models.shift_net.base_model import BaseModel
import time
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image

# Currently, it just supports center inpainting
# Offline_results: real_A and shifted features.
class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'


    def create_random_mask(self):
        if self.opt.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
                assert 1==2, "It is broken somehow, use another mask_sub_type please"
                mask = util.create_walking_mask()  # create an initial random mask.

            elif self.opt.mask_sub_type == 'rect':
                mask, rand_t, rand_l = util.create_rand_mask(self.opt)
                self.rand_t = rand_t
                self.rand_l = rand_l
                return mask

            elif self.opt.mask_sub_type == 'island':
                mask = util.wrapper_gmask(self.opt)
        return mask

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D', 'style', 'content']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        if self.opt.show_flow:
            self.visual_names = ['real_A', 'fake_B', 'real_B', 'flow_srcs']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']


        # For SR, mask_global is not quite useful.
        # However, it is used to localize the masked region,  and used feature in that region to concat back to the original feature.
        # For now, fineSize shuould be 256*256
        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(self.opt.batchSize, 1, \
                                 opt.fineSize, opt.fineSize)

        # Here we need to set an artificial mask_global(center hole is ok.)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        if len(opt.gpu_ids) > 0:
            self.mask_global = self.mask_global.to(self.device)


        self.netG_SR = networks.define_G_SR(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG_SR, opt, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            self.netD_SR = networks.define_D_SR(opt.which_model_netD, opt.norm, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        # add style extractor
        self.vgg16_extractor = util.VGG16FeatureExtractor().to(self.gpu_ids[0])
        self.vgg16_extractor = torch.nn.DataParallel(self.vgg16_extractor, self.gpu_ids)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

            # VGG loss
            self.criterionL2_style_loss = torch.nn.MSELoss()
            self.criterionL2_content_loss = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG_SR.parameters(),
                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_SR.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)
 
        self.print_networks(opt.verbose)

    def set_input(self, input):
        self.image_paths = input['A_paths']

        # 64*64
        real_A = input['A'].to(self.device)
        # 256*256
        real_B = input['B'].to(self.device)
        # load here ?
        self.shifted_feat = input[]

        assert self.opt.mask_type == 'center', "Only center mask is implemented"

        # overlap = 0 here
        if self.opt.mask_type == 'center':
            self.mask_global.zero_()
            self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.real_A = real_A
        self.real_B = real_B


    def forward(self):
        # crop the shifted features out.
        # For different features, crop the corresponding features out.
        # Need to find a better way to figure it out.
        self.shifted_feat[0] = self.shifted_feat[0][:, :, int(256/4) : int(256/2) + int(256/4), \
                            int(256/4) : int(256/2) + int(256/4)]
        self.shifted_feat[1] = self.shifted_feat[0][:, :, int(128/4) : int(128/2) + int(128/4), \
                            int(128/4) : int(128/2) + int(128/4)]
        self.shifted_feat[2] = self.shifted_feat[0][:, :, int(64/4) : int(64/2) + int(64/4), \
                            int(64/4) : int(64/2) + int(64/4)]

        self.fake_B = self.netG_SR(self.real_A, self.shifted_feat)

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_B = self.fake_B
        # Real
        real_B = self.real_B # GroundTruth

        self.pred_fake = self.netD_SR(fake_B.detach())
        self.pred_real = self.netD_SR(real_B)

        gradient_penalty, _ = util.cal_gradient_penalty(self.netD_SR, real_B, fake_B.detach(), self.device, constant=1, lambda_gp=self.opt.gp_lambda)
        self.loss_D_fake = torch.mean(self.pred_fake)
        self.loss_D_real = -torch.mean(self.pred_real)

        self.loss_D = self.loss_D_fake + self.loss_D_real + gradient_penalty

        self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        real_B = self.real_B

        pred_fake = self.netD_SR(fake_B)

        self.loss_G_GAN = -torch.mean(pred_fake) * self.opt.gan_weight

        self.loss_G_L1 = 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        self.loss_G = self.loss_G_L1 + self.loss_G_GAN


        # Finally, add style loss
        vgg_ft_fakeB = self.vgg16_extractor(fake_B)
        vgg_ft_realB = self.vgg16_extractor(real_B)
        self.loss_style = 0
        self.loss_content = 0

        for i in range(3):
            self.loss_style += self.criterionL2_style_loss(util.gram_matrix(vgg_ft_fakeB[i]), util.gram_matrix(vgg_ft_realB[i]))
            self.loss_content += self.criterionL2_content_loss(vgg_ft_fakeB[i], vgg_ft_realB[i])

        self.loss_style *= self.opt.style_weight
        self.loss_content *= self.opt.content_weight

        self.loss_G += (self.loss_style + self.loss_content)

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # update D
        self.set_requires_grad(self.netD_SR, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD_SR, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


