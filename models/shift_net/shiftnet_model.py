import torch
from torch.nn import functional as F
import util.util as util
from models import networks
from models.shift_net.base_model import BaseModel
import time


class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'


    def create_random_mask(self):
        if self.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
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
        self.loss_names = ['G_GAN', 'G_L1', 'D']
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


        # batchsize should be 1 for mask_global
        self.mask_global = torch.ByteTensor(1, 1, \
                                 opt.fineSize, opt.fineSize)

        # Here we need to set an artificial mask_global(not to make it broken, so center hole is ok.)
        self.mask_global.zero_()
        self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1

        self.mask_type = opt.mask_type
        self.gMask_opts = {}


        self.wgan_gp = False
        # added for wgan-gp
        if opt.gan_type == 'wgan_gp':
            self.gp_lambda = opt.gp_lambda
            self.ncritic = opt.ncritic
            self.wgan_gp = True


        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.to(self.device)

        # load/define networks
        # self.ng_innerCos_list is the constraint list in netG inner layers.
        # self.ng_mask_list is the mask list constructing shift operation.
        if opt.add_mask2input:
            input_nc = opt.input_nc + 1
        else:
            input_nc = opt.input_nc

        self.netG, self.ng_innerCos_list, self.ng_shift_list = networks.define_G(input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.use_spectral_norm_G, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.use_spectral_norm_D, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1_mask = util.Discounted_L1(opt).to(self.device) # make weights/buffers transfer to the correct device

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if self.wgan_gp:
                opt.beta1 = 0
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)

        # Add mask to real_A
        if self.opt.mask_type == 'center':
            self.mask_global.zero_()
            self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
            self.rand_t, self.rand_l = int(self.opt.fineSize/4) + self.opt.overlap, int(self.opt.fineSize/4) + self.opt.overlap
        elif self.opt.mask_type == 'random':
            self.mask_global = self.create_random_mask().type_as(self.mask_global).view_as(self.mask_global)
        else:
            raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)

        self.set_latent_mask(self.mask_global)

        #print(torch.max(real_A), torch.min(real_A))

        real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(self.mask_global, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(self.mask_global, 0.)#2*117.0/255.0 - 1.0

        if self.opt.add_mask2input:
            # make it 4 dimensions.
            # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
            real_A = torch.cat((real_A, (1 - self.mask_global).expand(real_A.size(0), 1, real_A.size(2), real_A.size(3)).type_as(real_A)), dim=1)

        self.real_A = real_A
        self.real_B = real_B
        self.image_paths = input['A_paths']
    

    def set_latent_mask(self, mask_global):
        for ng_shift in self.ng_shift_list: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global)
        for ng_innerCos in self.ng_innerCos_list: # ITERATE OVER THE LIST OF ng_innerCos_list:
            ng_innerCos.set_mask(mask_global)

    def set_gt_latent(self):
        if not self.opt.skip:
            if self.opt.add_mask2input:
                # make it 4 dimensions.
                # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
                real_B = torch.cat([self.real_B, (1 - self.mask_global).expand(self.real_B.size(0), 1, self.real_B.size(2), self.real_B.size(3)).type_as(self.real_B)], dim=1)
            else:
                real_B = self.real_B
            self.netG(real_B) # input ground truth


    def forward(self):
        self.fake_B = self.netG(self.real_A)

    # Just assume one shift layer.
    def set_flow_src(self):
        self.flow_srcs = self.ng_shift_list[0].get_flow()
        self.flow_srcs = F.interpolate(self.flow_srcs, scale_factor=8, mode='nearest')
        # Just to avoid forgetting setting show_map_false
        self.set_show_map_false()

    # Just assume one shift layer.
    def set_show_map_true(self):
        self.ng_shift_list[0].set_flow_true()

    def set_show_map_false(self):
        self.ng_shift_list[0].set_flow_false()

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_B = self.fake_B
        # Real
        real_B = self.real_B # GroundTruth

        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            # Using the cropped fake_B as the input of D.
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]

            real_B = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]  

        self.pred_fake = self.netD(fake_B.detach())
        self.pred_real = self.netD(real_B)

        if self.wgan_gp:
            self.loss_D_fake = torch.mean(self.pred_fake)
            self.loss_D_real = torch.mean(self.pred_real)

            # calculate gradient penalty
            alpha = torch.rand(real_B.size()).to(self.device)
            x_hat = alpha * real_B.detach() + (1 - alpha) * fake_B.detach()
            x_hat.requires_grad_(True)
            pred_hat = self.netD(x_hat)

            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = self.gp_lambda * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2).mean()

            self.loss_D = self.loss_D_fake - self.loss_D_real + gradient_penalty
        else:
            self.pred_fake = self.netD(fake_B.detach())

            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
                self.loss_D_real = self.criterionGAN (self.pred_real, True)

                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            elif self.opt.gan_type == 're_s_gan':
                self.loss_D = self.criterionGAN(self.pred_real - self.pred_fake, True)

            elif self.opt.gan_type == 're_avg_gan':
                self.loss_D =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), True) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), False)) / 2.
        # for `re_avg_gan`, need to retain graph of D.
        if self.opt.gan_type == 're_avg_gan':
            self.loss_D.backward(retain_graph=True)
        else:
            self.loss_D.backward()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_B = self.fake_B
        # Has been verfied, for square mask, let D discrinate masked patch, improves the results.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
        # Using the cropped fake_B as the input of D.
            fake_B = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                            self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
        pred_fake = self.netD(fake_B)


        if self.wgan_gp:
            self.loss_G_GAN = torch.mean(pred_fake)
        else:
            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_G_GAN = self.criterionGAN(pred_fake, True)

            elif self.opt.gan_type == 're_s_gan':
                pred_real = self.netD (self.real_B)
                self.loss_G_GAN = self.criterionGAN (pred_fake - pred_real, True)

            elif self.opt.gan_type == 're_avg_gan':
                self.pred_real = self.netD(self.real_B)
                self.loss_G_GAN =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), False) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), True)) / 2.


        # If we change the mask as 'center with random position', then we can replacing loss_G_L1_m with 'Discounted L1'.
        self.loss_G_L1, self.loss_G_L1_m = 0, 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A
        # calcuate mask construction loss
        # When mask_type is 'center' or 'random_with_rect', we can add additonal mask region construction loss (traditional L1).
        # Only when 'discounting_loss' is 1, then the mask region construction loss changes to 'discounting L1' instead of normal L1.
        if self.opt.mask_type == 'center' or self.opt.mask_sub_type == 'rect': 
            mask_patch_fake = self.fake_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                                self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            mask_patch_real = self.real_B[:, :, self.rand_t:self.rand_t+self.opt.fineSize//2-2*self.opt.overlap, \
                                        self.rand_l:self.rand_l+self.opt.fineSize//2-2*self.opt.overlap]
            # Using Discounting L1 loss
            self.loss_G_L1_m += self.criterionL1_mask(mask_patch_fake, mask_patch_real)*self.opt.mask_weight

        if self.wgan_gp:
            self.loss_G = self.loss_G_L1 + self.loss_G_L1_m - self.loss_G_GAN * self.opt.gan_weight
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_L1_m + self.loss_G_GAN * self.opt.gan_weight


        # Third add additional netG contraint loss!
        self.ng_loss_value = 0
        if not self.opt.skip:
            for gl in self.ng_innerCos_list:
                self.ng_loss_value += gl.loss
            self.loss_G += self.ng_loss_value

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # for other type of GAN, ncritic = 1.
        if not self.wgan_gp:
            self.ncritic = 1
        # update D
        self.set_requires_grad(self.netD, True)
        for i in range(self.ncritic):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


