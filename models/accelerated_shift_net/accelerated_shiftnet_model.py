import torch

import util.util as util
from models import networks
from models.shift_net.base_model import BaseModel


class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'


    def create_random_mask(self):
        if self.mask_type == 'random':
            if self.opt.mask_sub_type == 'fractal':
                mask = util.create_walking_mask ()  # create an initial random mask.

            elif self.opt.mask_sub_type == 'rect':
                mask = util.create_rand_mask ()

            elif self.opt.mask_sub_type == 'island':
                mask = util.wrapper_gmask (self.opt)
        return mask

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ['G_GAN', 'G_L1', 'D']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
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
        self.fixed_mask = opt.fixed_mask if opt.mask_type == 'center' else 0
        if opt.mask_type == 'center':
            assert opt.fixed_mask == 1, "Center mask must be fixed mask!"

        if self.mask_type == 'random':
            self.create_random_mask()

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
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()

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
        # When the mask is random, or the mask is not fixed, we all need to create_gMask
        if self.fixed_mask:
            if self.opt.mask_type == 'center':
                self.mask_global.zero_()
                self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                    int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
            elif self.opt.mask_type == 'random':
                self.mask_global = self.create_random_mask().type_as(self.mask_global)
            else:
                raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        else:
            self.mask_global = self.create_random_mask().type_as(self.mask_global)

        self.set_latent_mask(self.mask_global, 3)

        #print(torch.max(real_A), torch.min(real_A))

        real_A.narrow(1,0,1).masked_fill_(self.mask_global, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(self.mask_global, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(self.mask_global, 0.)#2*117.0/255.0 - 1.0

        if self.opt.add_mask2input:
            # make it 4 dimensions.
            # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
            real_A = torch.cat((real_A, (1 - self.mask_global).expand(self.opt.batchSize, 1, \
                                     self.opt.fineSize, self.opt.fineSize).type_as(real_A)), dim=1)

        self.real_A = real_A
        self.real_B = real_B
        self.image_paths = input['A_paths']

    # TODO: it has not been implemented totally.
    def set_input_with_mask(self, input, mask):
        real_A = input['A'].to(self.device)
        real_B = input['B'].to(self.device)

        self.mask_global = mask

        self.set_latent_mask(mask, 3)

        real_A.narrow(1,0,1).masked_fill_(mask, 0.)#2*123.0/255.0 - 1.0
        real_A.narrow(1,1,1).masked_fill_(mask, 0.)#2*104.0/255.0 - 1.0
        real_A.narrow(1,2,1).masked_fill_(mask, 0.)#2*117.0/255.0 - 1.0

        self.real_A = real_A
        self.real_B = real_B
        self.image_paths = input['A_paths']       

    def set_latent_mask(self, mask_global, layer_to_last):
        for ng_shift in self.ng_shift_list: # ITERATE OVER THE LIST OF ng_shift_list
            ng_shift.set_mask(mask_global, layer_to_last)

    def set_gt_latent(self):
        if not self.opt.skip:
            if self.opt.add_mask2input:
                # make it 4 dimensions.
                # Mention: the extra dim, the masked part is filled with 0, non-mask part is filled with 1.
                real_B = torch.cat([self.real_B, (1 - self.mask_global).expand(self.opt.batchSize, 1, \
                           self.opt.fineSize, self.opt.fineSize).type_as(self.real_B)], dim=1)
            else:
                real_B = self.real_B
            self.netG(real_B) # input ground truth

    def forward(self):
        self.fake_B = self.netG(self.real_A)
        print(self.ng_shift_list[0].get_flow().size())
        assert 1==2

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        real_AB = self.real_B # GroundTruth

        self.pred_fake = self.netD(fake_AB.detach())
        self.pred_real = self.netD(real_AB)

        if self.wgan_gp:
            self.loss_D_fake = torch.mean(self.pred_fake)
            self.loss_D_real = torch.mean(self.pred_real)

            # calculate gradient penalty
            alpha = torch.rand(real_AB.size()).to(self.device)
            x_hat = alpha * real_AB.detach() + (1 - alpha) * fake_AB.detach()
            x_hat.requires_grad_(True)
            pred_hat = self.netD(x_hat)

            gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = self.gp_lambda * ((gradients.view(gradients.size(0), -1).norm(2, 1) - 1) ** 2).mean()

            self.loss_D = self.loss_D_fake - self.loss_D_real + gradient_penalty
        else:
            self.pred_fake = self.netD(fake_AB.detach())

            if self.opt.gan_type in ['vanilla', 'lsgan']:
                self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
                self.loss_D_real = self.criterionGAN (self.pred_real, True)

                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

            elif self.opt.gan_type == 're_s_gan':
                self.loss_D = self.criterionGAN (self.pred_real - self.pred_fake, True)

            elif self.opt.gan_type == 're_avg_gan':
                self.loss_D =  (self.criterionGAN (self.pred_real - torch.mean(self.pred_fake), True) \
                               + self.criterionGAN (self.pred_fake - torch.mean(self.pred_real), False)) / 2.

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB)

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


        self.loss_G_L1 = 0
        self.loss_G_L1 += self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        if self.wgan_gp:
            self.loss_G = self.loss_G_L1 - self.loss_G_GAN * self.opt.gan_weight
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight


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
        for i in range(self.ncritic):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


