#-*-coding:utf-8-*-
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from PIL import Image
import torch.nn.functional as F
from .base_model import BaseModel
from . import networks

class ShiftNetModel(BaseModel):
    def name(self):
        return 'ShiftNetModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.opt = opt
        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)

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
            res = 0.06 # the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
            density = 0.25
            MAX_SIZE = 10000
            maxPartition = 30
            low_pattern = torch.rand(1, 1, int(res*MAX_SIZE), int(res*MAX_SIZE)).mul(255)
            pattern = F.upsample(low_pattern, (MAX_SIZE, MAX_SIZE), mode='bilinear').data
            low_pattern = None
            pattern.div_(255)
            pattern = torch.lt(pattern,density).byte()  # 25% 1s and 75% 0s
            pattern = torch.squeeze(pattern).byte()
            print('...Random pattern generated')
            self.gMask_opts['pattern'] = pattern
            self.gMask_opts['MAX_SIZE'] = MAX_SIZE
            self.gMask_opts['fineSize'] = opt.fineSize
            self.gMask_opts['maxPartition'] = maxPartition
            self.gMask_opts['mask_global'] = self.mask_global
            self.mask_global = util.create_gMask(self.gMask_opts) # create an initial random mask.


        self.wgan_gp = False
        # added for wgan-gp
        if opt.gan_type == 'wgan_gp':
            self.gp_lambda = opt.gp_lambda
            self.ncritic = opt.ncritic
            self.wgan_gp = True


        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.mask_global = self.mask_global.cuda()

        # load/define networks
        # self.ng_innerCos_list is the constraint list in netG inner layers.
        # self.ng_mask_list is the mask list constructing shift operation.
        self.netG, self.ng_innerCos_list, self.ng_shift_list = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt, self.mask_global, opt.norm, opt.use_dropout, opt.init_type, self.gpu_ids, opt.init_gain) # add opt, we need opt.shift_sz and other stuffs
        if self.isTrain:
            use_sigmoid = False
            if opt.gan_type == 'vanilla':
                use_sigmoid = True  # only vanilla GAN using BCECriterion
            # don't use cGAN
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, opt.init_gain)
        if not self.isTrain or opt.continue_train:
            print('Loading pre-trained network!')
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(gan_type=opt.gan_type, tensor=self.Tensor)
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

            print('---------- Networks initialized -------------')
            networks.print_network(self.netG)
            if self.isTrain:
                networks.print_network(self.netD)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths']

        # Add mask to input_A
        # When the mask is random, or the mask is not fixed, we all need to create_gMask
        if self.fixed_mask:
            if self.opt.mask_type == 'center':
                self.mask_global.zero_()
                self.mask_global[:, :, int(self.opt.fineSize/4) + self.opt.overlap : int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap,\
                                    int(self.opt.fineSize/4) + self.opt.overlap: int(self.opt.fineSize/2) + int(self.opt.fineSize/4) - self.opt.overlap] = 1
            elif self.opt.mask_type == 'random':
                self.mask_global = util.create_gMask(self.gMask_opts).type_as(self.mask_global)
            else:
                raise ValueError("Mask_type [%s] not recognized." % self.opt.mask_type)
        else:
            self.mask_global = util.create_gMask(self.gMask_opts).type_as(self.mask_global)

        self.set_latent_mask(self.mask_global, 3, self.opt.threshold)

        self.input_A.narrow(1,0,1).masked_fill_(self.mask_global, 2*123.0/255.0 - 1.0)
        self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)
        self.input_A.narrow(1,2,1).masked_fill_(self.mask_global, 2*117.0/255.0 - 1.0)


    def set_latent_mask(self, mask_global, layer_to_last, threshold):
        self.ng_shift_list[0].set_mask(mask_global, layer_to_last, threshold)

    # It is quite convinient, as one forward-pass, all the innerCos will get the GT_latent!
    def set_gt_latent(self):
        self.netG.forward(Variable(self.input_B, requires_grad=False)) # input ground truth

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B)

    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        fake_AB = self.fake_B
        # Real
        real_AB = self.real_B # GroundTruth


        if self.wgan_gp:
            self.pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = torch.mean(self.pred_fake)

            self.pred_real = self.netD(real_AB)
            self.loss_D_real = torch.mean(self.pred_real)

            # calculate gradient penalty
            if self.use_gpu:
                alpha = torch.rand(real_AB.size()).cuda()
            else:
                alpha = torch.rand(real_AB.size())

            x_hat = Variable(alpha * real_AB.data + (1 - alpha) * fake_AB.detach().data, requires_grad=True)

            pred_hat = self.netD(x_hat)
            if self.use_gpu:
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            else:
                gradients = torch.autograd.grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

            gradient_penalty = self.gp_lambda * ((gradients.view(gradients.size()[0], -1).norm(2, 1) - 1) ** 2).mean()

            self.loss_D = self.loss_D_fake - self.loss_D_real + gradient_penalty
        else:
            self.pred_fake = self.netD(fake_AB.detach())
            self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
            self.pred_real = self.netD(real_AB)
            self.loss_D_real = self.criterionGAN(self.pred_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        # When two losses are ready, together backward.
        # It is op, so the backward will be called from a leaf.(quite different from LuaTorch)
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_AB = self.fake_B
        pred_fake = self.netD(fake_AB)
        if self.wgan_gp:
            self.loss_G_GAN = torch.mean(pred_fake)
        else:
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A

        if self.wgan_gp:
            self.loss_G = self.loss_G_L1 - self.loss_G_GAN * self.opt.gan_weight
        else:
            self.loss_G = self.loss_G_L1 + self.loss_G_GAN * self.opt.gan_weight

        # Third add additional netG contraint loss!
        self.ng_loss_value = 0
        if not self.opt.skip:
            for gl in self.ng_innerCos_list:
                self.ng_loss_value += Variable(gl.loss, requires_grad=True)
            self.loss_G += self.ng_loss.value

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

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

