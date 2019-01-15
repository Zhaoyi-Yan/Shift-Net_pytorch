#-*-coding:utf-8-*-
from torch.nn import init
from torch.optim import lr_scheduler


from .modules import *

###############################################################################
# Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net



def define_G(input_nc, output_nc, ngf, which_model_netG, opt, mask_global, norm='batch', use_dropout=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    innerCos_list = []
    shift_list = []

    print('input_nc {}'.format(input_nc))
    print('output_nc {}'.format(output_nc))
    print('which_model_netG {}'.format(which_model_netG))

    # Here we need to initlize an artificial mask_global to construct the init model.
    # When training, we need to set mask for special layers(mostly for Shift layers) first.
    # If mask is fixed during training, we only need to set mask for these layers once,
    # else we need to set the masks each iteration, generating new random masks and mask the input
    # as well as setting masks for these special layers.
    print('[CREATED] MODEL')
    if which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'unet_shift_triple':
        netG = UnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'res_unet_shift_triple':
        netG = ResUnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'inception_unet_shift_triple':
        netG = InceptionUnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'soft_unet_shift_triple':
        netG = SoftUnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'patch_soft_unet_shift_triple':
        netG = PatchSoftUnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    elif which_model_netG == 'res_patch_soft_unet_shift_triple':
        netG = ResPatchSoftUnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_dropout=use_dropout)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    print('[CREATED] MODEL')
    print('Constraint in netG:')
    print(innerCos_list)

    print('Shift in netG:')
    print(shift_list)

    return init_net(netG, init_type, init_gain, gpu_ids), innerCos_list, shift_list


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, use_spectral_norm=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, use_spectral_norm=use_spectral_norm)

    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, use_spectral_norm=use_spectral_norm)

    elif which_model_netD == 'densenet':
        # haven't implemented SN for densenet
        netD = DenseNetDiscrimator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)

    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)

