import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--dataroot', default='./datasets/Paris/train', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=350, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        parser.add_argument('--which_model_netD', type=str, default='densenet', help='selects model to use for netD')
        parser.add_argument('--which_model_netG', type=str, default='acc_unet_shift_triple', help='selects model to use for netG [unet_256| unet_shift_triple|acc_unet_shift_triple|soft_unet_shift_triple]')
        parser.add_argument('--model', type=str, default='accelerated_shiftnet',
                                 help='chooses which model to use. [shiftnet|accelerated_shiftnet|soft_shiftnet|test]')
        parser.add_argument('--triple_weight', type=float, default=1, help='The weight on the gradient of skip connections from the gradient of swapped')
        parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2')
        parser.add_argument('--dataset_mode', type=str, default='aligned', help='chooses how datasets are loaded. [aligned | aligned_resized | single]')
        parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, default='./log', help='models are saved here')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')

        parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator') #it means if using '--use_dropout then this option is true.
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        ## new added
        parser.add_argument('--mask_type', type=str, default='random',
                            help='the type of mask you want to apply, \'center\' or \'random\'')
        parser.add_argument('--mask_sub_type', type=str, default='island',
                            help='the type of mask you want to apply, \'rect \' or \'fractal \ or \'island \'')

        parser.add_argument('--fixed_mask', type=int, default=1, help='1 or 0, whether mask is fixed')
        parser.add_argument('--lambda_A', type=int, default=100, help='weight on L1 term in objective')
        parser.add_argument('--stride', type=int, default=1, help='should be dense, 1 is a good option.')
        parser.add_argument('--shift_sz', type=int, default=1, help='when it is 3, the next opt `mask_thred` should not be 1')
        parser.add_argument('--mask_thred', type=int, default=1, help='number to decide whether a patch is masked')
        parser.add_argument('--bottleneck', type=int, default=512, help='neurals of fc')
        parser.add_argument('--gp_lambda', type=float, default=10.0, help='gradient penalty coefficient')
        parser.add_argument('--ncritic', type=int, default=5, help='critic iterations')
        parser.add_argument('--constrain', type=str, default='MSE', help='guidance loss type')
        parser.add_argument('--strength', type=float, default=1, help='the weight of guidance loss')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--skip', type=int, default=0, help='Define whether the guidance layer is skipped. Useful when using multiGPUs.')
        parser.add_argument('--gan_type', type=str, default='vanilla', help='wgan_gp, '
                                                                            'lsgan, '
                                                                            'vanilla, '
                                                                            're_s_gan (Relativistic Standard GAN), '
                                                                            're_avg_gan (Relativistic average Standard GAN), '
                                                                            're_avg_hinGan (Relativistic average HingeGAN),  WARNING: wgan_gp should never be used here.')
        parser.add_argument('--gan_weight', type=float, default=0.2, help='the weight of gan loss')
        parser.add_argument('--overlap', type=int, default=4, help='the overlap for center mask')

        parser.add_argument('--only_lastest', type=bool, default=True,
                            help='If True, it will save only the lastest weights and the one minimizing the L1 loss')
        parser.add_argument('--add_mask2input', type=bool, default=False,
                            help='If True, It will add the mask as a fourth dimension over input space')
        parser.add_argument('--residual_soft_attention', type=bool, default=False,
                            help='If True, It will add the mask as a fourth dimension over input space')

        self.initialized = True
        return parser

    def gather_options(self, options=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)


        self.parser = parser
        if options == None:
            return parser.parse_args()
        else:
            return parser.parse_args(options)

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, options=None):

        opt = self.gather_options(options=options)
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
