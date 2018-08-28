The code derives from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which results in not quite readable. I write this guide to give a brief description of our code.

# Structure of Shift-Net
## Two main scripts
`train.py`, `test.py`. You can see the main procedure of training and testing. The two scripts call funcitons from other folders.

## Options folder
Experimental options are placed in the folder of `options`. For some options which are commonly used in both training and testing stages, they are placed in `options/base_options`. These options contains `dataroot`, `which_model_netG`, `name`, `batchSize`, `gpu_ids`, etc. `options/train_options` and `options/test_options` are respectively in charge of training and testing specific options. `niter`, `niter_decay`, `print_freq`, `save_epoch_freq` are the most popular ones that you may want to alter. Always, be patient.

## Data folder
Scripts of data processing are placed in `data` folder. `data/aligned_dataset` may be the only folder that you need to pay attention to. In this script, it calls function `make_dataset` inside `data/image_folder`. Then we can get all the paths
of images in `self.dir_A`. Mention, as you can see `self.dir_A = os.path.join(opt.dataroot, opt.phase)` in Line 14. Therefore, you need to place all images of groundtruth images inside the folder `opt.dataroot/opt.phase`. Masked images are generated online during training and testing, and this will be illustrated in the following sections. **All preprocessing opertions on data should be  written in `data/aligned_dataset`.** Usually, random crop, resizing, flipping as well as normalization are processed.

## Model folder
