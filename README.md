# This branch aims to repalce the inner blocks with resnet_blocks.
unet_resBlock_my_shift_1: Add 5 resblocks to replace the inner layers.
unet_resBlock_my_shift_2
unet_resBlock_my_shift_3
unet_resBlock_my_shift_4

# Architecutre
<img src="architecture.png" width="1000"/> 

# Shift layer
<img src="shift_layer.png" width="800"/> 

# Why no pretrain models ?
It is because the code is still in active development, making pretrained models broken from time to time.

Just pull the latest code and train by following the instructions.

## Prerequisites
- Linux or Windows.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.
- Tested on pytorch >= 1.0

## Getting Started
### Installation
- Install PyTorch and dependencies from http://pytorch.org/
- Install python libraries [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate).

```bash
pip install visdom
pip install dominate
```
- Clone this repo:
```bash
git clone https://github.com/Zhaoyi-Yan/Shift-Net_pytorch
cd Shift-Net_pytorch

```

### tain and test
- Download your own inpainting datasets.

- Train a model:
Please read this paragraph carefully before running the code.

By now, 5 kinds of shift-nets are proposed.

Usually, we train and test a model with `center` mask.

**Mention: For now, the best performance of `center mask inpainiting` can be achieved if you train this line:**.

```bash
python train.py --batchsize=1 --use_spectral_norm_D=1 --which_model_netD='basic' --mask_type='center'
```

**DO NOT** set batchsize larger than 1 for `square` mask training, the performance degrades a lot(I don't know why...)

For `random mask`(`mask_sub_type` is NOT `rect`), the batchsize can be larger than 1 without hurt of performance.

For training random mask, you need to train the model by setting
`mask_type='random'` and also `mask_sub_type='rect'` or `mask_sub_type='island'`.


For `navie shift-net`:
```bash
python train.py --which_model_netG='unet_shift_triple' --model='shiftnet' --shift_sz=1 --mask_thred=1
```

**These 4 models are just experimental**

For `res navie shift-net`:
```bash
python train.py --which_model_netG='res_unet_shift_triple' --model='res_shiftnet' --shift_sz=1 --mask_thred=1
```

For `pixel soft shift-net`:
```bash
python train.py --which_model_netG='soft_unet_shift_triple' --model='soft_shiftnet' --shift_sz=1 --mask_thred=1
```

For `patch soft shift-net`:
```bash
python train.py --which_model_netG='patch_soft_unet_shift_triple' --model='patch_soft_shiftnet' --shift_sz=3 --mask_thred=4
```

For `res patch soft shift-net`:
```bash
python train.py --which_model_netG='res_patch_soft_unet_shift_triple' --model='res_patch_soft_shiftnet' --shift_sz=3 --mask_thred=4
```
DO NOT change the shift_sz and mask_thred. Otherwise, it errors with a high probability.

For `patch soft shift-net` or `res patch soft shift-net`. You may set `fuse=1` to see whether it delivers better results.


- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097. The checkpoints will be saved in `./log` by default.

- Test the model

**Keep the same settings as those during training phase to avoid errors or bad performance**

For example, if you train `patch soft shift-net`, then the following testing command is appropriate.
```bash
python test.py --fuse=1/0 --which_model_netG='patch_soft_unet_shift_triple' --model='patch_soft_shiftnet' --shift_sz=3 --mask_thred=4 
```
The test results will be saved to a html file here: `./results/`.

### Testing models on given masks
You should firstly generate masks by running `generate_masks.py`, we assume that only `mask_type=random`, then it makes sense to generate masks by yourself.
Make sure that you should **keep the same setting with what you train the model** when generating masks.
It means that you when you train the model with `mask_type='random'` and `mask_sub_type='island'`, then keep the same setting when generating masks using this `generate_masks.py`.
It generates masks with the names by adding a suffix of `_mask.png` to corresponding names of testing images.
Then set `offline_testing=1` when testing, the program will read corresponding masks when testing.

## Performance degrades when batchsize > 1
-^_^, I trying to solve it...
A very strange thing is that if the `batchSize>1`, then the performance degrades.
I wonder whether it is due to some incompatibility of IN with UNet.
I will try to solve this problem.

## Kindly remindier
If you find it a little hard to read the code, you may read [Guides](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch/blob/master/guides.md).


## New things that I want to add
- [x] Make U-Net handle with inputs of any sizes. (By resizing the size of features of decoder to fit that of the corresponding features of decoder.
- [x] Update the code for pytorch >= 0.4.
- [x] Clean the code and delete useless comments.
- [x] Guides of our code, we hope it helps you understand our code more easily.
- [x] Add more GANs, like spectural norm and relativelistic GAN.
- [x] Boost the efficiency of shift layer.
- [x] Directly resize the global_mask to get the mask in feature space.
- [x] Visualization of flow. It is still experimental now.
- [x] Extensions of Shift-Net. Still active in absorbing new features.
- [x] Fix bug in guidance loss when adopting it in multi-gpu.
- [x] Add composit L1 loss between mask loss and non-mask loss
- [x] Finish optimizing soft-shift
- [ ] Fix performance degradance when batchsize is larger than 1.
- [ ] Add random batch of masks

## Citation
If you find this work useful or gives you some insights, please cite:
```
@InProceedings{Yan_2018_Shift,
author = {Yan, Zhaoyi and Li, Xiaoming and Li, Mu and Zuo, Wangmeng and Shan, Shiguang},
title = {Shift-Net: Image Inpainting via Deep Feature Rearrangement},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## Acknowledgments
We benefit a lot from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
