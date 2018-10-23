# Architecutre
<img src="architecture.png" width="1000"/> 

# Shift layer
<img src="shift_layer.png" width="800"/> 

# Shift-Net_pytorch
This repositity is our Pytorch implementation for Shift-Net, it is just for those who are interesting in our work and want to get a skeleton Pytorch implemention. The original code is https://github.com/Zhaoyi-Yan/Shift-Net.
I will upload pytorch models in months(Sorry for the delay).

## Prerequisites
- Linux or OSX.
- Python 2 or Python 3.
- CPU or NVIDIA GPU + CUDA CuDNN.
- Tested on pytorch 0.3

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
```bash
python train.py
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Test the model
```bash
python test.py
```
The test results will be saved to a html file here: `./results/`.

If you find this work useful, please cite:
```
@InProceedings{Yan_2018_Shift,
author = {Yan, Zhaoyi and Li, Xiaoming and Li, Mu and Zuo, Wangmeng and Shan, Shiguang},
title = {Shift-Net: Image Inpainting via Deep Feature Rearrangement},
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

## A bug when training with single GPU
This verison of code makes that `InnerCos` DO NOT support single gpu training. It is weried and I have not idea to
solve it now. If you train our model in a single GPU, please set `skip=1` in `options/base_options`. Otherwise,
when training with `InnerCos` working on a single GPU, please refer the code before commit [aa2382b](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch/tree/aa2382b194f36cabf40dabc6d3007cdcdc112153)

## Kindly remindier
If you find it a little hard to read the code, you may read [Guides](https://github.com/Zhaoyi-Yan/Shift-Net_pytorch/blob/master/guides.md).


## New things that I want to add
Note: I am busy with other work, will continue this work after 12.1.
- [x] Make U-Net handle with inputs of any sizes. (By resizing the size of features of decoder to fit that of the corresponding features of decoder.
- [ ] Update the code for pytorch >= 0.4.
- [x] Clean the code and delete useless comments.
- [x] Guides of our code, we hope it helps you understand our code more easily.
- [ ] Directly resize the mask to save computation.
- [ ] Guidance loss seems defined on the global region. Need make it work only in masked region.
- [ ] Add more GANs, like spectural norm and relativelistic GAN.
- [ ] Boost the efficiency of shift layer.
- [ ] Extensions of Shift-Net, which will help the performance a lot.


## Acknowledgments
We benefit a lot from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
