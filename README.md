# Shift-Net_pytorch

This repositity is our Pytorch implementation for Shift-Net, it is just for those who are interesting in our work and want to get a skeleton Pytorch implemention.
**We DO NOT guarantee the efficiency and performance.**

The torch version [Shift-Net](https://github.com/Zhaoyi-Yan/Shift-Net) is the right choice if you would like to reproduce the fine results.


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

### pix2pix train/test
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


## Acknowledgments
We benefit a lot from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)