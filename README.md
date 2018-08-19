# Shift-Net_pytorch
This repositity is our Pytorch implementation for Shift-Net, it is just for those who are interesting in our work and want to get a skeleton Pytorch implemention.

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
@article{yan2018shift,
  title={Shift-Net: Image Inpainting via Deep Feature Rearrangement},
  author={Yan, Zhaoyi and Li, Xiaoming and Li, Mu and Zuo, Wangmeng and Shan, Shiguang},
  journal={arXiv preprint arXiv:1801.09392},
  year={2018}
}
```
## New things that I want to add
- [x] Make U-Net handle with inputs of any sizes. (By resizing the size of features of decoder to fit that of the corresponding features of decoder.
- [ ] Update the code for pytorch >= 0.4.
- [x] Clean the code and delete useless comments.
- [ ] Lots of more crazy operations that help in performance.



## Acknowledgments
We benefit a lot from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
