The code derives from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which results in not quite readable. I write this guide to give a brief description of our code.

# Structure of Shift-Net
## Two main scripts
`train.py`, `test.py`. You can see the main procedure of training and testing. The two scripts call funcitons from other folders.

## Options folder
Experimental options are placed in the folder of `options`. For some options which are commonly used in both training and testing stages, they are placed in `options/base_options`. These options contains `dataroot`, `which_model_netG`, `name`, `batchSize`, `gpu_ids`, etc. `options/train_options` and `options/test_options` are respectively in charge of training and testing specific options. `niter`, `niter_decay`, `print_freq`, `save_epoch_freq` are the most popular ones that you may want to alter. Always, be patient.

## Data folder
Scripts of data processing are placed in `data` folder. `data/aligned_dataset` may be the only file that you need to pay attention to. In this script, it calls function `make_dataset` inside `data/image_folder`. Then we can get all the paths
of images in `self.dir_A`. Mention, as you can see `self.dir_A = os.path.join(opt.dataroot, opt.phase)` in Line 14. Therefore, you need to place all images of groundtruth images inside the folder `opt.dataroot/opt.phase`. Masked images are generated online during training and testing, and this will be illustrated in the following sections. **All preprocessing opertions on data should be  written in `data/aligned_dataset`.** Usually, random crop, resizing, flipping as well as normalization are processed. `__getitem__` returns a dict with two keys, stores a pair images, respectively representing `input` and `groundtruth`. **Batch images are loaded in this way:**, in `train.py`, it calls `CreateDataLoader` function in `data/data_loader`. `CreateDataLoader` firstly create an instance of `CustomDatasetDataLoader`, and intializes it, and finally returns the instance. `train.py` receives the instance and calls `load_data()`. Therefore, let's step into the `load_data()` in `CustomDatasetDataLoader` class in file `custom_dataset_data_loader.py`. 
The `initialized()` is called in `CreateDataLoader`, this function adaptively selects the correct `data/aligned_dataset` or `data/single_dataset`, or `data/unaligned_dataset`(Deleted by me, as it is useless in my case). The instance of one of these classes is `dataset`, it will be passed in `torch.utils.data.DataLoader`. You can see:
```python
  self.dataloader = torch.utils.data.DataLoader(
      self.dataset,
      batch_size=opt.batchSize,
      shuffle=not opt.serial_batches,
      num_workers=int(opt.nThreads))
```
This is the code defining the `dataloader`.
You can see
```
def load_data(self):
    return self
```
Thus, it returns the whole class, making it easy for us to get the data in the `train.py`. In `train.py`, we get the `data`,
here, the `data` is dict contains `A` and `B`. Then we call `model.set_input(data)`, successfully pass the dict into `set_input()` in the file `shiftnet_model`.  More details will be illustrated below.

## Model folder
- `models` works as a selection on which model you want to adopt. Usually, just ignore it.
- `base_model` is the parent class of `shiftnet_model`. Three functions are not inherited by `shiftnet_model`. They are `save_network`, `load_network` and `update_learning_rate`. As these three functions are compatible with any models you define. Usually, you can just ignore this script.

### Shiftnet_model
`shiftnet_model` **is one of the main script that you need to spend much time reading and writing.** 
- In `initialize` function, it defines the G and D networks, two optimizers, three criterions(Reconstruction loss, GAN loss and guidance loss), and schedulers. As our model accepts masked images of three channels, `set_input` aims at filling `mean value` in the mask region on the input image, code refering to `self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)`. Of course, the mask is generated online.
- Guidance loss is implemented in this way: as the guidance loss takes the encoder feature of groundtruth
