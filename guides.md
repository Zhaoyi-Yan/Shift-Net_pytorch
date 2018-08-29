The code derives from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), which results in its some unreadability. I write this guide to give a brief description of our code. I hope this can help you understand our code more easily.

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
`shiftnet_model` **is one of the main scripts that you need to spend much time reading and writing.** 
- In `initialize` function, it defines the G and D networks, two optimizers, three criterions(Reconstruction loss, GAN loss and guidance loss), and schedulers. As our model accepts masked images of three channels, 
- `set_input` get the `input` receving from `train.py` and aims at filling `mean value` in the mask region on the input images, code refering to `self.input_A.narrow(1,1,1).masked_fill_(self.mask_global, 2*104.0/255.0 - 1.0)`. Of course, the mask is generated online. By now, you will know, why we just let `B` the same context with `A`. `B` is the groundtruth image, and `A` is the context of `B` with mask region.

### How guidance loss is implemented
The class of `guidance loss` is defined as `models/InnerCos.py`. As it is actually a `L2` constrain with novel target:
the encoder feature of groundtruth(B). This means, for the same input, the target changes as the parameters of network vary
in different iterations. Therefore, in each iteration, we firstly call `set_input`. This function also sets the `resized mask`(mask in shift layer) in `InnerCos`, which is essentially for shift operation. Then we call `set_gt_latent`, 
```python
    def set_gt_latent(self):
        self.netG.forward(Variable(self.input_B, requires_grad=False)) # input ground truth
        gt_latent = self.ng_innerCos_list[0].get_target()  # then get gt_latent(should be variable require_grad=False)
        self.ng_innerCos_list[0].set_target(gt_latent)
```
it acts as a role, providing the latent of encoder feature of groundtruth. Thus the dynamic `target` of `InnerCos` is obtained.
In the second iteration, we pass the `A` into the model as usual.
In `InnerCos.py`, we can see that this class mainly computes the loss of input and target, proving the gradient of guidance
loss.

### Where is the model constructed
It is defined in `models/networks.py`. The construction of `Unet` is interesting. `UnetSkipConnectionBlock` works as base component of Unet. Unet is constructed firstly from the innermost block, then we warp it with a new layer, it returns a new block on which we can continue warpping it with an instance of class `UnetSkipConnectionBlock`. When it reaches the outermost border, we can see:
```python
def forward(self, x):
    if self.outermost:  # if it is the outermost, directly pass the input in.
        return self.model(x)
    else:
        x_latter = self.model(x)
        _, _, h, w = x.size()

        if h != x_latter.size(2) or w != x_latter.size(3):
            x_latter = F.upsample(x_latter, (h, w), mode='bilinear')
        return torch.cat([x_latter, x], 1)  # cat in the C channel
```
It is easy to understand.
As for our shift model, we need to add layer `Guidance loss layer` and `shift layer` inside the model.
`UnetSkipConnectionShiftTripleBlock` is based on `UnetSkipConnectionBlock`. It demonstrates distinctiveness in
```python
      # shift triple differs in here. It is `*3` not `*2`.
      upconv = nn.ConvTranspose2d(inner_nc * 3, outer_nc,
                                  kernel_size=4, stride=2,
                                  padding=1)
      down = [downrelu, downconv, downnorm]
      # shift should be placed after uprelu
      # Note: innerCos are placed before shift. So need to add the latent gredient to
      # to former part.
      up = [uprelu, innerCos, shift, upconv, upnorm]
```
As the network is defined in this way, it is not quite elegant to directly get specific layer of the model. Thus, we
pass in `innerCos_list` and `shift_list` as parameters. We build `guidance loss layer` and `shift layer` respectively by `InnerCos` and `InnerShiftTriple` in `UnetGeneratorShiftTriple`. `UnetGeneratorShiftTriple` is called in `define_G`. `define_G` decides which network architecture you will choose for our generative model. It returns `netG` as well as extra
two layers `innerCos_list` and `shift_list`. And finally, we can get these two special layers in `shiftnet_model`. So we
can construct the target of guidance loss layer with `set_gt_latent`. If you are still a bit confused, please refer to 
the above section **`How guidance loss is implemented`**.

### How shift is implemented
`InnerShiftTriple` and `InnerShiftTripleFunction` are the two main scripts. `InnerShiftTriple` get the features in `forward(self, input)`. As `input` here is the data consists of the concatenation of encoder feature with corresponding decoder feature. We split out `former_all` and `latter_all` in `forward` in `InnerShiftTripleFunction`. The known region of `latter_all` will be used to fill mask region of `former_all`. As for more details, it is a little bit complex, so please refer to the code.
