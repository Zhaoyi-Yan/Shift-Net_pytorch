#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

# For SR let fineSize=256. Then input should be 64*64.
# Using resized_paris(256*256) as training data.
class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dir_A = opt.dataroot
        self.A_paths = sorted(make_dataset(self.dir_A))
        if self.opt.offline_loading_mask:
            self.mask_folder = self.opt.training_mask_folder if self.opt.isTrain else self.opt.testing_mask_folder
            self.mask_paths = sorted(make_dataset(self.mask_folder))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')

        A = self.transform(A)


        # B is the ground-truth
        B = A.clone()

        # Then resized A to the input size.
        A = F.interpolate(A, (64, 64), mode='bilinear')
        
        
        return {'A': A, 'B': B, #'M': mask,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
