#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class AlignedDatasetResized(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = opt.dataroot # More Flexible for users

        self.A_paths = sorted(make_dataset(self.dir_A))

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        A = Image.open(A_path).convert('RGB')

        A = A.resize ((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)

        A = self.transform(A)

        #if (not self.opt.no_flip) and random.random() < 0.5:
        #    idx = [i for i in range(A.size(2) - 1, -1, -1)] # size(2)-1, size(2)-2, ... , 0
        #    idx = torch.LongTensor(idx)
        #    A = A.index_select(2, idx)

        # let B directly equals A
        B = A.clone()
        return {'A': A, 'B': B,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDatasetResized'
