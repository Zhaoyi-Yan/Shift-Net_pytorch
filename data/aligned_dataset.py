#-*-coding:utf-8-*-
import os.path
import random
import torchvision.transforms as transforms
import torch
import random
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

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
        w, h = A.size

        if w < h:
            ht_1 = self.opt.loadSize * h // w
            wd_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)
        else:
            wd_1 = self.opt.loadSize * w // h
            ht_1 = self.opt.loadSize
            A = A.resize((wd_1, ht_1), Image.BICUBIC)

        A = self.transform(A)
        h = A.size(1)
        w = A.size(2)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            A = torch.flip(A, [2])

        # let B directly equals to A
        B = A.clone()
        A_flip = torch.flip(A, [2])
        B_flip = A_flip.clone()

        # Just zero the mask is fine if not offline_loading_mask.
        mask = A.clone().zero_()
        if self.opt.offline_loading_mask:
            if self.opt.isTrain:
                mask = Image.open(self.mask_paths[random.randint(0, len(self.mask_paths)-1)])
            else:
                mask = Image.open(self.mask_paths[index % len(self.mask_paths)])
            mask = mask.resize((self.opt.fineSize, self.opt.fineSize), Image.NEAREST)
            mask = transforms.ToTensor()(mask)
        
        return {'A': A, 'B': B, 'A_F': A_flip, 'B_F': B_flip, 'M': mask,
                'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'
