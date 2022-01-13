import glob
import random
import os
import numpy as np
from torch import nn

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, hr_shape, test=False):
        hr_height, hr_width = hr_shape
        self.test = test
        # Transforms for low resolution images and high resolution images
        if test is True:
            self.common_transfomr = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            self.common_transfomr = transforms.Compose([
                transforms.RandomApply(
                    nn.ModuleList(
                        [transforms.Resize((int(hr_height * 1.25) + 1, int(hr_height * 1.25) + 1), Image.BICUBIC),
                         transforms.RandomCrop((hr_height, hr_height))
                         ]), p=0.5),
                # transforms.RandomApply(
                #     nn.ModuleList([transforms.ColorJitter(brightness=.3, hue=.5, contrast=.2, saturation=.2), ]),
                #     p=0.25),

                transforms.RandomHorizontalFlip(p=0.5),
                # transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
                # transforms.RandomRotation(degrees=(0, 30)),
                # transforms.RandomInvert(p=0.1),
                # transforms.RandomPosterize(bits=2, p=0.35),
                # transforms.RandomEqualize(p=0.25),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.7, scale=(0.02, 0.16), ratio=(0.2, 5), value=0, inplace=False),
                transforms.RandomErasing(p=0.7, scale=(0.02, 0.16), ratio=(0.2, 5), value=0, inplace=False),
                transforms.RandomErasing(p=0.7, scale=(0.02, 0.16), ratio=(0.2, 5), value=0, inplace=False),
                transforms.Normalize(mean, std),
            ])
        self.lr_transform = transforms.Compose([transforms.Resize((hr_height // 4, hr_height // 4), Image.BICUBIC)])
        self.hr_transform = transforms.Compose([transforms.Resize((hr_height, hr_height), Image.BICUBIC), ])
        self.files = sorted(glob.glob(root + "/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img = self.common_transfomr(img)
        img_lr = self.lr_transform(img)
        img_hr = self.hr_transform(img)

        return {"lr": img_lr, "hr": img_hr}

    def __len__(self):
        return len(self.files)
