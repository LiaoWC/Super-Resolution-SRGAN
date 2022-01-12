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


def cutblur(im1, im2, prob=1.0):
    if im1.size() != im2.size():
        raise ValueError(f"im1 and im2 have to be the same resolution. ({im1.size()} and {im2.size()})")

    # cut_ratio = min(np.random.randn() * 0.01 + alpha, 0.999)
    cut_ratio = np.random.random() * 0.3 + 0.3

    h, w = im2.size(1), im2.size(2)
    ch, cw = np.int(h * cut_ratio), np.int(w * cut_ratio)
    cy = np.random.randint(0, h - ch + 1)
    cx = np.random.randint(0, w - cw + 1)

    # apply CutBlur to inside or outside
    if np.random.random() > 0.5:
        im2[..., cy:cy + ch, cx:cx + cw] = im1[..., cy:cy + ch, cx:cx + cw]
    else:
        im2_aug = im1.clone()
        im2_aug[..., cy:cy + ch, cx:cx + cw] = im2[..., cy:cy + ch, cx:cx + cw]
        im2 = im2_aug

    return im1, im2


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

                transforms.ColorJitter(brightness=.4, hue=.5, contrast=.25, saturation=.25),
                # transforms.RandomPerspective(distortion_scale=0.25, p=0.25),
                # transforms.RandomRotation(degrees=(0, 30)),
                # transforms.RandomInvert(p=0.1),
                # transforms.RandomPosterize(bits=2, p=0.35),
                transforms.RandomEqualize(p=0.25),
                transforms.ToTensor(),
                # transforms.RandomErasing(p=0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
                # transforms.RandomErasing(p=0.7, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False),
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

        if self.test is not True:
            imgs_lr_ups = self.hr_transform(img_lr)
            _, img_hr = cutblur(imgs_lr_ups, img_hr)
            img_lr = self.lr_transform(img_hr)
        return {"lr": img_lr, "hr": img_hr}
        # return {"lr": imgs_lr_ups, "hr": img_hr}

    def __len__(self):
        return len(self.files)
