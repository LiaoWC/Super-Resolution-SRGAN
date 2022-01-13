import argparse
import os

import numpy as np
import math
import itertools
import sys
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader
from torch.autograd import Variable
from models import *
from datasets import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from PIL import Image

os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=2000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=200, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=480, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=480, help="high res. image width")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

hr_shape = (opt.hr_height, opt.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet()

if cuda:
    generator = generator.cuda()

# Load pretrained models
generator.load_state_dict(torch.load(f"./generator_1052.pth"))

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

dataloader = DataLoader(
    ImageDataset("./testing_lr_images/testing_lr_images", hr_shape=hr_shape,
                 test=True),
    batch_size=1,
    shuffle=False,
    num_workers=opt.n_cpu,
)

# ----------
#  Training
# ----------
gen_hrs = []
for epoch in range(opt.epoch, opt.n_epochs):
    for i, imgs in enumerate(dataloader):
        # Configure model input
        imgs_lr = Variable(imgs["lr"].type(Tensor))
        imgs_hr = Variable(imgs["hr"].type(Tensor))

        # optimizer_G.zero_grad()
        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)

        # Resize to correct size
        lr_img = Image.open(f'./datasets/testing_lr_images/testing_lr_images/{i:02}.png')
        lr_size = np.array(lr_img).shape
        hr_size = (lr_size[0] * 3, lr_size[1] * 3)
        tr = transforms.Resize(hr_size)
        tr2 = transforms.ToTensor()
        # pil_upsample = np.array(lr_img.resize((lr_size[1] * 3, lr_size[0] * 3), resample=Image.BOX))
        pil_upsample = np.array(lr_img.resize((lr_size[1] * 3, lr_size[0] * 3), resample=Image.BICUBIC))
        res = tr(gen_hr)

        # Save
        img = res[0].cpu().detach().numpy().transpose(1, 2, 0)
        img = ((img + 1.) / 2. * 255.)
        alpha = 0.
        img = ((img + 1.) / 2. * 255.) * alpha + pil_upsample * (1. - alpha)
        # img = pil_upsample
        img = Image.fromarray((img * 1).astype(np.uint8))
        # print(img, img.min(), img.max())
        # import cv2`````
        # cv2.imwrite('kkk.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        os.makedirs('results', exist_ok=True)
        # gen_hrs.append(img)
        img.save(f'results/{i:02}_pred.png')
    break
