import numpy as np
import scipy as sp
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader, Subset

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from __future__ import print_function
import random

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

import matplotlib.animation as animation
from IPython.display import HTML

import argparse
import os
import time
import copy
import math
from zipfile import ZipFile


class RetinaDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = datasets.ImageFolder(root=image_folder)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard ImageNet normalization
        ])
        
        # Filter only selected classes: normal, cataract, glaucoma
        self.classes = ['cataract', 'glaucoma', 'normal']
        self.indices = [self.image_folder.class_to_idx[cls] for cls in self.classes]
        
        # Filter dataset for selected classes
        self.filtered_dataset = [(img, label) for img, label in self.image_folder if label in self.indices]

    def __len__(self):
        return len(self.filtered_dataset)

    def __getitem__(self, idx):
        image, label = self.filtered_dataset[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


config = { 'batch_size'        : 32,
           'image_size'        : 128,
           'n_channel'         : 3,
           'latent_space_size' : 100,
           'feature_mapG'      : 64,
           'feature_mapD'      : 64,
           'num_epochs'        : 50,
           'lr'                : 2e-3,
           'beta1'             : 0.5,
           'device'            : 'cuda'
 
}

# We will be working with GPU:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device : ' , device)

# Number of GPUs available. 
num_GPU = torch.cuda.device_count()
print('Number of GPU : ', num_GPU)


data_dir = 'dataset/'
data_gan = RetinaDataset(image_folder=data_dir,
                        transform = transforms.Compose([
                          # transforms.ColorJitter([0.9,0.9]),
                          # transforms.RandomGrayscale(p = 0.3),
                          # transforms.RandomAffine((-30,30)),
                          # transforms.RandomPerspective(),
                          # transforms.GaussianBlur(3),
                          # transforms.RandomHorizontalFlip(p = 0.2),
                          # transforms.RandomVerticalFlip(p = 0.2),

                          # Important parts, above can be ignored
                          transforms.Resize(int(config['image_size'] * 1.1)),
                          transforms.CenterCrop(config['image_size']),
                          transforms.ToTensor(),
                          transforms.Normalize(mean = (0.5,0.5,0.5),
                                               std  = (0.5,0.5,0.5))       
                        ]))

dataloader = torch.utils.data.DataLoader(dataset = data_gan,
                                         shuffle = True,
                                         batch_size = config['batch_size'],
                                         num_workers = 2,
                                         drop_last = True,
                                         pin_memory = True) 


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

class Generator(nn.Module):
  def __init__(self,latent_space_size = config['latent_space_size'],
               ngf = config['feature_mapG'], n_channel = config['n_channel']):
    
    super(Generator,self).__init__()
     
    self.generator = nn.Sequential(
        nn.ConvTranspose2d(latent_space_size, ngf * 8, 4, 1, 0, bias=False),  # 1x1 -> 4x4
        nn.BatchNorm2d(ngf * 8),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),  # 4x4 -> 8x8
        nn.BatchNorm2d(ngf * 4),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 8x8 -> 16x16
        nn.BatchNorm2d(ngf * 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 16x16 -> 32x32
        nn.BatchNorm2d(ngf),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  # 32x32 -> 64x64 ⭐️ added layer
        nn.BatchNorm2d(ngf // 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf // 2, n_channel, 4, 2, 1, bias=False),  # 64x64 -> 128x128
        nn.Tanh()
    )


  def forward(self,inp):
    return self.generator(inp)
  

netG = Generator().to(device)

if (device.type == 'cuda') and torch.cuda.device_count() > 1:
  netG = nn.DataParallel(netG, list(range(torch.cuda.device_count())))

netG.apply(weights_init)


# batch = next(iter(dataloader))
# print(batch.size())

batch = next(iter(dataloader))

# Unpack the images and labels
images, labels = batch

# Check the size of the images tensor
print(images.size())  # This should print the dimensions of the image batch

# Optionally, you can also print the labels
print(labels.size())  # This prints the size of the label tensor (if you want to check it)


class Discriminator(nn.Module):
  def __init__(self,n_channel = config['n_channel'], ndf = config['feature_mapD']):
    super(Discriminator,self).__init__()

    self.discriminator = nn.Sequential(
        nn.Conv2d(n_channel, ndf // 2, 4, 2, 1, bias=False),  # 128x128 -> 64x64 ⭐️ added layer
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),  # 64x64 -> 32x32
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 32x32 -> 16x16
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 16x16 -> 8x8
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 8x8 -> 4x4
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # 4x4 -> 1x1
        nn.Sigmoid()
    )

    
  @staticmethod
  def linear_block(in_ftrs,out_ftrs,p):
    return nn.Sequential(
        nn.Linear(in_ftrs,out_ftrs),
        nn.BatchNorm1d(out_ftrs),
        nn.ReLU(),
        nn.Dropout(p)
    )

    
  def forward(self,inp):
    return self.discriminator(inp)
  

netD = Discriminator().to(device)

# Handle multi-gpu if desired
# if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
#     netD = nn.DataParallel(netD, list(torch.cuda.device_count()))

netD.apply(weights_init)


criterion = nn.BCELoss().to(device)

fixed_noise = torch.randn(64, config['latent_space_size'], 1, 1, device=device)

print(fixed_noise.size())

# label smoothing
real_label = 0.9
fake_label = 0.1

optimizerD = optim.Adam(netD.parameters(),
                        lr = 0.00011, 
                        betas = (0.8,0.999))

optimizerG = optim.Adam(netG.parameters(),
                        lr = 0.0022, 
                        betas = (0.09,0.999))


