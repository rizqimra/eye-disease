# %% [markdown]
# # **DCGAN Data Augmentation**

# %%


import numpy as np
import scipy as sp
# %matplotlib inline
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
# from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
# import wandb
from losses.losses import FocalLoss, BCELoss, LDAMLoss, DiceLoss, MultiLoss
from ada import AdaptiveDiscriminatorAugmentation

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


# %%
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


# %%
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

# %%
def timeit(Func):
    def _timeStamp(*args, **kwargs):
        since = time.time()
        result = Func(*args, **kwargs)
        time_elapsed = time.time() - since

        if time_elapsed > 60:
           print('Time Consumed : {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))  
        else:        
          print('Time Consumed : ' , round((time_elapsed),4) , 's')
        return result
    return _timeStamp

# %%
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

# %%
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))
plt.show()

# %%
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

# %%
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

        nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),  # 32x32 -> 64x64
        nn.BatchNorm2d(ngf // 2),
        nn.ReLU(True),

        nn.ConvTranspose2d(ngf // 2, n_channel, 4, 2, 1, bias=False),  # 64x64 -> 128x128
        nn.Tanh()
    )


  def forward(self,inp):
    return self.generator(inp)

# %%
netG = Generator().to(device)

if (device.type == 'cuda') and torch.cuda.device_count() > 1:
  netG = nn.DataParallel(netG, list(range(torch.cuda.device_count())))

netG.apply(weights_init)


# %%
# batch = next(iter(dataloader))
# print(batch.size())

batch = next(iter(dataloader))

# Unpack the images and labels
images, labels = batch

# Check the size of the images tensor
print(images.size())  # This should print the dimensions of the image batch

# Optionally, you can also print the labels
print(labels.size())  # This prints the size of the label tensor (if you want to check it)


# %%
class Discriminator(nn.Module):
  def __init__(self,n_channel = config['n_channel'], ndf = config['feature_mapD']):
    super(Discriminator,self).__init__()

    self.discriminator = nn.Sequential(
        nn.Conv2d(n_channel, ndf // 2, 4, 2, 1, bias=False),  # 128x128 -> 64x64
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

# %%
netD = Discriminator().to(device)

# Handle multi-gpu if desired
# if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
#     netD = nn.DataParallel(netD, list(torch.cuda.device_count()))

netD.apply(weights_init)

# %%
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

# %%
class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
        Parameters
        ----------
        inp : torch.autograd.Variable
            Input tensor of shape Bx3xHxW. Values are expected to be in
            range (0, 1)
        Returns
        -------
        List of torch.autograd.Variable, corresponding to the selected output
        block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model=model.cuda()

# %%
def calculate_activation_statistics(images,model,batch_size=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# %%
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = sp.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sp.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

# %%
def calculate_fretchet(images_real,images_fake,model):
     mu_1,std_1=calculate_activation_statistics(images_real,model,cuda=True)
     mu_2,std_2=calculate_activation_statistics(images_fake,model,cuda=True)
    
     """get fretched distance"""
     fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
     return fid_value

# %%
print("Generator Parameters:",sum(p.numel() for p in netG.parameters() if p.requires_grad))
print("Discriminator Parameters:",sum(p.numel() for p in netD.parameters() if p.requires_grad))

# %%
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training Loop...")
for epoch in range(config['num_epochs']):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
       
        # add some noise to the input to discriminator
        real_cpu=0.9*real_cpu+0.1*torch.randn((real_cpu.size()), device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, config['latent_space_size'], 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        
        fake=0.9*fake+0.1*torch.randn((fake.size()), device=device)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        D_G_z2 = output.mean().item()
        
        # Calculate gradients for G
        errG.backward()
        # Update G
        optimizerG.step()
        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == config['num_epochs']-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fixed_noise = torch.randn(config['feature_mapG'], config['latent_space_size'], 1, 1, device=device)
                fake_display = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))
            
         
            
        iters += 1   
        
    G_losses.append(errG.item())
    D_losses.append(errD.item())     
    fretchet_dist=calculate_fretchet(real_cpu,fake,model) 
    
    if ((epoch+1)%5==0):    
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tFretchet_Distance: %.4f'
                      % (epoch+1, config['num_epochs'],
                         errD.item(), errG.item(),fretchet_dist))
        
        
        plt.figure(figsize=(8,8))
        plt.axis("off")
        pictures=vutils.make_grid(fake_display[torch.randint(len(fake_display), (10,))],nrow=5,padding=2, normalize=True)
        plt.imshow(np.transpose(pictures,(1,2,0)))
        plt.show()

