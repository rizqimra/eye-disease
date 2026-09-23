import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, latent_space_size, ngf, n_channel, image_size):
        super(Generator, self).__init__()
        layers = []
        
        # Always start with 4x4
        layers += [
            nn.ConvTranspose2d(latent_space_size, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True)
        ]
        # 4x4 -> 8x8
        layers += [
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        ]
        # 8x8 -> 16x16
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        ]
        # 16x16 -> 32x32
        layers += [
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        ]

        if image_size == 256:
            # 32x32 -> 64x64
            layers += [
                nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf // 2),
                nn.ReLU(True)
            ]
            # 64x64 -> 128x128
            layers += [
                nn.ConvTranspose2d(ngf // 2, ngf // 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf // 4),
                nn.ReLU(True)
            ]
            # 128x128 -> 256x256
            layers += [
                nn.ConvTranspose2d(ngf // 4, n_channel, 4, 2, 1, bias=False),
                nn.Tanh()
            ]

        elif image_size == 128:
            # 32x32 -> 64x64
            layers += [
                nn.ConvTranspose2d(ngf, ngf // 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf // 2),
                nn.ReLU(True)
            ]
            # 64x64 -> 128x128
            layers += [
                nn.ConvTranspose2d(ngf // 2, n_channel, 4, 2, 1, bias=False),
                nn.Tanh()
            ]

        elif image_size == 64:
            # 32x32 -> 64x64
            layers += [
                nn.ConvTranspose2d(ngf, n_channel, 4, 2, 1, bias=False),
                nn.Tanh()
            ]

        else:
            raise ValueError("Unsupported image_size: {}".format(image_size))
        self.generator = nn.Sequential(*layers)

    def forward(self, inp):
        return self.generator(inp)


class Discriminator(nn.Module):
    def __init__(self, n_channel, ndf, image_size):
        super(Discriminator, self).__init__()
        layers = []

        if image_size == 256:
            # 256x256 architecture
            layers += [
                nn.Conv2d(n_channel, ndf // 4, 4, 2, 1, bias=False), # 256x256 -> 128x128
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf // 4, ndf // 2, 4, 2, 1, bias=False),  # 128x128 -> 64x64
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),       # 64x64 -> 32x32
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),        # 32x32 -> 16x16
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),    # 16x16 -> 8x8
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),    # 8x8 -> 4x4
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),          # 4x4 -> 1x1
                nn.Sigmoid()
            ]

        elif image_size == 128:
            # 128x128 architecture
            layers += [
                nn.Conv2d(n_channel, ndf // 2, 4, 2, 1, bias=False), # 128x128 -> 64x64
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf // 2, ndf, 4, 2, 1, bias=False),       # 64x64 -> 32x32
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),        # 32x32 -> 16x16
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),    # 16x16 -> 8x8
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),    # 8x8 -> 4x4
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),          # 4x4 -> 1x1
                nn.Sigmoid()
            ]

        elif image_size == 64:
            # 64x64 architecture
            layers += [
                nn.Conv2d(n_channel, ndf, 4, 2, 1, bias=False),      # 64x64 -> 32x32
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),        # 32x32 -> 16x16
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),    # 16x16 -> 8x8
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),    # 8x8 -> 4x4
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),          # 4x4 -> 1x1
                nn.Sigmoid()
            ]
        else:
            raise ValueError("Unsupported image_size: {}".format(image_size))
        self.discriminator = nn.Sequential(*layers)

    def forward(self, inp):
        return self.discriminator(inp)




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


class ACGANGenerator(nn.Module):
    """Conditional generator.

    Two styles of conditioning are supported:
      * ``use_concat=False`` (default) – learn an embedding for each label and
        add it to the noise vector (previous behaviour).
      * ``use_concat=True`` – concatenate a one‑hot label vector to the noise
        along the channel dimension, matching the Keras example you posted.

    In either case the result is fed through the standard DCGAN transposed
    convolution stack defined by :class:`Generator`.
    """

    def __init__(self,
                 latent_dim: int,
                 n_classes: int,
                 ngf: int,
                 n_channel: int,
                 image_size: int,
                 use_concat: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.use_concat = use_concat

        if use_concat:
            # generator expects noise channels + one‑hot channels
            input_dim = latent_dim + n_classes
            self.net = Generator(input_dim, ngf, n_channel, image_size)
        else:
            self.label_emb = nn.Embedding(n_classes, latent_dim)
            self.net = Generator(latent_dim, ngf, n_channel, image_size)

    def forward(self, noise, labels):
        # noise: [B, latent_dim, 1, 1]
        # labels: [B] long tensor
        if self.use_concat:
            onehot = F.one_hot(labels, num_classes=self.n_classes).to(noise.dtype)
            onehot = onehot.view(labels.size(0), self.n_classes, 1, 1)
            z = torch.cat([noise, onehot], dim=1)
        else:
            emb = self.label_emb(labels).view(labels.size(0),
                                              self.latent_dim, 1, 1)
            z = noise + emb
        return self.net(z)


class ACGANDiscriminator(nn.Module):
    """Discriminator with auxiliary classifier head (two‑headed)."""

    def __init__(self, n_channel, ndf, image_size, n_classes):
        super().__init__()
        # reuse the conv‑stack from the unconditional discriminator
        base = Discriminator(n_channel, ndf, image_size).discriminator
        # drop the final sigmoid layer so we can add our own heads
        self.features = nn.Sequential(*list(base.children())[:-2])

        self.adv_head = nn.Sequential(
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.cls_head = nn.Conv2d(ndf*8, n_classes, 4, 1, 0, bias=False)

    def forward(self, x):
        feats = self.features(x)
        validity = self.adv_head(feats).view(-1)
        class_logits = self.cls_head(feats).view(x.size(0), -1)
        return validity, class_logits


class ACGANClassifier(nn.Module):
    """Standalone classifier sharing features with the discriminator.

    Useful if you want to train/evaluate the auxiliary head separately (as in
    the Keras example)."""

    def __init__(self, n_channel, ndf, image_size, n_classes):
        super().__init__()
        base = Discriminator(n_channel, ndf, image_size).discriminator
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.cls_head = nn.Conv2d(ndf*8, n_classes, 4, 1, 0, bias=False)

    def forward(self, x):
        feats = self.features(x)
        return self.cls_head(feats).view(x.size(0), -1)

