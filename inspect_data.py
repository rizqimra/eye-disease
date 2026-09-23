import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np

from datasets import RetinaDataset

# Set up your dataset and loader
dataset = RetinaDataset(
    image_folder="dataset",  # adjust path as needed
    target_class=None  # or specify a class, e.g. "normal"
)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Get one batch
images, labels = next(iter(loader))

# Make grid
grid = make_grid(images[:64], nrow=8, normalize=True, value_range=(0, 1))

# Convert to numpy and plot
plt.figure(figsize=(8,8))
plt.imshow(np.transpose(grid.cpu().numpy(), (1, 2, 0)))
plt.axis('off')
plt.show()