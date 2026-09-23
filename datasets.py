import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class RetinaDataset(Dataset):
    def __init__(self, image_folder, transform=None, target_class=None):
        self.image_folder = datasets.ImageFolder(root=image_folder)
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if target_class is not None:
            class_idx = self.image_folder.class_to_idx[target_class]
            self.filtered_samples = [s for s in self.image_folder.samples if s[1] == class_idx]
        else:
            self.filtered_samples = self.image_folder.samples

    def __len__(self):
        return len(self.filtered_samples)

    def __getitem__(self, idx):
        path, label = self.filtered_samples[idx]
        image = self.image_folder.loader(path)
        if self.transform:
            image = self.transform(image)
        return image, label