import torch
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import RetinaDataset
from models import ACGANGenerator, InceptionV3
from metrics import calculate_fretchet, calculate_inception_score
import argparse
import os
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACGAN checkpoint with per-class metrics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--exp_name', type=str, default='eval_acgan', help='Experiment name for output folders/files')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--latent_space_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--feature_mapG', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of fake samples to generate per class')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='Path to dataset')
    parser.add_argument('--use_concat', action='store_true', help='Whether generator uses concat conditioning')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Define classes
    classes_list = ['cataract', 'glaucoma', 'normal']
    class_to_idx = {cls: i for i, cls in enumerate(classes_list)}
    n_classes = len(classes_list)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Instantiate ACGANGenerator and load weights
    netG = ACGANGenerator(
        latent_dim=args.latent_space_size,
        n_classes=n_classes,
        ngf=args.feature_mapG,
        n_channel=3,
        image_size=args.image_size,
        use_concat=args.use_concat
    ).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.eval()

    # Prepare transform
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])

    # Prepare InceptionV3 for FID/IS
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], normalize_input=False).to(device)

    # Create output directory
    os.makedirs(f'samples/{args.exp_name}', exist_ok=True)

    # Evaluate per class
    for class_name in classes_list:
        class_idx = class_to_idx[class_name]
        print(f"\nEvaluating class: {class_name} (index: {class_idx})")

        # Generate fake images for this class
        labels = torch.full((args.num_samples,), class_idx, dtype=torch.long, device=device)
        noise = torch.randn(args.num_samples, args.latent_space_size, 1, 1, device=device)
        with torch.no_grad():
            fake_images = netG(noise, labels).detach().cpu()

        # Save generated images for this class
        nrow = int(math.ceil(args.num_samples ** 0.5))
        vutils.save_image(fake_images, f'samples/{args.exp_name}/fake_images_{class_name}.png', normalize=True, nrow=nrow)

        # Get real images for this class
        real_dataset = RetinaDataset(
            image_folder=args.data_dir,
            transform=transform,
            target_class=class_name
        )
        real_loader = DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
        )
        real_images, _ = next(iter(real_loader))
        real_images = real_images[:args.num_samples]

        # # Calculate FID for this class
        # fid = calculate_fretchet(real_images.to(device), fake_images.to(device), model)
        # print(f"FID score for {class_name}: {fid:.4f}")

        # # Calculate Inception Score for generated images of this class
        # is_mean, is_std = calculate_inception_score(fake_images, cuda=(args.device=='cuda'), splits=10)
        # print(f"Inception Score for {class_name}: {is_mean:.4f} ± {is_std:.4f}")

        print(f"Saved generated images for {class_name} to samples/{args.exp_name}/fake_images_{class_name}.png")

if __name__ == "__main__":
    main()