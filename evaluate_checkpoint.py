import torch
import torchvision.utils as vutils
from torchvision import transforms
from datasets import RetinaDataset
from models import Generator, InceptionV3
from metrics import calculate_fretchet, calculate_inception_score
import argparse
import os
import numpy as np
import warnings
import math
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DCGAN checkpoint with metrics")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--exp_name', type=str, default='eval', help='Experiment name for output folders/files')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--latent_space_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--feature_mapG', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--target_class', type=str, default=None, help='Class to use for real images')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of fake samples to generate')
    parser.add_argument('--data_dir', type=str, default='dataset/', help='Path to dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Instantiate Generator and load weights
    netG = Generator(
        latent_space_size=args.latent_space_size,
        ngf=args.feature_mapG,
        n_channel=3,
        image_size=args.image_size
    ).to(device)
    netG.load_state_dict(checkpoint['netG_state_dict'])
    netG.eval()

    # Generate fake images
    noise = torch.randn(args.num_samples, args.latent_space_size, 1, 1, device=device)
    with torch.no_grad():
        fake_images = netG(noise).detach().cpu()

    # Save generated images
    os.makedirs(f'samples/{args.exp_name}', exist_ok=True)
    
    # Calculate nrow for save_image
    if args.num_samples == 1:
        nrow = 1
    else:
        nrow = int(math.ceil(args.num_samples ** 0.5))
    
    vutils.save_image(fake_images, f'samples/{args.exp_name}/fake_images_eval.png', normalize=True, nrow=nrow)
    print(f"Saved generated images to samples/{args.exp_name}/fake_images_eval.png")

    # Prepare real images for metrics
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    real_dataset = RetinaDataset(
        image_folder=args.data_dir,
        transform=transform,
        target_class=args.target_class
    )
    real_loader = torch.utils.data.DataLoader(
        real_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True
    )
    real_images, _ = next(iter(real_loader))
    real_images = real_images[:args.num_samples]

    # Prepare InceptionV3 for FID/IS
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx], normalize_input=False).to(device)

    # Calculate FID
    fid = calculate_fretchet(real_images.to(device), fake_images.to(device), model)
    print(f"FID score: {fid:.4f}")

    # Calculate Inception Score
    is_mean, is_std = calculate_inception_score(fake_images, cuda=True, splits=10)
    print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")


if __name__ == "__main__":
    main()