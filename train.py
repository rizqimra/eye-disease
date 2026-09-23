import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import os
import sys
import argparse
import time
import json
from pathlib import Path
from tqdm import tqdm

from datasets import RetinaDataset
from models import Generator, Discriminator, InceptionV3
from metrics import calculate_fretchet, calculate_inception_score

import warnings
warnings.filterwarnings("ignore")

try:
    from config import config as external_config
except ImportError:
    external_config = None

def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train DCGAN with configurable parameters")
    parser.add_argument('--target_class', type=str, default=None, help='Class to use for training (normal, cataract, glaucoma)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--latent_space_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--feature_mapG', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--feature_mapD', type=int, default=64, help='Discriminator feature map size')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Learning rate for generator')
    parser.add_argument('--optimizer_g', type=str, default='Adam', help='Optimizer for generator')
    parser.add_argument('--beta1_g', type=float, default=0.5, help='Beta1 for generator')
    parser.add_argument('--beta2_g', type=float, default=0.999, help='Beta2 for generator')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Learning rate for discriminator')
    parser.add_argument('--optimizer_d', type=str, default='Adam', help='Optimizer for discriminator')
    parser.add_argument('--beta1_d', type=float, default=0.5, help='Beta1 for discriminator')
    parser.add_argument('--beta2_d', type=float, default=0.999, help='Beta2 for discriminator')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--exp_name', type=str, default='default', help='Experiment name for output folders/files')
    parser.add_argument('--preset', type=str, choices=['64','128','256'], default=None, help='Use a preset config: 64, 128 or 256')
    parser.add_argument('--config_file', type=str, default=None, help='Path to a JSON config file to load')
    return parser.parse_args()

def weights_init(m):
    """Custom weights initialization for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_checkpoint(state, path):
    """Save model checkpoint."""
    torch.save(state, path)

def save_report(config, G_losses, D_losses, FID_scores, IS_scores, exp_name):
    """Write a training report with all metrics and stats."""
    report_path = f'checkpoints/{exp_name}/training_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== DCGAN Training Report ===\n")
        f.write(f"Experiment: {exp_name}\n")
        f.write(f"Epochs: {config['num_epochs']}\n")
        f.write(f"Batch size: {config['batch_size']}\n")
        f.write("\n--- Generator Hyperparameters ---\n")
        f.write(f"Learning rate (G): {config.get('lr_g', 'N/A')}\n")
        f.write(f"Optimizer (G): {config.get('optimizer_g', 'N/A')}\n")
        f.write(f"Beta1 (G): {config.get('beta1_g', 'N/A')}\n")
        f.write(f"Beta2 (G): {config.get('beta2_g', 'N/A')}\n")
        f.write("\n--- Discriminator Hyperparameters ---\n")
        f.write(f"Learning rate (D): {config.get('lr_d', 'N/A')}\n")
        f.write(f"Optimizer (D): {config.get('optimizer_d', 'N/A')}\n")
        f.write(f"Beta1 (D): {config.get('beta1_d', 'N/A')}\n")
        f.write(f"Beta2 (D): {config.get('beta2_d', 'N/A')}\n")
        f.write("\n--- Metrics ---\n")
        f.write(f"Final Generator Loss: {G_losses[-1]:.4f}\n")
        f.write(f"Final Discriminator Loss: {D_losses[-1]:.4f}\n")
        f.write(f"Best FID: {min(FID_scores):.4f}\n")
        f.write(f"Final FID: {FID_scores[-1]:.4f}\n")
        f.write(f"Best Inception Score: {max(IS_scores):.4f}\n")
        f.write(f"Final Inception Score: {IS_scores[-1]:.4f}\n")
        f.write("\n--- Per-Epoch FID ---\n")
        for i, fid in enumerate(FID_scores):
            f.write(f"Epoch {i+1}: {fid:.4f}\n")
        f.write("\n--- Per-Epoch Inception Score ---\n")
        for i, iscore in enumerate(IS_scores):
            f.write(f"Epoch {i+1}: {iscore:.4f}\n")
    print(f"\nTraining report saved to {report_path}")


def train(config, dataloader, netG, netD, optimizerG, optimizerD, criterion, real_label, fake_label, model):
    """Main training loop for DCGAN."""
    img_list = []
    G_losses = []
    D_losses = []
    FID_scores = []
    IS_scores = []
    iters = 0
    best_fid = float('inf')
    exp_name = config.get('exp_name', 'default')
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)
    os.makedirs(f'samples/{exp_name}', exist_ok=True)

    print("Starting Training Loop...")
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        for i, data in enumerate(dataloader, 0):
            netD.zero_grad()
            real_cpu = data[0].to(config['device'])
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=config['device'])
            real_cpu = 0.9 * real_cpu + 0.1 * torch.randn(real_cpu.size(), device=config['device'])
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            noise = torch.randn(b_size, config['latent_space_size'], 1, 1, device=config['device'])
            fake = netG(noise)
            label.fill_(fake_label)
            fake = 0.9 * fake + 0.1 * torch.randn(fake.size(), device=config['device'])
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if (iters % 500 == 0) or ((epoch == config['num_epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fixed_noise = torch.randn(config['feature_mapG'], config['latent_space_size'], 1, 1, device=config['device'])
                    fake_display = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))
            iters += 1

            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # End of epoch: metrics
        fretchet_dist = calculate_fretchet(real_cpu, fake, model)
        FID_scores.append(fretchet_dist)
        is_mean, is_std = calculate_inception_score(fake_display, model, cuda=(config['device']=='cuda'), splits=10)
        IS_scores.append(is_mean)

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID: %.4f\tIS: %.4f'
              % (epoch+1, config['num_epochs'],
                 D_losses[-1], G_losses[-1], D_x, D_G_z1, D_G_z2, fretchet_dist, is_mean))

        # Save checkpoint and generated images every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
                'IS_scores': IS_scores,
                'FID_scores': FID_scores,
            }, f'checkpoints/{exp_name}/dcgan_epoch_{epoch+1}.pth')

            sample_path = f'samples/{exp_name}/fake_images_epoch_{epoch+1}.png'
            vutils.save_image(fake_display, sample_path, normalize=True, nrow=8)
            print(f"Saved checkpoint and sample images for epoch {epoch+1}")

        # Track and save best model by FID
        if epoch == 0 or fretchet_dist < best_fid:
            best_fid = fretchet_dist
            save_checkpoint({
                'epoch': epoch + 1,
                'netG_state_dict': netG.state_dict(),
                'netD_state_dict': netD.state_dict(),
                'optimizerG_state_dict': optimizerG.state_dict(),
                'optimizerD_state_dict': optimizerD.state_dict(),
                'G_losses': G_losses,
                'D_losses': D_losses,
                'IS_scores': IS_scores,
                'FID_scores': FID_scores,
            }, f'checkpoints/{exp_name}/best_model_fid.pth')
            print(f"Best FID improved to {best_fid:.4f}, model saved.")

    # ---- FINAL REPORT ----
    save_report(config, G_losses, D_losses, FID_scores, IS_scores, exp_name)

    # Also print summary to console
    print("\n=== Training Summary ===")
    print(f"Experiment: {exp_name}")
    print(f"Best FID: {min(FID_scores):.4f}")
    print(f"Final FID: {FID_scores[-1]:.4f}")
    print(f"Best Inception Score: {max(IS_scores):.4f}")
    print(f"Final Inception Score: {IS_scores[-1]:.4f}")

    return min(FID_scores)

def main():
    """Set up the training environment and start the training process."""
    args = parse_args()
    cli_config = vars(args)

    # Determine base config (preset JSON > provided JSON file > config.py > none)
    base_config = {}
    if args.preset is not None:
        preset_path = Path('configs') / f'dcgan_{args.preset}.json'
        if preset_path.exists():
            with open(preset_path, 'r') as f:
                base_config = json.load(f)
            print(f"Loaded preset config from {preset_path}")
        else:
            raise FileNotFoundError(f"Preset config not found: {preset_path}")
    elif args.config_file is not None:
        cfg_path = Path(args.config_file)
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                base_config = json.load(f)
            print(f"Loaded configuration from {cfg_path}")
        else:
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
    elif external_config is not None:
        base_config = external_config
        print("Loaded configuration from config.py")
    else:
        base_config = {}
        print("No preset/config.py found — falling back to command line arguments")

    # Get only CLI args that were actually set by the user
    cli_args = sys.argv[1:]
    cli_keys = {arg.lstrip('-').split('=')[0] for arg in cli_args if arg.startswith('--')}
    cli_overrides = {k: v for k, v in cli_config.items() if k in cli_keys}

    # Merge: base_config < CLI overrides (CLI wins)
    config = {**base_config, **cli_overrides}

    # Clean helper CLI-only keys
    config.pop('preset', None)
    config.pop('config_file', None)

    print("Final configuration loaded (preset/config.py/CLI overrides applied)")

    data_dir = 'dataset/'
    data_gan = RetinaDataset(
        image_folder=data_dir,
        transform=transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Resize(int(config['image_size'] * 1.1)),
            transforms.CenterCrop(config['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ]),
        target_class=config['target_class']
    )

    dataloader = DataLoader(
        dataset=data_gan,
        shuffle=True,
        batch_size=config['batch_size'],
        num_workers=8,
        drop_last=True,
        pin_memory=True
    )

    device = torch.device(config['device'])

    netG = Generator(
        latent_space_size=config['latent_space_size'],
        ngf=config['feature_mapG'],
        n_channel=3,
        image_size=config['image_size']
    ).to(device)
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
    netG.apply(weights_init)

    netD = Discriminator(
        n_channel=3,
        ndf=config['feature_mapD'],
        image_size=config['image_size']
    ).to(device)
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
    netD.apply(weights_init)

    criterion = nn.BCELoss().to(device)
    real_label = 0.9
    fake_label = 0.1

    # Use separate optimizers and betas for G and D if provided
    if config.get('optimizer_g'):
        optimizer_g_type = config['optimizer_g'].lower()
        lr_g = config['lr_g']
        beta1_g = config['beta1_g']
        beta2_g = config['beta2_g']
    else:
        optimizer_g_type = config['optimizer'].lower()
        lr_g = config['lr']
        beta1_g = config['beta1']
        beta2_g = config['beta2']

    if config.get('optimizer_d'):
        optimizer_d_type = config['optimizer_d'].lower()
        lr_d = config['lr_d']
        beta1_d = config['beta1_d']
        beta2_d = config['beta2_d']
    else:
        optimizer_d_type = config['optimizer'].lower()
        lr_d = config['lr']
        beta1_d = config['beta1']
        beta2_d = config['beta2']

    if optimizer_g_type == 'adam':
        optimizerG = optim.Adam(netG.parameters(), lr=lr_g, betas=(beta1_g, beta2_g))
    elif optimizer_g_type == 'sgd':
        optimizerG = optim.SGD(netG.parameters(), lr=lr_g, momentum=beta1_g)
    else:
        raise ValueError(f"Unknown optimizer for generator: {optimizer_g_type}")

    if optimizer_d_type == 'adam':
        optimizerD = optim.Adam(netD.parameters(), lr=lr_d, betas=(beta1_d, beta2_d))
    elif optimizer_d_type == 'sgd':
        optimizerD = optim.SGD(netD.parameters(), lr=lr_d, momentum=beta1_d)
    else:
        raise ValueError(f"Unknown optimizer for discriminator: {optimizer_d_type}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)

    start_time = time.time()
    best_fid = train(config, dataloader, netG, netD, optimizerG, optimizerD, criterion, real_label, fake_label, model)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training complete! Total time: {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
    return best_fid

if __name__ == "__main__":
    main()