import torch
import torch.nn as nn
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
from models import ACGANGenerator, ACGANDiscriminator, InceptionV3
from metrics import calculate_fretchet, calculate_inception_score

import warnings
warnings.filterwarnings("ignore")

try:
    from config import config as external_config
except ImportError:
    external_config = None


def parse_args():
    parser = argparse.ArgumentParser(description="Train ACGAN-DCGAN with configurable parameters")
    parser.add_argument('--target_class', type=str, default=None,
                        help='Optional single class to train on (normal, cataract, glaucoma)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--latent_space_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--feature_mapG', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--feature_mapD', type=int, default=64, help='Discriminator feature map size')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Total number of classes in the dataset')
    parser.add_argument('--use_concat', action='store_true',
                        help='Concatenate one-hot label to noise instead of embedding sum')
    parser.add_argument('--lr_g', type=float, default=2e-4, help='Learning rate for generator')
    parser.add_argument('--optimizer_g', type=str, default='Adam', help='Optimizer for generator')
    parser.add_argument('--beta1_g', type=float, default=0.5, help='Beta1 for generator')
    parser.add_argument('--beta2_g', type=float, default=0.999, help='Beta2 for generator')
    parser.add_argument('--lr_d', type=float, default=2e-4, help='Learning rate for discriminator')
    parser.add_argument('--optimizer_d', type=str, default='Adam', help='Optimizer for discriminator')
    parser.add_argument('--beta1_d', type=float, default=0.5, help='Beta1 for discriminator')
    parser.add_argument('--beta2_d', type=float, default=0.999, help='Beta2 for discriminator')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--exp_name', type=str, default='acgan', help='Experiment name for output folders/files')
    parser.add_argument('--preset', type=str, choices=['64','128','256'], default=None, help='Use a preset config: 64, 128 or 256')
    parser.add_argument('--config_file', type=str, default=None, help='Path to a JSON config file to load')
    return parser.parse_args()


def weights_init(m):
    """Custom initialization used by DCGAN and ACGAN scripts."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def train(config, dataloader, netG, netD, optimizerG, optimizerD,
          adv_criterion, cls_criterion, real_label, fake_label, model):
    """Training loop for ACGAN-DCGAN."""
    img_list = []
    G_losses = []
    D_losses = []
    FID_scores = []
    IS_scores = []
    cls_acc_real = []          # discriminator accuracy on real images
    cls_acc_fake = []          # discriminator accuracy on generated images
    iters = 0
    best_fid = float('inf')
    exp_name = config.get('exp_name', 'acgan')
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)
    os.makedirs(f'samples/{exp_name}', exist_ok=True)

    print("Starting Training Loop...")
    for epoch in tqdm(range(config['num_epochs']), desc="Epochs"):
        # counters for classification accuracy within epoch
        total_real = 0
        total_fake = 0
        correct_real = 0
        correct_fake = 0
        for i, data in enumerate(dataloader, 0):
            real_cpu, real_labels = data[0].to(config['device']), data[1].to(config['device'])
            b_size = real_cpu.size(0)

            # -- train D on real
            netD.zero_grad()
            label_real = torch.full((b_size,), real_label, device=config['device'])
            output, logits_real = netD(real_cpu)
            # accuracy on real batch
            preds_real = logits_real.argmax(dim=1)
            correct_real += (preds_real == real_labels).sum().item()
            total_real += b_size

            errD_real = adv_criterion(output, label_real)
            cls_real = cls_criterion(logits_real, real_labels)
            errD_real_total = errD_real + cls_real
            D_x = output.mean().item()

            # -- train D on fake
            noise = torch.randn(b_size, config['latent_space_size'], 1, 1, device=config['device'])
            rand_labels = torch.randint(0, 3, (b_size,), device=config['device'])
            fake = netG(noise, rand_labels)
            label_fake = torch.full((b_size,), fake_label, device=config['device'])
            output, logits_fake = netD(fake.detach())
            # accuracy on fake batch (discriminator classification)
            preds_fake = logits_fake.argmax(dim=1)
            correct_fake += (preds_fake == rand_labels).sum().item()
            total_fake += b_size

            errD_fake = adv_criterion(output, label_fake)
            cls_fake = cls_criterion(logits_fake, rand_labels)
            errD_fake_total = errD_fake + cls_fake
            # combine both real and fake losses before backprop
            errD = errD_real_total + errD_fake_total
            errD.backward()
            D_G_z1 = output.mean().item()
            optimizerD.step()

            # -- train G
            netG.zero_grad()
            label_gen = torch.full((b_size,), real_label, device=config['device'])
            output, logits_fake = netD(fake)
            g_adv = adv_criterion(output, label_gen)
            g_cls = cls_criterion(logits_fake, rand_labels)
            errG = g_adv + g_cls
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # record
            if (iters % 500 == 0) or ((epoch == config['num_epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fixed_noise = torch.randn(config['feature_mapG'], config['latent_space_size'], 1, 1, device=config['device'])
                    fixed_labels = torch.arange(config['feature_mapG'], device=config['device']) % 3
                    fake_display = netG(fixed_noise, fixed_labels).detach().cpu()
                img_list.append(vutils.make_grid(fake_display, padding=2, normalize=True))

            iters += 1
            G_losses.append(errG.item())
            D_losses.append(errD.item())

        # epoch metrics
        fretchet_dist = calculate_fretchet(real_cpu, fake, model)
        FID_scores.append(fretchet_dist)
        is_mean, is_std = calculate_inception_score(fake_display, cuda=(config['device']=='cuda'), splits=10)
        IS_scores.append(is_mean)
        # compute classifier accuracy for epoch
        acc_real = correct_real / total_real if total_real > 0 else 0.0
        acc_fake = correct_fake / total_fake if total_fake > 0 else 0.0
        cls_acc_real.append(acc_real)
        cls_acc_fake.append(acc_fake)

        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f\tFID: %.4f\tIS: %.4f\tAcc_real: %.3f\tAcc_fake: %.3f'
              % (epoch+1, config['num_epochs'],
                 D_losses[-1], G_losses[-1], D_x, D_G_z1, D_G_z2, fretchet_dist, is_mean,
                 acc_real, acc_fake))

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
                'cls_acc_real': cls_acc_real,
                'cls_acc_fake': cls_acc_fake,
            }, f'checkpoints/{exp_name}/acgan_epoch_{epoch+1}.pth')
            sample_path = f'samples/{exp_name}/fake_images_epoch_{epoch+1}.png'
            vutils.save_image(fake_display, sample_path, normalize=True, nrow=8)
            print(f"Saved checkpoint and sample images for epoch {epoch+1}")

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

    save_report(config, G_losses, D_losses, FID_scores, IS_scores, cls_acc_real, cls_acc_fake, exp_name)
    print("\n=== Training Summary ===")
    print(f"Experiment: {exp_name}")
    print(f"Best FID: {min(FID_scores):.4f}")
    print(f"Final FID: {FID_scores[-1]:.4f}")
    print(f"Best Inception Score: {max(IS_scores):.4f}")
    print(f"Final Inception Score: {IS_scores[-1]:.4f}")
    return min(FID_scores)


def save_checkpoint(state, path):
    torch.save(state, path)


def save_report(config, G_losses, D_losses, FID_scores, IS_scores, cls_acc_real, cls_acc_fake, exp_name):
    report_path = f'checkpoints/{exp_name}/training_report.txt'
    with open(report_path, 'w') as f:
        f.write("=== ACGAN-DCGAN Training Report ===\n")
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
        f.write(f"Best classifier acc (real): {max(cls_acc_real):.4f}\n")
        f.write(f"Final classifier acc (real): {cls_acc_real[-1]:.4f}\n")
        f.write(f"Best classifier acc (fake): {max(cls_acc_fake):.4f}\n")
        f.write(f"Final classifier acc (fake): {cls_acc_fake[-1]:.4f}\n")
        f.write("\n--- Per-Epoch FID ---\n")
        for i, fid in enumerate(FID_scores):
            f.write(f"Epoch {i+1}: {fid:.4f}\n")
        f.write("\n--- Per-Epoch Inception Score ---\n")
        for i, iscore in enumerate(IS_scores):
            f.write(f"Epoch {i+1}: {iscore:.4f}\n")
        f.write("\n--- Per-Epoch Classifier Accuracy (real) ---\n")
        for i, acc in enumerate(cls_acc_real):
            f.write(f"Epoch {i+1}: {acc:.4f}\n")
        f.write("\n--- Per-Epoch Classifier Accuracy (fake) ---\n")
        for i, acc in enumerate(cls_acc_fake):
            f.write(f"Epoch {i+1}: {acc:.4f}\n")
    print(f"\nTraining report saved to {report_path}")


def main():
    args = parse_args()
    cli_config = vars(args)

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

    cli_args = sys.argv[1:]
    cli_keys = {arg.lstrip('-').split('=')[0] for arg in cli_args if arg.startswith('--')}
    cli_overrides = {k: v for k, v in cli_config.items() if k in cli_keys}
    config = {**base_config, **cli_overrides}
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
        target_class=config.get('target_class')
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

    netG = ACGANGenerator(
        latent_dim=config['latent_space_size'],
        n_classes=3,
        ngf=config['feature_mapG'],
        n_channel=3,
        image_size=config['image_size'],
        use_concat=config.get('use_concat', False)
    ).to(device)
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        netG = nn.DataParallel(netG)
    netG.apply(weights_init)

    netD = ACGANDiscriminator(
        n_channel=3,
        ndf=config['feature_mapD'],
        image_size=config['image_size'],
        n_classes=3
    ).to(device)
    if (device.type == 'cuda') and torch.cuda.device_count() > 1:
        netD = nn.DataParallel(netD)
    netD.apply(weights_init)

    adv_criterion = nn.BCELoss().to(device)
    cls_criterion = nn.CrossEntropyLoss().to(device)
    real_label = 0.9
    fake_label = 0.1

    # optimizer setup (same as DCGAN script)
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
    model = InceptionV3([block_idx], normalize_input=False).to(device)

    start_time = time.time()
    best_fid = train(config, dataloader, netG, netD, optimizerG, optimizerD,
                     adv_criterion, cls_criterion, real_label, fake_label, model)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training complete! Total time: {elapsed/60:.2f} minutes ({elapsed:.2f} seconds)")
    return best_fid


if __name__ == "__main__":
    main()
