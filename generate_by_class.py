import torch
import torchvision.utils as vutils
from torchvision import transforms
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
    parser = argparse.ArgumentParser(description="Generate grid of class-conditional samples from ACGAN checkpoint")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--exp_name', type=str, default='samples', help='Experiment name for output folders/files')
    parser.add_argument('--image_size', type=int, default=64, help='Image size')
    parser.add_argument('--latent_space_size', type=int, default=100, help='Latent space size')
    parser.add_argument('--feature_mapG', type=int, default=64, help='Generator feature map size')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for real data (unused)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of classes in the model')
    parser.add_argument('--n_per_class', type=int, default=10, help='Samples to generate per class')
    parser.add_argument('--use_concat', action='store_true', help='Whether the generator was trained with label concatenation')
    parser.add_argument('--class_names', type=str, default=None,
                        help='Comma-separated names for each class (in order)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # load checkpoint (disable weights_only to permit non-tensor metadata)
    # PyTorch 2.6+ defaults weights_only=True which can raise a
    # WeightsUnpickler error on older checkpoints.  We trust our own files
    # so explicitly set weights_only=False.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # instantiate conditional generator
    netG = ACGANGenerator(
        latent_dim=args.latent_space_size,
        n_classes=args.num_classes,
        ngf=args.feature_mapG,
        n_channel=3,
        image_size=args.image_size,
        use_concat=args.use_concat,
    ).to(device)
    netG.load_state_dict(ckpt['netG_state_dict'])
    netG.eval()

    # prepare output directory
    outdir = f'samples/{args.exp_name}'
    os.makedirs(outdir, exist_ok=True)

    # determine class names if none were supplied
    if args.class_names is None:
        # try to read the folder structure of the dataset
        try:
            from torchvision import datasets
            imgfolder = datasets.ImageFolder('dataset/')
            # class_to_idx is a dict mapping name->index
            inv = {v: k for k, v in imgfolder.class_to_idx.items()}
            names = [inv[i] for i in range(len(inv))]
            print(f"Inferred class order from dataset/ : {names}")
        except Exception as e:
            print("Could not infer class names from dataset folder:", e)
            names = [str(i) for i in range(args.num_classes)]
    else:
        names = args.class_names.split(',')

    # generate samples for each class
    all_samples = []
    for cls in range(args.num_classes):
        z = torch.randn(args.n_per_class, args.latent_space_size, 1, 1, device=device)
        labels = torch.full((args.n_per_class,), cls, dtype=torch.long, device=device)
        with torch.no_grad():
            fake = netG(z, labels)
        all_samples.append(fake.cpu())

    grid = vutils.make_grid(torch.cat(all_samples, 0), nrow=args.n_per_class, padding=2, normalize=True)

    # convert to PIL image so we can annotate rows
    from PIL import Image, ImageDraw, ImageFont
    pil = transforms.ToPILImage()(grid)
    draw = ImageDraw.Draw(pil)
    # choose a simple font if available
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # names variable should already be set above (inferred or supplied)

    img_h = args.image_size
    pad = 2
    for idx, name in enumerate(names[: args.num_classes]):
        y = pad + idx * (img_h + pad)
        draw.text((5, y + 5), name, fill="white", font=font)

    save_path = os.path.join(outdir, 'by_class.png')
    pil.save(save_path)
    print(f"Saved grid of {args.num_classes}x{args.n_per_class} samples to {save_path}")

    # optionally compute metrics if real data available
    # (duplicate evaluate_checkpoint code)
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize(int(args.image_size * 1.1)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
    ])
    # only compute FID/IS if dataset folder exists
    if os.path.isdir('dataset/'):
        real_dataset = RetinaDataset(
            image_folder='dataset/',
            transform=transform
        )
        real_loader = torch.utils.data.DataLoader(
            real_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=8, drop_last=True
        )
        real_images, _ = next(iter(real_loader))
        real_images = real_images[: args.num_classes * args.n_per_class]
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        model = InceptionV3([block_idx]).to(device)
        fid = calculate_fretchet(real_images.to(device), torch.cat(all_samples,0).to(device), model)
        is_mean, is_std = calculate_inception_score(torch.cat(all_samples,0), cuda=(args.device=='cuda'), model=model, splits=10)
        print(f"FID score: {fid:.4f}")
        print(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")


if __name__ == "__main__":
    main()
