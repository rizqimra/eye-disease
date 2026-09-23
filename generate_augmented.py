import os, argparse
import torch
import torchvision.utils as vutils
from models import Generator, ACGANGenerator, ACGANDiscriminator
# optional classifier for unconditional case
from models import ACGANClassifier


def parse_args():
    p = argparse.ArgumentParser(description='Dump GAN samples with labels')
    p.add_argument('--checkpoint', required=True,
                   help='path to dcgan/acgan checkpoint')
    p.add_argument('--out_dir', default='augmented/',
                   help='where to write images')
    p.add_argument('--num_per_class', type=int, default=100,
                   help='images to generate for each label')
    p.add_argument('--num_classes', type=int, default=3,
                   help='number of labels (ACGAN only)')
    p.add_argument('--fixed_label', type=str, default=None,
                   help='If provided with an unconditional model, save all samples under this label')
    p.add_argument('--latent_dim', type=int, default=100)
    p.add_argument('--feature_mapG', type=int, default=64)
    p.add_argument('--image_size', type=int, default=64,
                   help='generator output resolution')
    p.add_argument('--use_concat', action='store_true',
                   help='if the ACGAN used label concatenation')
    p.add_argument('--device', default='cuda')
    p.add_argument('--classifier_checkpoint', default=None,
                   help='optional classifier to label unconditional GAN samples')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device,
                      weights_only=False)

    # determine whether checkpoint is conditional by inspecting generator keys
    g_state = ckpt.get('netG_state_dict', ckpt)

    # look for ACGAN-specific keys like label_emb or net.label_emb
    has_label_emb = any('label_emb' in k for k in g_state.keys())
    has_net_prefix = any(k.startswith('net.') for k in g_state.keys())

    conditional = has_label_emb or ('n_classes' in ckpt) or ('netD_state_dict' in ckpt and isinstance(ckpt.get('n_classes', None), int))

    # instantiate appropriate model
    if conditional:
        # infer n_classes from checkpoint if available
        if 'n_classes' in ckpt and isinstance(ckpt['n_classes'], int):
            n_classes = ckpt['n_classes']
        else:
            # try to infer from embedding weight
            emb_key = next((k for k in g_state.keys() if k.endswith('label_emb.weight') or k == 'label_emb.weight' or k.endswith('.label_emb.weight')), None)
            if emb_key is not None:
                n_classes = g_state[emb_key].shape[0]
            else:
                n_classes = args.num_classes

        netG = ACGANGenerator(
            latent_dim=args.latent_dim,
            n_classes=n_classes,
            ngf=args.feature_mapG,
            n_channel=3,
            image_size=args.image_size,
            use_concat=args.use_concat,
        ).to(device)
    else:
        netG = Generator(
            latent_space_size=args.latent_dim,
            ngf=args.feature_mapG,
            n_channel=3,
            image_size=args.image_size,
        ).to(device)

    # normalize state_dict keys: handle common prefixes like 'net.' or 'module.'
    def normalize_state_dict_keys(state_dict, model):
        new_sd = {}
        model_keys = list(model.state_dict().keys())
        # detect if model expects a 'net.' prefix
        model_has_net = any(k.startswith('net.') for k in model_keys)
        for k, v in state_dict.items():
            new_k = k
            # strip common DataParallel prefix
            if new_k.startswith('module.'):
                new_k = new_k[len('module.'):]
            # strip leading 'net.' if model doesn't have it
            if new_k.startswith('net.') and not model_has_net:
                new_k = new_k[len('net.'):]
            # add 'net.' if model expects it but state doesn't have it
            if (not new_k.startswith('net.')) and model_has_net:
                cand = 'net.' + new_k
                # prefer adding prefix only if that key exists in model
                if cand in model.state_dict():
                    new_k = cand
            new_sd[new_k] = v
        return new_sd

    sd = normalize_state_dict_keys(g_state, netG)
    netG.load_state_dict(sd)

    netG.eval()

    # optional classifier to label unconditional images
    if (not conditional) and args.classifier_checkpoint is not None:
        # assume classifier has same n_classes as passed arg
        clf = ACGANClassifier(
            n_channel=3,
            ndf=args.feature_mapG,
            image_size=args.image_size,
            n_classes=args.num_classes,
        ).to(device)
        clf.load_state_dict(torch.load(args.classifier_checkpoint,
                                       map_location=device))
        clf.eval()
    else:
        clf = None

    os.makedirs(args.out_dir, exist_ok=True)

    with torch.no_grad():
        if conditional:
            for cls in range(args.num_classes):
                folder = os.path.join(args.out_dir, str(cls))
                os.makedirs(folder, exist_ok=True)
                z = torch.randn(args.num_per_class,
                                args.latent_dim, 1, 1, device=device)
                labels = torch.full((args.num_per_class,), cls,
                                    dtype=torch.long, device=device)
                fake = netG(z, labels).cpu()
                for i, img in enumerate(fake):
                    vutils.save_image(img,
                                      os.path.join(folder,
                                                   f"{cls}_{i}.png"),
                                      normalize=True)
        else:
            label_name = args.fixed_label if args.fixed_label is not None else 'u'
            folder = os.path.join(args.out_dir, label_name)
            os.makedirs(folder, exist_ok=True)
            total = args.num_per_class * (args.num_classes if args.num_classes>0 else 1)
            z = torch.randn(total, args.latent_dim, 1, 1, device=device)
            fake = netG(z).cpu()
            if clf is not None:
                with torch.no_grad():
                    logits = clf(fake.to(device))
                    preds = logits.argmax(dim=1).cpu().tolist()
            else:
                preds = [label_name] * total
            for i, img in enumerate(fake):
                label = preds[i]
                vutils.save_image(img,
                                  os.path.join(folder,
                                               f"{label}_{i}.png"),
                                  normalize=True)

if __name__ == '__main__':
    main()
