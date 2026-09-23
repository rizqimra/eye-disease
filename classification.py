import os, time, argparse, random
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from datasets import RetinaDataset
# stratified splitting
from sklearn.model_selection import train_test_split
# evaluation metrics
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

try:
    import wandb
except ImportError:
    wandb = None

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',   default='dataset/')
    p.add_argument('--batch_size',type=int, default=32)
    p.add_argument('--image_size',type=int, default=224)
    p.add_argument('--num_epochs',type=int, default=20)
    p.add_argument('--lr',        type=float, default=1e-3)
    p.add_argument('--momentum',  type=float, default=0.9)
    p.add_argument('--device',    type=str, default='cuda')
    p.add_argument('--exp_name',  type=str, default='resnet50')
    p.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    p.add_argument('--val_pct',   type=float, default=0.2,
                        help='Fraction of data to use for validation')
    p.add_argument('--test_pct',  type=float, default=0.2,
                        help='Fraction of data to use for test')
    p.add_argument('--aug_dir',   type=str, default=None,
                        help='Optional folder containing GAN-augmented images')
    p.add_argument('--use_weighted_loss', action='store_true',
                        help='Use class-weighted cross entropy to compensate imbalance')
    p.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases experiment tracking')
    p.add_argument('--wandb_project', type=str, default='eye-disease-classification',
                        help='W&B project name')
    p.add_argument('--wandb_entity', type=str, default=None,
                        help='W&B entity (team/user)')
    p.add_argument('--wandb_run_name', type=str, default=None,
                        help='W&B run name')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Reproducibility: fix all random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Initialize Weights & Biases (optional)
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            config={
                **vars(args),
                "device": str(device),
            },
        )
    elif args.wandb:
        print("[WARNING] wandb is not installed. Install with `pip install wandb` to enable logging.")

    # transforms for ResNet
    train_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((args.image_size, args.image_size)),
        # Add any train-time augmentations here (only applied to training)
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])

    eval_tf = transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485,0.456,0.406),
                             std=(0.229,0.224,0.225))
    ])

    # Build datasets separately for train + eval so evaluation does not receive augmentations
    real_train_ds = RetinaDataset(image_folder=args.data_dir, transform=train_tf)
    real_eval_ds = RetinaDataset(image_folder=args.data_dir, transform=eval_tf)

    n_classes = len(real_eval_ds.image_folder.classes)  # e.g. 3

    # Optional GAN augmentation (only used in training)
    if args.aug_dir is not None:
        aug_train_ds = RetinaDataset(image_folder=args.aug_dir, transform=train_tf)

    # Split real data to train/val/test (val/test come only from real data)
    labels = np.array([lbl for _, lbl in real_eval_ds.filtered_samples])
    real_indices = np.arange(len(real_eval_ds))

    temp_frac = args.val_pct + args.test_pct
    if temp_frac > 0:
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            real_indices, labels, stratify=labels, test_size=temp_frac, random_state=42)

        val_fraction_of_temp = args.val_pct / temp_frac
        val_idx, test_idx, y_val, y_test = train_test_split(
            temp_idx, y_temp, stratify=y_temp,
            test_size=1-val_fraction_of_temp, random_state=42)
    else:
        train_idx, val_idx, test_idx = real_indices, np.array([], dtype=int), np.array([], dtype=int)

    # training set includes augmented images (if provided)
    train_ds = Subset(real_train_ds, train_idx)
    if args.aug_dir is not None:
        train_ds = torch.utils.data.ConcatDataset([train_ds, aug_train_ds])

    val_ds   = Subset(real_eval_ds, val_idx)
    test_ds  = Subset(real_eval_ds, test_idx)

    # build labels for computing class weights (train set only)
    train_labels = labels[train_idx]
    if args.aug_dir is not None:
        aug_labels = np.array([lbl for _, lbl in aug_train_ds.filtered_samples])
        train_labels = np.concatenate([train_labels, aug_labels])

    # dataloaders; optionally use weighted sampler for training
    # weights are computed from the flattened training labels (including augmentations)
    if args.use_weighted_loss:
        class_counts = np.bincount(train_labels)
        class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    else:
        class_weights = None

    # Use a generator for reproducible DataLoader shuffling.
    g = torch.Generator()
    g.manual_seed(args.seed)

    def _worker_init_fn(worker_id):
        seed = args.seed + worker_id
        np.random.seed(seed)
        random.seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        generator=g,
        worker_init_fn=_worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )

    n_train = len(train_ds)  # needed for loss/accuracy computation
    n_val   = len(val_ds)

    # model
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model = model.to(device)

    if args.wandb and wandb is not None:
        wandb.watch(model, log="all", log_freq=100)

    criterion = nn.CrossEntropyLoss(weight=class_weights) if class_weights is not None else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0
    os.makedirs(f'checkpoints/{args.exp_name}', exist_ok=True)

    # history for plotting later
    train_losses = []
    train_accs  = []
    val_losses   = []
    val_accs     = []
    val_precisions = []
    val_recalls = []
    val_f1s = []
    val_aucs = []

    for epoch in range(1, args.num_epochs+1):
        # train
        model.train()
        running_loss = 0.0
        running_corrects = 0
        train_y_true = []
        train_y_pred = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device); labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds==labels.data)

            # store for metrics
            train_y_true.extend(labels.cpu().numpy())
            train_y_pred.extend(preds.cpu().numpy())

        epoch_loss = running_loss / n_train
        epoch_acc  = running_corrects.double() / n_train

        scheduler.step()

        # validate
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        y_true = []
        y_pred = []
        y_probs = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                # record for metrics
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(nn.functional.softmax(outputs, dim=1).cpu().numpy())
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds==labels.data)
        val_loss = val_loss / (len(val_ds) if len(val_ds)>0 else 1)
        val_acc  = val_corrects.double() / (len(val_ds) if len(val_ds)>0 else 1)

        # compute extra metrics
        if len(y_true) > 0:
            val_precision = precision_score(y_true, y_pred, average='macro')
            val_recall = recall_score(y_true, y_pred, average='macro')
            val_f1 = f1_score(y_true, y_pred, average='macro')
            try:
                if n_classes == 2:
                    # for binary, use probability of positive class
                    val_auc = roc_auc_score(y_true, [p[1] for p in y_probs])
                else:
                    val_auc = roc_auc_score(y_true, y_probs, multi_class='ovo', average='macro')
            except Exception:
                val_auc = None
        else:
            val_precision = val_recall = val_f1 = 0.0
            val_auc = None

        # store metrics for plotting
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)
        val_aucs.append(val_auc if val_auc is not None else 0.0)

        metrics_str = (f'val_acc {val_acc:.4f} '
                       f'val_prec {val_precision:.4f} '
                       f'val_recall {val_recall:.4f} '
                       f'val_f1 {val_f1:.4f}')
        if val_auc is not None:
            metrics_str += f' val_auc {val_auc:.4f}'

        print(f'Epoch {epoch}/{args.num_epochs} '
              f'train_loss {epoch_loss:.4f} train_acc {epoch_acc:.4f} '
              + metrics_str)

        # compute train metrics (precision/recall/f1) after epoch
        if len(train_y_true) > 0:
            train_precision = precision_score(train_y_true, train_y_pred, average='macro')
            train_recall = recall_score(train_y_true, train_y_pred, average='macro')
            train_f1 = f1_score(train_y_true, train_y_pred, average='macro')
        else:
            train_precision = train_recall = train_f1 = 0.0

        if args.wandb and wandb is not None:
            log_data = {
                "epoch": epoch,
                "train_loss": epoch_loss,
                "train_acc": epoch_acc.item(),
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_f1": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc.item(),
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
            }
            if val_auc is not None:
                log_data["val_auc"] = val_auc
            wandb.log(log_data, step=epoch)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(),
                       f'checkpoints/{args.exp_name}/best_resnet50.pth')

    # Save final validation confusion matrix (once at end of training)
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Validation Confusion Matrix')

        plt.savefig(f'checkpoints/{args.exp_name}/confusion_matrix_val.png')

        if args.wandb and wandb is not None:
            wandb.log({"val_confusion_matrix": plt})

        plt.close()
    except ImportError:
        pass

    # optional final test evaluation
    if len(test_ds) > 0:
        model.eval()
        test_correct = 0
        y_true = []
        y_pred = []
        y_probs = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device); labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_correct += torch.sum(preds==labels.data)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_probs.extend(nn.functional.softmax(outputs, dim=1).cpu().numpy())
        test_acc = test_correct.double() / len(test_ds)

        # compute additional metrics on test set
        if len(y_true) > 0:
            test_precision = precision_score(y_true, y_pred, average='macro')
            test_recall = recall_score(y_true, y_pred, average='macro')
            test_f1 = f1_score(y_true, y_pred, average='macro')
            try:
                if n_classes == 2:
                    test_auc = roc_auc_score(y_true, [p[1] for p in y_probs])
                else:
                    test_auc = roc_auc_score(y_true, y_probs, multi_class='ovo', average='macro')
            except Exception:
                test_auc = None
        else:
            test_precision = test_recall = test_f1 = 0.0
            test_auc = None

        metric_summary = (f'accuracy {test_acc:.4f} '
                          f'precision {test_precision:.4f} '
                          f'recall {test_recall:.4f} '
                          f'f1 {test_f1:.4f}')
        if test_auc is not None:
            metric_summary += f' auc {test_auc:.4f}'
        print(f'Test metrics: {metric_summary}')

        if args.wandb and wandb is not None:
            log_data = {
                "test_accuracy": test_acc.item(),
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
            }
            if test_auc is not None:
                log_data["test_auc"] = test_auc
            wandb.log(log_data)

        # confusion matrix plot
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(6,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            if args.wandb and wandb is not None:
                wandb.log({"confusion_matrix": plt})

            plt.savefig(f'checkpoints/{args.exp_name}/confusion_matrix.png')
            plt.close()
        except ImportError:
            pass


    # optionally save training/validation curves if plotting libraries are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        epochs = range(1, len(train_losses) + 1)
        # loss curve
        plt.figure()
        plt.plot(epochs, train_losses, label='train_loss')
        plt.plot(epochs, val_losses, label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'checkpoints/{args.exp_name}/loss.png')
        plt.close()
        # accuracy curve
        plt.figure()
        plt.plot(epochs, train_accs, label='train_acc')
        plt.plot(epochs, val_accs, label='val_acc')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'checkpoints/{args.exp_name}/acc.png')
        plt.close()
        # f1/auc curve
        plt.figure()
        plt.plot(epochs, val_f1s, label='val_f1')
        plt.plot(epochs, val_aucs, label='val_auc')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(f'checkpoints/{args.exp_name}/val_metrics.png')
        plt.close()
    except ImportError:
        pass

    print(f'Best val acc: {best_acc:.4f}')

    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == '__main__':
    main()