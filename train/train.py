"""Fine-tune ViT or BEiT on a plankton image folder with a choice of long-tail losses.

The data directory follows the torchvision ``ImageFolder`` layout (one
sub-folder per class). A fraction of the images is held out for validation, and
the best checkpoint (by validation accuracy) is saved for every
(loss, learning rate, epochs) combination in the grid.

Example::

    python train/train.py --data-dir data/train --output-dir runs/vit \\
        --model vit --loss MiSLAS_vit --lr 5e-5 --epochs 30
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.training import LOSSES, PRETRAINED, LossArgs, train_and_validate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='ImageFolder-style training directory')
    p.add_argument('--output-dir', required=True, help='where checkpoints are written')
    p.add_argument('--model', choices=sorted(PRETRAINED), default='vit')
    p.add_argument('--loss', nargs='+', choices=LOSSES, default=['MiSLAS_vit'])
    p.add_argument('--lr', nargs='+', type=float, default=[5e-5])
    p.add_argument('--epochs', nargs='+', type=int, default=[30])
    p.add_argument('--val-fraction', type=float, default=0.1)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument('--smoothing', type=float, default=0.1, help='label smoothing for LS_CE and BCE')
    p.add_argument('--bal-tau', type=float, default=1.0, help='logit-adjustment temperature for Bal_CE')
    p.add_argument('--attention-mode', choices=['entropy', 'legacy'], default='entropy',
                   help="MiSLAS_vit attention weighting; 'legacy' reproduces the published results")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    print("Class indices and labels:")
    for idx, class_name in enumerate(dataset.classes):
        print(f"{idx}: {class_name}")

    class_counts = [dataset.targets.count(i) for i in range(len(dataset.classes))]
    loss_args = LossArgs(class_counts, cfg.smoothing, cfg.bal_tau, cfg.device, cfg.attention_mode)

    n_val = int(round(cfg.val_fraction * len(dataset)))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val])
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False)

    best, best_acc = None, 0.0
    for loss_type in cfg.loss:
        for lr in cfg.lr:
            for epochs in cfg.epochs:
                acc = train_and_validate(cfg.model, loss_type, lr, epochs, loss_args, train_loader, val_loader,
                                         cfg.output_dir, cfg.device)
                print(f"model={cfg.model} loss={loss_type} lr={lr} epochs={epochs} best val acc={acc:.4f}")
                if acc > best_acc:
                    best, best_acc = (loss_type, lr, epochs), acc

    if best is not None:
        print(f"Best: model={cfg.model} loss={best[0]} lr={best[1]} epochs={best[2]} val acc={best_acc:.4f}")


if __name__ == '__main__':
    main()
