"""Nested cross-validation over long-tail losses for ViT or BEiT.

The inner folds pick the best (loss, learning rate, epochs) combination; that
combination is then retrained on each outer training fold and scored on the
outer validation fold.

Example::

    python train/train_nested_cv.py --data-dir data/train --output-dir runs/cv \\
        --model vit --loss ST_CE Bal_CE MiSLAS MiSLAS_vit --epochs 30
"""
import argparse
import os
import sys

import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.training import LOSSES, PRETRAINED, LossArgs, train_and_validate  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='ImageFolder-style training directory')
    p.add_argument('--output-dir', required=True, help='where checkpoints are written')
    p.add_argument('--model', choices=sorted(PRETRAINED), default='vit')
    p.add_argument('--loss', nargs='+', choices=LOSSES,
                   default=['ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LDAM', 'CB_CE'])
    p.add_argument('--lr', nargs='+', type=float, default=[5e-5])
    p.add_argument('--epochs', nargs='+', type=int, default=[30])
    p.add_argument('--outer-folds', type=int, default=5)
    p.add_argument('--inner-folds', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--smoothing', type=float, default=0.1)
    p.add_argument('--bal-tau', type=float, default=1.0)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def loader(dataset, cfg, shuffle):
    return DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=shuffle)


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(cfg.data_dir, transform=transform)
    class_counts = [dataset.targets.count(i) for i in range(len(dataset.classes))]
    loss_args = LossArgs(class_counts, cfg.smoothing, cfg.bal_tau, cfg.device)

    def fit(loss_type, lr, epochs, train_set, val_set):
        return train_and_validate(cfg.model, loss_type, lr, epochs, loss_args,
                                  loader(train_set, cfg, True), loader(val_set, cfg, False),
                                  cfg.output_dir, cfg.device)

    best, best_acc = None, 0.0
    outer_kf = KFold(n_splits=cfg.outer_folds, shuffle=True, random_state=cfg.seed)
    for fold, (train_idx, val_idx) in enumerate(outer_kf.split(dataset)):
        outer_train, outer_val = Subset(dataset, train_idx), Subset(dataset, val_idx)

        inner_kf = KFold(n_splits=cfg.inner_folds, shuffle=True, random_state=cfg.seed)
        for inner_train_idx, inner_val_idx in inner_kf.split(outer_train):
            inner_train, inner_val = Subset(outer_train, inner_train_idx), Subset(outer_train, inner_val_idx)
            for loss_type in cfg.loss:
                for lr in cfg.lr:
                    for epochs in cfg.epochs:
                        acc = fit(loss_type, lr, epochs, inner_train, inner_val)
                        print(f"[fold {fold}] loss={loss_type} lr={lr} epochs={epochs} inner val acc={acc:.4f}")
                        if acc > best_acc:
                            best, best_acc = (loss_type, lr, epochs), acc

        loss_type, lr, epochs = best
        outer_acc = fit(loss_type, lr, epochs, outer_train, outer_val)
        print(f"[fold {fold}] outer validation accuracy with loss={loss_type} lr={lr} epochs={epochs}: {outer_acc:.4f}")

    print(f"Best: model={cfg.model} loss={best[0]} lr={best[1]} epochs={best[2]} inner val acc={best_acc:.4f}")


if __name__ == '__main__':
    main()
