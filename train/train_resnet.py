"""Train a ResNet18 baseline (RGB or greyscale) on a train/test ImageFolder split.

``--data-dir`` must contain ``train/`` and ``test/`` sub-folders, each with one
sub-folder per class. This script replaces the earlier ``045-train.py``
(greyscale, class-weighted loss, 30 epochs) and ``046-train.py`` (RGB,
unweighted loss, 100 epochs); both are reproduced by the examples below.

Examples::

    # greyscale, class-weighted cross-entropy (was 045-train.py)
    python train/train_resnet.py --data-dir data/split --output-dir runs/resnet-gray \\
        --gray --weighted-loss --epochs 30

    # RGB, plain cross-entropy (was 046-train.py)
    python train/train_resnet.py --data-dir data/split --output-dir runs/resnet-rgb --epochs 100
"""
import argparse
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torchvision import datasets, models, transforms  # noqa: E402
from torchvision.models import ResNet18_Weights  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='folder containing train/ and test/ ImageFolders')
    p.add_argument('--output-dir', required=True, help='where weights and the loss plot are written')
    p.add_argument('--gray', action='store_true', help='train a single-channel (greyscale) model')
    p.add_argument('--weighted-loss', action='store_true', help='weight cross-entropy by inverse class frequency')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def plot_loss(loss_values, path):
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(loss_values) + 1), loss_values, label='Test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Test loss over epochs')
    plt.legend()
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def macro_f1(true_labels, preds, num_classes):
    scores = []
    for c in range(num_classes):
        tp = sum(p == c and t == c for p, t in zip(preds, true_labels))
        fp = sum(p == c and t != c for p, t in zip(preds, true_labels))
        fn = sum(p != c and t == c for p, t in zip(preds, true_labels))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0)
    return sum(scores) / max(len(scores), 1)


def build_model(num_classes, gray):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if gray:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_transform(gray):
    steps = [transforms.Grayscale(num_output_channels=1)] if gray else []
    steps += [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.)),
        transforms.RandomResizedCrop(size=(224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ToTensor(),
    ]
    return transforms.Compose(steps)


def train_model(model, criterion, optimizer, train_loader, test_loader, cfg, num_classes):
    loss_values = []
    for epoch in range(cfg.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loss, correct, all_true, all_pred = 0.0, 0, [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(cfg.device), labels.to(cfg.device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                all_true += labels.tolist()
                all_pred += pred.tolist()

        n_test = len(test_loader.dataset)
        loss_values.append(test_loss / n_test)
        plot_loss(loss_values, os.path.join(cfg.output_dir, 'loss_plot.png'))
        print(f'Epoch {epoch + 1}/{cfg.epochs}, Loss: {loss_values[-1]:.4f}, '
              f'Accuracy: {correct / n_test:.4f}, Macro-F1: {macro_f1(all_true, all_pred, num_classes):.4f}')


def main():
    cfg = parse_args()
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    transform = build_transform(cfg.gray)
    train_set = datasets.ImageFolder(os.path.join(cfg.data_dir, 'train'), transform=transform)
    test_set = datasets.ImageFolder(os.path.join(cfg.data_dir, 'test'), transform=transform)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)
    num_classes = len(train_set.classes)
    print("Class indices:")
    for idx, name in enumerate(train_set.classes):
        print(f"{name}: {idx}")

    weight = None
    if cfg.weighted_loss:
        counts = [0] * num_classes
        for _, label in train_set.samples:
            counts[label] += 1
        weight = torch.tensor([sum(counts) / c for c in counts], dtype=torch.float).to(cfg.device)

    model = build_model(num_classes, cfg.gray).to(cfg.device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    train_model(model, criterion, optimizer, train_loader, test_loader, cfg, num_classes)

    name = f"resnet18_{'gray' if cfg.gray else 'rgb'}{'_weighted' if cfg.weighted_loss else ''}.pth"
    torch.save(model.state_dict(), os.path.join(cfg.output_dir, name))
    print(f"Saved weights to {os.path.join(cfg.output_dir, name)}")


if __name__ == '__main__':
    main()
