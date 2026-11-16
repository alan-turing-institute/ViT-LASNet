"""Shared training helpers for the ViT/BEiT scripts in ``train/``."""
import os

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from transformers import BeitForImageClassification, ViTForImageClassification

from tools.loss import (
    Bal_CE_loss, BCE_loss, CB_CE_loss, LADE_loss, LDAM_loss, LS_CE_loss,
    MiSLAS_loss, MiSLAS_vit_loss, ST_CE_loss,
)

LOSSES = ['ST_CE', 'Bal_CE', 'BCE', 'LS_CE', 'MiSLAS', 'MiSLAS_vit', 'LADE', 'LDAM', 'CB_CE', 'CE']
# Losses that expect one-hot targets rather than class indices (LADE takes indices).
ONE_HOT_LOSSES = {'ST_CE', 'Bal_CE', 'BCE', 'MiSLAS', 'MiSLAS_vit', 'LDAM', 'CB_CE'}
PRETRAINED = {
    'vit': (ViTForImageClassification, 'google/vit-base-patch16-224'),
    'beit': (BeitForImageClassification, 'microsoft/beit-base-patch16-224-pt22k-ft22k'),
}


class LossArgs:
    """Container for the attributes the loss classes in tools/loss.py read."""

    def __init__(self, cls_num, smoothing=0.1, bal_tau=1.0, device='cpu'):
        self.cls_num = list(cls_num)
        self.nb_classes = len(self.cls_num)
        self.smoothing = smoothing
        self.bal_tau = bal_tau
        self.device = device


def get_loss_function(loss_type, args):
    factories = {
        'ST_CE': lambda: ST_CE_loss(),
        'Bal_CE': lambda: Bal_CE_loss(args),
        'BCE': lambda: BCE_loss(args),
        'LS_CE': lambda: LS_CE_loss(smoothing=args.smoothing),
        'MiSLAS': lambda: MiSLAS_loss(args),
        'MiSLAS_vit': lambda: MiSLAS_vit_loss(args),
        'LADE': lambda: LADE_loss(args),
        'LDAM': lambda: LDAM_loss(args),
        'CB_CE': lambda: CB_CE_loss(args),
    }
    return factories.get(loss_type, nn.CrossEntropyLoss)()


def one_hot_encode(labels, num_classes):
    return torch.eye(num_classes, device=labels.device)[labels]


def run_epoch(model, loader, loss_fn, loss_type, num_classes, device, optimizer=None, desc=None):
    """One pass over ``loader``; trains when an optimizer is given. Returns (loss sum, n correct)."""
    total_loss, correct = 0.0, 0
    for images, labels in tqdm(loader, desc=desc, disable=desc is None):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        targets = one_hot_encode(labels, num_classes) if loss_type in ONE_HOT_LOSSES else labels
        if loss_type == 'MiSLAS_vit':
            loss = loss_fn(outputs.logits, targets, outputs.attentions)
        else:
            loss = loss_fn(outputs.logits, targets)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        correct += (outputs.logits.argmax(dim=-1) == labels).sum().item()
    return total_loss, correct


def train_and_validate(model_type, loss_type, learning_rate, num_epochs, loss_args, train_loader, val_loader,
                       output_dir, device):
    """Fine-tune a pretrained model and save the checkpoint with the best validation accuracy."""
    num_classes = loss_args.nb_classes
    model_cls, checkpoint = PRETRAINED[model_type]
    model = model_cls.from_pretrained(checkpoint, num_labels=num_classes, output_attentions=True,
                                      ignore_mismatched_sizes=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = get_loss_function(loss_type, loss_args)
    run_name = f"{model_type}_finetuned_{loss_type}_lr{learning_rate}_epochs{num_epochs}"
    n_val = len(val_loader.dataset)

    best_val_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        run_epoch(model, train_loader, loss_fn, loss_type, num_classes, device, optimizer,
                  desc=f"Epoch {epoch + 1}/{num_epochs}")
        model.eval()
        with torch.no_grad():
            _, val_correct = run_epoch(model, val_loader, loss_fn, loss_type, num_classes, device)
        val_accuracy = val_correct / n_val
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_name}.pt"))
            model.save_pretrained(os.path.join(output_dir, run_name))
            print(f"Saved best model to {os.path.join(output_dir, run_name)}")
    return best_val_accuracy
