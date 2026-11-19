#!/usr/bin/env python3
"""Classify plankton images with a trained ResNet, ViT or BEiT model and write a CSV.

Model versions: 0 = ResNet50, 1 = ResNet18, 2 = ViT, 3 = BEiT. ResNet weights are a
``.pth`` or ``.safetensors`` state dict; ViT/BEiT weights are a ``save_pretrained``
folder containing ``config.json`` and ``model.safetensors``. ``-w`` also accepts a
Hugging Face model as ``<user>/<repo>/<sub-folder>``, e.g. ``Noushineft/ViT-LASNet/vit-mislas``.

With ``--top-k 1`` (the default) the CSV has ``Filename, Predicted Class``. With
``--top-k 3`` it has ``Top1 Prediction, Top1 Probability, ...`` columns, which is
the format ``tools/evaluate.py`` reads.

Examples::

    python test/test.py -m 2 -n 18 -w weights/vit_finetuned_MiSLAS_vit_lr5e-05_epochs30 \\
        -f images/ -o predictions.csv
    python test/test.py -m 1 -w weights/ResNet_18_3classes_RGB.pth -f images/ -o predictions.csv
"""
import argparse
import csv
import logging
import os
from io import BytesIO

import tifffile as tiff
import torch
import torchvision
from PIL import Image
from torchvision.transforms import functional
from transformers import (AutoImageProcessor, BeitForImageClassification, BeitImageProcessor,
                          ViTForImageClassification)

logging.basicConfig(level=logging.INFO)

CLASS_LABELS = {
    3: ["copepod", "detritus", "noncopepod"],
    18: [
        "Detritus",
        "Phyto_diatom",
        "Phyto_diatom_chaetocerotanae_Chaetoceros",
        "Phyto_diatom_rhisoleniales_Guinardia flaccida",
        "Phyto_diatom_rhisoleniales_Rhizosolenia",
        "Phyto_dinoflagellate_gonyaulacales_Tripos",
        "Phyto_dinoflagellate_gonyaulacales_Tripos macroceros",
        "Phyto_dinoflagellate_gonyaulacales_Tripos muelleri",
        "Zoo_cnidaria",
        "Zoo_crustacea_copepod",
        "Zoo_crustacea_copepod_calanoida",
        "Zoo_crustacea_copepod_calanoida_Acartia",
        "Zoo_crustacea_copepod_calanoida_Centropages",
        "Zoo_crustacea_copepod_cyclopoida",
        "Zoo_crustacea_copepod_cyclopoida_Oithona",
        "Zoo_crustacea_copepod_nauplii",
        "Zoo_other",
        "Zoo_tintinnidae",
    ],
}


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")
    return device


def resolve_weights(weights):
    """Return a local path for ``weights``.

    A path that exists locally is returned unchanged. Anything else is read as a
    Hugging Face model, ``<user>/<repo>/<sub-folder>``, for example
    ``Noushineft/ViT-LASNet/vit-mislas``, and downloaded.
    """
    if os.path.exists(weights):
        return weights
    from huggingface_hub import snapshot_download
    parts = weights.strip("/").split("/")
    if len(parts) < 3:
        raise FileNotFoundError(f"{weights!r} is not a local path or a Hugging Face '<user>/<repo>/<sub-folder>'")
    repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    local = snapshot_download(repo_id, allow_patterns=[f"{subfolder}/*"])
    return os.path.join(local, subfolder)


def load_state_dict(path):
    if os.path.isdir(path):
        path = os.path.join(path, "model.safetensors")
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_model(weights, device, model_version, num_classes, gray=False):
    """Return (model, processor). ``processor`` is None for the ResNet models."""
    weights = resolve_weights(weights)
    if model_version == 2:
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        model = ViTForImageClassification.from_pretrained(weights)
    elif model_version == 3:
        processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")
        model = BeitForImageClassification.from_pretrained(weights)
    else:
        processor = None
        model = torchvision.models.resnet50() if model_version == 0 else torchvision.models.resnet18()
        if gray:
            model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        model.load_state_dict(load_state_dict(weights))
    model = model.to(device)
    model.eval()
    return model, processor


def resnet_tensor(image_bytes, gray):
    t = functional.to_tensor(tiff.imread(BytesIO(image_bytes)))
    t = functional.resize(t, (256, 256))
    if gray:
        t = (0.2125 * t[0] + 0.7154 * t[1] + 0.0721 * t[2]).unsqueeze(0)
    return t


# Matches the deterministic part of the training transform in train/train.py.
VIT_EVAL_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def classify_batch(image_bytes_list, device, model, labels, processor=None, gray=False, top_k=1,
                   preprocess="train"):
    """Return, for each image, a list of ``top_k`` (label, probability) pairs.

    For ViT/BEiT, ``preprocess="train"`` uses the same resize, crop and ImageNet
    normalisation as training. ``"processor"`` uses the Hugging Face image processor
    (resize to 224, mean/std 0.5), which is how results were produced before
    November 2026.
    """
    with torch.no_grad():
        if processor is not None:
            images = [Image.open(BytesIO(b)).convert("RGB") for b in image_bytes_list]
            if preprocess == "train":
                pixel_values = torch.stack([VIT_EVAL_TRANSFORM(im) for im in images]).to(device)
            else:
                pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits
        else:
            batch = torch.stack([resnet_tensor(b, gray) for b in image_bytes_list]).to(device)
            logits = model(batch)
        probs, preds = torch.topk(torch.softmax(logits, dim=1), top_k)
    return [[(labels[p], pr.item()) for p, pr in zip(row_p, row_pr)] for row_p, row_pr in zip(preds, probs)]


def write_csv(path, filenames, results, top_k):
    if top_k == 1:
        fieldnames = ["Filename", "Predicted Class"]
        rows = [{"Filename": f, "Predicted Class": r[0][0]} for f, r in zip(filenames, results)]
    else:
        fieldnames = ["Filename"]
        for k in range(1, top_k + 1):
            fieldnames += [f"Top{k} Prediction", f"Top{k} Probability"]
        rows = []
        for f, r in zip(filenames, results):
            row = {"Filename": f}
            for k, (label, prob) in enumerate(r, start=1):
                row[f"Top{k} Prediction"], row[f"Top{k} Probability"] = label, prob
            rows.append(row)
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logging.info(f"Results saved to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("filename", help="TIFF file, or folder of TIFF files with -f")
    parser.add_argument("-w", "--weights", required=True,
                        help="ResNet .pth/.safetensors file, ViT/BEiT save_pretrained folder, "
                             "or Hugging Face <user>/<repo>/<sub-folder>")
    parser.add_argument("-m", "--model_version", type=int, choices=[0, 1, 2, 3], default=0,
                        help="0 = ResNet50, 1 = ResNet18, 2 = ViT, 3 = BEiT (default 0)")
    parser.add_argument("-n", "--num_classes", type=int, choices=sorted(CLASS_LABELS), default=3,
                        help="number of class labels (default 3)")
    parser.add_argument("-g", "--gray", action="store_true", help="ResNet weights were trained on greyscale input")
    parser.add_argument("-f", "--folder", action="store_true", help="classify every .tif/.tiff file in a folder")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="images per forward pass")
    parser.add_argument("-k", "--top_k", type=int, default=1, help="number of predictions per image")
    parser.add_argument("--preprocess", choices=["train", "processor"], default="train",
                        help="ViT/BEiT input preprocessing: 'train' matches training (default); "
                             "'processor' reproduces results produced before November 2026")
    parser.add_argument("-o", "--output_csv", default="predictions.csv", help="output CSV path")
    args = parser.parse_args()

    labels = CLASS_LABELS[args.num_classes]
    device = get_device()
    model, processor = load_model(args.weights, device, args.model_version, len(labels), args.gray)

    if args.folder:
        filenames = sorted(f for f in os.listdir(args.filename) if f.lower().endswith((".tif", ".tiff")))
        paths = [os.path.join(args.filename, f) for f in filenames]
    else:
        filenames, paths = [args.filename], [args.filename]

    results = []
    for i in range(0, len(paths), args.batch_size):
        batch = []
        for path in paths[i:i + args.batch_size]:
            with open(path, "rb") as fh:
                batch.append(fh.read())
        results += classify_batch(batch, device, model, labels, processor, args.gray, args.top_k, args.preprocess)

    write_csv(args.output_csv, filenames, results, args.top_k)


if __name__ == "__main__":
    main()
