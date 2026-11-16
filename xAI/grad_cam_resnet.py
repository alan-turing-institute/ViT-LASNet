"""Grad-CAM heat maps for a trained ResNet18 plankton classifier.

``--data-dir`` holds one sub-folder per class; sub-folders are sorted
alphabetically to give class indices, matching torchvision ``ImageFolder``.
A random sample of images from each class is explained against its true class.

Example::

    python xAI/grad_cam_resnet.py --weights weights/resnet18_rgb.pth \\
        --data-dir data/test --output-dir results/grad-cam --num-classes 3
"""
import argparse
import os
import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torchvision import models, transforms  # noqa: E402


class GradCam:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.feature_maps = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_features)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_features(self, module, inputs, output):
        self.feature_maps = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, class_idx):
        output = self.model(input_image)
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[:, class_idx] = 1.0
        output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        return F.relu(torch.sum(weights * self.feature_maps, dim=1, keepdim=True))


def apply_grad_cam(grad_cam, image_path, class_idx, output_dir):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_tensor = transform(Image.open(image_path).convert('RGB'))
    input_image = image_tensor.unsqueeze(0)
    cam = grad_cam.generate_cam(input_image, class_idx)
    cam = transforms.Resize(tuple(input_image.shape[2:]))(cam)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(transforms.ToPILImage()(image_tensor))
    axs[0].set_title('Original image')
    axs[0].set_aspect('auto')
    axs[1].imshow(transforms.ToPILImage()(image_tensor))
    pcm = axs[1].imshow(cam.squeeze().detach().numpy(), alpha=0.5, cmap='jet')
    axs[1].set_title('Grad-CAM')
    axs[1].set_aspect('auto')
    fig.colorbar(pcm, ax=axs[1], shrink=0.6).set_label('Intensity')
    fig.tight_layout()
    stem = os.path.splitext(os.path.basename(image_path))[0]
    fig.savefig(os.path.join(output_dir, f'{stem}_GRAD.png'))
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True, help='ResNet18 state dict (.pth)')
    p.add_argument('--data-dir', required=True, help='folder with one sub-folder of images per class')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--num-classes', type=int, default=3)
    p.add_argument('--num-images', type=int, default=20, help='images sampled per class')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, args.num_classes)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    grad_cam = GradCam(model, target_layer=model.layer4[-1])

    class_dirs = sorted(d for d in os.listdir(args.data_dir)
                        if os.path.isdir(os.path.join(args.data_dir, d)) and not d.startswith('.'))
    samples = []
    for class_idx, name in enumerate(class_dirs):
        directory = os.path.join(args.data_dir, name)
        files = [f for f in os.listdir(directory) if not f.startswith('.')]
        chosen = random.sample(files, min(args.num_images, len(files)))
        samples += [(os.path.join(directory, f), class_idx) for f in chosen]
    random.shuffle(samples)

    for image_path, class_idx in samples:
        apply_grad_cam(grad_cam, image_path, class_idx, args.output_dir)
    print(f'Wrote {len(samples)} Grad-CAM images to {args.output_dir}')


if __name__ == '__main__':
    main()
