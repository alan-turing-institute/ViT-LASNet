"""Exploratory analysis of an ImageFolder-style plankton dataset.

Writes a class-count table (PNG and CSV), a class-distribution bar chart and a
t-SNE scatter of downsampled raw pixels to ``--output-dir``.

Note: t-SNE is fitted separately on each batch of ``--batch-size`` images, and
images in a final incomplete batch are skipped.

Example::

    python EDA/eda.py --data-dir data/train --output-dir results/eda
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402


def class_counts(data_dir):
    names = sorted(n for n in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, n)) and not n.startswith('.'))
    return [(n, len([f for f in os.listdir(os.path.join(data_dir, n)) if not f.startswith('.')])) for n in names]


def save_distribution(class_data, output_dir):
    names, counts = zip(*class_data)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, counts, color='skyblue')
    ax.set_title('Class distribution', fontsize=16)
    ax.set_xlabel('Class name', fontsize=12)
    ax.set_ylabel('Number of images', fontsize=12)
    ax.tick_params(axis='x', rotation=90)
    ax.set_ylim(0, max(counts) + 100)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, 'bar_chart.png'))
    plt.close(fig)

    print('Total images:', sum(counts))
    print('Minimum class count:', min(counts))
    print('Maximum class count:', max(counts))
    print('Mean class count:', np.mean(counts))
    print('Median class count:', np.median(counts))

    table_data = [['Class Name', 'Number of Images']] + [list(row) for row in class_data]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    table = ax.table(cellText=table_data, loc='center', cellLoc='center',
                     cellColours=[['lightgray'] * 2] + [['white', 'white']] * len(class_data))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    fig.savefig(os.path.join(output_dir, 'table.png'), bbox_inches='tight', pad_inches=0.2)
    plt.close(fig)

    with open(os.path.join(output_dir, 'class_data.csv'), 'w', newline='') as fh:
        csv.writer(fh).writerows(table_data)


def image_batches(data_dir, class_names, batch_size, size):
    images, labels = [], []
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in sorted(os.listdir(class_path)):
            if img_name.startswith('.'):
                continue
            image = Image.open(os.path.join(class_path, img_name)).convert('RGB').resize((size, size))
            images.append(np.asarray(image).flatten())
            labels.append(class_name)
            if len(images) == batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True, help='folder with one sub-folder of images per class')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--image-size', type=int, default=128, help='images are resized to this square size')
    p.add_argument('--batch-size', type=int, default=100)
    p.add_argument('--perplexity', type=float, default=15)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    class_data = class_counts(args.data_dir)
    save_distribution(class_data, args.output_dir)

    tsne = TSNE(n_components=2, perplexity=args.perplexity, max_iter=250, random_state=args.seed)
    points, all_labels = [], []
    for X, y in image_batches(args.data_dir, [n for n, _ in class_data], args.batch_size, args.image_size):
        points.append(tsne.fit_transform(X))
        all_labels.extend(y)
    if not points:
        print('Not enough images for one t-SNE batch; skipping the scatter plot.')
        return

    result = np.concatenate(points)
    fig, ax = plt.subplots(figsize=(13, 10))
    sns.scatterplot(x=result[:, 0], y=result[:, 1], hue=all_labels, ax=ax)
    ax.set_title('t-SNE visualisation')
    ax.legend(loc='upper right', bbox_to_anchor=(0, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 't-sne_visualization.png'))
    plt.close(fig)


if __name__ == '__main__':
    main()
