"""Score top-3 prediction CSVs from ``test/test.py --top_k 3`` against ground-truth labels.

For each ``*_topthree.csv`` in ``--predictions-dir`` this writes a per-class
classification report, summary metrics (accuracy, weighted precision, recall
and F1, top-3 accuracy) and a confusion-matrix plot, then reports which model
has the best F1 on ``--minority-class``.

The labels CSV needs ``Filename`` and ``Label`` columns.

Example::

    python tools/evaluate.py --predictions-dir results/ --labels-csv labels/image_labels.csv \\
        --output-dir results/eval --minority-class Zoo_other
"""
import argparse
import glob
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,  # noqa: E402
                             precision_score, recall_score)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def process_file(pred_file, df_true, acc_dir, cm_dir):
    df_merged = pd.merge(df_true, pd.read_csv(pred_file), on='Filename', how='inner')
    true_labels = df_merged['Label']
    predicted = df_merged['Top1 Prediction']
    classes = sorted(df_true['Label'].unique())
    stem = os.path.splitext(os.path.basename(pred_file))[0]

    report = pd.DataFrame(classification_report(true_labels, predicted, output_dict=True, zero_division=0))
    report = report.transpose().reset_index().rename(columns={'index': 'Label'})
    report.to_csv(os.path.join(acc_dir, f'{stem}_report.csv'), index=False)

    top3_hits = df_merged.apply(
        lambda r: r['Label'] in (r['Top1 Prediction'], r['Top2 Prediction'], r['Top3 Prediction']), axis=1)
    metrics = {
        'Accuracy': accuracy_score(true_labels, predicted),
        'Precision': precision_score(true_labels, predicted, average='weighted', zero_division=0),
        'Recall': recall_score(true_labels, predicted, average='weighted', zero_division=0),
        'F1 Score': f1_score(true_labels, predicted, average='weighted', zero_division=0),
        'Top-3 Accuracy': top3_hits.mean(),
    }
    print(f'\n{stem}')
    with open(os.path.join(acc_dir, f'{stem}_metrics.txt'), 'w') as fh:
        for name, value in metrics.items():
            print(f'  {name}: {value:.4f}')
            fh.write(f'{name}: {value}\n')

    cm = confusion_matrix(true_labels, predicted, labels=classes)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm_pct, annot=True, fmt='.2f', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion matrix (% of true class)')
    plt.xlabel('Predicted class')
    plt.ylabel('True class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, f'{stem}.png'), transparent=True)
    plt.close()
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions-dir', required=True, help='folder of top-3 prediction CSVs')
    p.add_argument('--labels-csv', required=True, help='CSV with Filename and Label columns')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pattern', default='*_topthree.csv', help='glob for prediction files')
    p.add_argument('--minority-class', default='Zoo_other')
    args = p.parse_args()

    acc_dir = os.path.join(args.output_dir, 'ACC')
    cm_dir = os.path.join(args.output_dir, 'CM')
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)
    df_true = pd.read_csv(args.labels_csv)

    f1_scores = {}
    for pred_file in sorted(glob.glob(os.path.join(args.predictions_dir, args.pattern))):
        report = process_file(pred_file, df_true, acc_dir, cm_dir)
        row = report.loc[report['Label'] == args.minority_class, 'f1-score']
        if not row.empty:
            f1_scores[os.path.basename(pred_file)] = row.values[0]

    if not f1_scores:
        print(f'No prediction files with class {args.minority_class!r} found in {args.predictions_dir}')
        return
    best = max(f1_scores, key=f1_scores.get)
    summary = f'Best model on minority class {args.minority_class}: {best} (F1 {f1_scores[best]:.4f})'
    print(f'\n{summary}')
    with open(os.path.join(args.output_dir, 'comparison_results.txt'), 'w') as fh:
        for name, score in f1_scores.items():
            fh.write(f'{name}: {score}\n')
        fh.write(f'\n{summary}\n')


if __name__ == '__main__':
    main()
