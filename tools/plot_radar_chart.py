"""Radar charts of precision, recall and F1 across classification-report CSVs.

Each CSV in ``--reports-dir`` (one per model) needs ``Class Name``, ``precision``,
``recall`` and ``f1-score`` columns; the first ``--n-rows`` rows are plotted.

Example::

    python tools/plot_radar_chart.py --reports-dir results/reports --output-dir results/radar
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csv_files(folder_path, n_rows):
    frames = []
    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder_path, file_name), nrows=n_rows,
                             usecols=['Class Name', 'precision', 'recall', 'f1-score'])
            df['Filename'] = file_name
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def create_radar_chart(df, metric, output_dir=None):
    class_names = df['Class Name'].unique()
    files = df['Filename'].unique()
    angles = np.linspace(0, 2 * np.pi, len(files), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    for class_name in class_names:
        values = df.loc[df['Class Name'] == class_name, metric].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=class_name)
        ax.fill(angles, values, alpha=0.1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_title(f'Radar chart for {metric}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, f'radar_{metric}.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reports-dir', required=True, help='folder of classification-report CSVs')
    p.add_argument('--output-dir', help='save PNGs here instead of showing them')
    p.add_argument('--n-rows', type=int, default=3, help='number of class rows to read from each CSV')
    args = p.parse_args()

    df = read_csv_files(args.reports_dir, args.n_rows)
    for metric in ['precision', 'recall', 'f1-score']:
        create_radar_chart(df, metric, args.output_dir)


if __name__ == '__main__':
    main()
