import os

import numpy as np

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

import argparse

from plotting.utils import cell_type_to_short

parser = argparse.ArgumentParser(description='Plot a source x receiver heatmap of the number of significant interactions.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_heatmap_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_heatmap_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--statistical_test',
    type=str,
    required=True,
)
parser.add_argument(
    '--significance_threshold',
    type=float,
    default=0.1,
)
parser.add_argument(
    '--width_in_cm',
    type=float,
    default=8.0,
)
parser.add_argument(
    '--height_in_cm',
    type=float,
    default=7.0,
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_heatmap_png: str = args.path_to_heatmap_png
path_to_heatmap_pdf: str = args.path_to_heatmap_pdf

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

width_in_cm: float = args.width_in_cm
height_in_cm: float = args.height_in_cm

cm: float = 1/2.54  # cm to inches conversion factor


def main():
    for output_path in [path_to_heatmap_png, path_to_heatmap_pdf]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    p_values_adj = mnc_object.p_values_adj[statistical_test]

    # Count significant interactions per (source, receiver); NaN p-values (untested tetrads) compare as False
    n_significant = (p_values_adj < significance_threshold).sum(dim='interaction').to_numpy()
    n_tested = p_values_adj.notnull().sum(dim='interaction').to_numpy()

    source_labels = [cell_type_to_short.get(c, c) for c in mnc_object.source_cell_types]
    receiver_labels = [cell_type_to_short.get(c, c) for c in mnc_object.receiver_cell_types]

    fig, ax = plt.subplots(1, 1, figsize=(width_in_cm * cm, height_in_cm * cm))
    image = ax.imshow(n_significant, cmap='Reds', vmin=0, vmax=max(1, n_significant.max()))

    for i in range(n_significant.shape[0]):
        for j in range(n_significant.shape[1]):
            text_color = 'white' if n_significant[i, j] > 0.6 * n_significant.max() else 'black'
            ax.text(
                j, i, f'{n_significant[i, j]}\n/{n_tested[i, j]}',
                ha='center', va='center', fontsize=5, color=text_color,
            )

    ax.set_xticks(np.arange(len(receiver_labels)))
    ax.set_xticklabels(receiver_labels, fontsize=6, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(source_labels)))
    ax.set_yticklabels(source_labels, fontsize=6)
    ax.set_xlabel('Receiver', fontsize=7)
    ax.set_ylabel('Source', fontsize=7)
    ax.set_title(f'{statistical_test}: significant / tested interactions\n(adj. p < {significance_threshold})', fontsize=7)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(labelsize=6)
    colorbar.set_label('# significant interactions', fontsize=6)
    colorbar.ax.yaxis.get_major_locator().set_params(integer=True)

    fig.tight_layout()
    fig.savefig(path_to_heatmap_png, dpi=300)
    fig.savefig(path_to_heatmap_pdf, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
