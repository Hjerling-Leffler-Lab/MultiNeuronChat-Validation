import os

import numpy as np

import loompy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import cm

import argparse


def plot_state_legend(
        state_colors_dict: dict[str, str],
        path_to_output: str,
        width_in_cm: float,
        height_in_cm: float,
        font_size: float = 8,
):
    fig = plt.figure(figsize=(width_in_cm * cm, height_in_cm * cm))

    legend_handles = []

    for state, color in state_colors_dict.items():
        legend_handles.append(
            mpatches.Circle(
                (0, 0),
                0.5,
                color=color,
                label=state
            )
        )

    plt.legend(handles=legend_handles, loc='center', prop={'size': font_size}, ncol=3)
    plt.axis('off')

    plt.savefig(path_to_output, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_loom',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_legend',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_umap_cell_type_annotated',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_umap_disease_annotated',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--width_in_cm_umap',
        type=float,
        default=10.5
    )
    parser.add_argument(
        '--height_in_cm_umap',
        type=float,
        default=10.5
    )

    parser.add_argument(
        '--width_in_cm_legend',
        type=float,
        default=21,
    )
    parser.add_argument(
        '--height_in_cm_legend',
        type=float,
        default=5,
    )

    parser.add_argument(
        '--font_size',
        type=float,
        default=8,
    )

    args = parser.parse_args()

    path_to_loom: str = args.path_to_loom
    path_to_legend: str = args.path_to_legend
    path_to_umap_cell_type_annotated: str = args.path_to_umap_cell_type_annotated
    path_to_umap_disease_annotated: str = args.path_to_umap_disease_annotated

    width_in_cm_umap: float = args.width_in_cm_umap
    height_in_cm_umap: float = args.height_in_cm_umap

    width_in_cm_legend: float = args.width_in_cm_legend
    height_in_cm_legend: float = args.height_in_cm_legend

    font_size: float = args.font_size

    os.makedirs(os.path.dirname(path_to_legend), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_umap_cell_type_annotated), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_umap_disease_annotated), exist_ok=True)

    cluster_colors_dict = {
        # 'none (removed)': '#C0C0C0',
        'Excitatory Layer 2-3 IT neurons II': '#21EC1D',
        'Excitatory Layer 3-6 IT neurons': '#58D2CF',
        'Excitatory Layer 5-6 CT and NP neurons': '#0D5A8B',
        'Excitatory Layer 5-6 IT neurons II': '#394FD3', 'Excitatory Layer 2-3 IT neurons I': '#2EBF5E',
        'Excitatory Layer 3-4 IT neurons': '#5959AD',
        'Excitatory Layer 5-6 IT neurons I': '#9AB7E2',
        'Inhibitory VIP neurons': '#B864CC',
        'Inhibitory PVALB neurons': '#D93137',
        'Inhibitory LAMP5 neurons': '#DA808C',
        'Inhibitory SST neurons': '#FF9900',
        'Oligodendrocytes': '#53776C',
        'Astrocytes': '#665C47',
        'Microglial cells': '#94AF97',
        'Oligodendrocyte progenitor cells': '#697255',
        'Endothelial and mural cells': '#66341E'
    }

    state_colors_dict = {
        'CASE': '#78111C',
        'CTRL': '#16498F',
    }


    plot_state_legend(
        state_colors_dict=state_colors_dict,
        path_to_output=path_to_legend,
        width_in_cm=width_in_cm_legend,
        height_in_cm=height_in_cm_legend,
        font_size=font_size,
    )

    with loompy.connect(path_to_loom) as ds:
        meta_data: dict[str, np.ndarray] = {key: ds.ca[key] for key in ds.ca.keys()}

        case_mask = meta_data['Disease'] == 'CASE'
        ctrl_mask = meta_data['Disease'] == 'CTRL'

        data = ds[:, :]

    min_x = np.min(data[0, :])
    max_x = np.max(data[0, :])
    min_y = np.min(data[1, :])
    max_y = np.max(data[1, :])

    fig, ax = plt.subplots(figsize=(width_in_cm_umap * cm, height_in_cm_umap * cm))

    # Shuffle the dots to avoid overplotting
    np.random.seed(42)
    shuffle_indices = np.random.permutation(data.shape[1])
    data = data[:, shuffle_indices]
    meta_data = {key: meta_data[key][shuffle_indices] for key in meta_data.keys()}

    dot_colors = [cluster_colors_dict[cluster] for cluster in meta_data['cluster_name_15CTs']]

    ax.scatter(
        data[0, :],
        data[1, :],
        color=dot_colors,
        s=0.1,
        alpha=0.5
    )

    # Disable all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    fig.tight_layout()
    fig.savefig(path_to_umap_cell_type_annotated, dpi=300)

    plt.close(fig)

    fig, ax = plt.subplots(figsize=(width_in_cm_umap * cm, height_in_cm_umap * cm))

    dot_colors = [state_colors_dict[state] for state in meta_data['Disease']]

    ax.scatter(
        data[0, :],
        data[1, :],
        color=dot_colors,
        s=0.1,
        alpha=0.5
    )

    # Disable all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    fig.tight_layout()
    fig.savefig(path_to_umap_disease_annotated, dpi=300)

    plt.close(fig)


if __name__ == '__main__':
    main()
