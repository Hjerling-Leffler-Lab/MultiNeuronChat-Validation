import os

import numpy as np

import loompy

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import cm

import argparse

def plot_legend(
        colors: dict[str, str],
        path_to_output: str,
        width_in_cm: float,
        height_in_cm: float,
        font_size: float = 8,
):
    fig = plt.figure(figsize=(width_in_cm * cm, height_in_cm * cm))

    legend_handles = []

    for cell_type, color in colors.items():
        legend_handles.append(
            mpatches.Circle(
                (0, 0),
                0.5,
                color=color,
                label=cell_type
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
        '--path_to_umap_all_original',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_umap_all_simulated',
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
    path_to_umap_all_original: str = args.path_to_umap_all_original
    path_to_umap_all_simulated: str = args.path_to_umap_all_simulated

    width_in_cm_umap: float = args.width_in_cm_umap
    height_in_cm_umap: float = args.height_in_cm_umap

    width_in_cm_legend: float = args.width_in_cm_legend
    height_in_cm_legend: float = args.height_in_cm_legend

    font_size: float = args.font_size

    os.makedirs(os.path.dirname(path_to_legend), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_umap_all_original), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_umap_all_simulated), exist_ok=True)

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

    plot_legend(
        colors=cluster_colors_dict,
        font_size=font_size,
        width_in_cm=width_in_cm_legend,
        height_in_cm=height_in_cm_legend,
        save_path=path_to_legend
    )

    with loompy.connect(path_to_loom) as ds:
        meta_data: dict[str, np.ndarray] = { key: ds.ca[key] for key in ds.ca.keys() }

        print(set(meta_data['Donor']))
        print(len(list(filter(lambda x: x.startswith('F') or x.startswith('M'), set(meta_data['Donor'])))))

        data = ds[:, :]


    # Shuffle the data
    np.random.seed(42)
    indices = np.random.permutation(data.shape[1])
    data = data[:, indices]
    meta_data = { key: meta_data[key][indices] for key in meta_data.keys() }

    simulated_mask = meta_data['Simulated'] == 'Yes'
    original_mask = ~simulated_mask

    simulated_data = data[:, simulated_mask]
    simulated_meta = { key: meta_data[key][simulated_mask] for key in meta_data.keys() }

    original_data = data[:, original_mask]
    original_meta = { key: meta_data[key][original_mask] for key in meta_data.keys() }

    # Plot original data
    fig, axs = plt.subplots(1, 1, figsize=(width_in_cm_umap * cm, height_in_cm_umap * cm))

    colors = [cluster_colors_dict[cluster] for cluster in original_meta['cluster_name_15CTs']]

    axs.scatter(
        original_data[0, :],
        original_data[1, :],
        c=colors,
        s=0.1,
        alpha=0.5
    )

    # Disable spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)

    # Disable ticks
    axs.set_xticks([])
    axs.set_yticks([])

    axs.set_xlim(np.min(original_data[0, :]), np.max(original_data[0, :]))
    axs.set_ylim(np.min(original_data[1, :]), np.max(original_data[1, :]))

    fig.tight_layout()

    fig.savefig(path_to_umap_all_original, dpi=300)

    plt.close(fig)

    # Plot simulated data
    fig, axs = plt.subplots(1, 1, figsize=(width_in_cm_umap * cm, height_in_cm_umap * cm))

    colors = [cluster_colors_dict[cluster] for cluster in simulated_meta['cluster_name_15CTs']]

    axs.scatter(
        simulated_data[0, :],
        simulated_data[1, :],
        c=colors,
        s=0.1,
        alpha=0.5
    )

    # Disable spines
    axs.spines['top'].set_visible(False)
    axs.spines['right'].set_visible(False)
    axs.spines['bottom'].set_visible(False)
    axs.spines['left'].set_visible(False)

    # Disable ticks
    axs.set_xticks([])
    axs.set_yticks([])

    axs.set_xlim(np.min(simulated_data[0, :]), np.max(simulated_data[0, :]))
    axs.set_ylim(np.min(simulated_data[1, :]), np.max(simulated_data[1, :]))

    fig.tight_layout()

    fig.savefig(path_to_umap_all_simulated, dpi=300)

    plt.close(fig)



if __name__ == '__main__':
    main()