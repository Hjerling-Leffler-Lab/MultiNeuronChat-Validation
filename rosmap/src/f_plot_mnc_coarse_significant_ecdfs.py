import os

import math

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure, format_p_value, tetrad_file_stem, save_single_panel

parser = argparse.ArgumentParser(description='Plot NCI vs AD empirical CDFs of the communication score for every significant tetrad.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_ecdfs_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_ecdfs_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_single_panel_dir_png',
    type=str,
    required=True,
    help='Directory receiving one figure per significant tetrad, for assembling main and supplementary figures.'
)
parser.add_argument(
    '--path_to_single_panel_dir_pdf',
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
    '--n_cols',
    type=int,
    default=3,
)
parser.add_argument(
    '--width_in_cm_per_panel',
    type=float,
    default=4.5,
)
parser.add_argument(
    '--height_in_cm_per_panel',
    type=float,
    default=4.5,
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_ecdfs_png: str = args.path_to_ecdfs_png
path_to_ecdfs_pdf: str = args.path_to_ecdfs_pdf
path_to_single_panel_dir_png: str = args.path_to_single_panel_dir_png
path_to_single_panel_dir_pdf: str = args.path_to_single_panel_dir_pdf

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

n_cols: int = args.n_cols
width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor


def plot_ecdf(ax, mnc_object: MultiNeuronChatObject, source: str, receiver: str, interaction: str, p_adj: float):
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_numpy()
        scores = np.sort(scores[~np.isnan(scores)])
        ecdf = np.arange(1, scores.shape[0] + 1) / scores.shape[0]
        # Start the curve at F=0 so the jump at zero (fraction of donors without communication) is visible
        ax.step(
            np.concatenate([[scores[0]], scores]), np.concatenate([[0], ecdf]),
            where='post', color=condition_colors[condition], linewidth=1.0,
            label=f'{condition} (n={scores.shape[0]})',
        )

    ax.set_ylim(0, 1.02)
    ax.tick_params(axis='both', labelsize=6)
    ax.set_title(
        f'{cell_type_to_short.get(source, source)} → {cell_type_to_short.get(receiver, receiver)}\n{interaction}',
        fontsize=7,
    )
    ax.text(0.97, 0.05, format_p_value(p_adj), transform=ax.transAxes, ha='right', va='bottom', fontsize=6)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    for output_path in [path_to_ecdfs_png, path_to_ecdfs_pdf]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for single_panel_dir in [path_to_single_panel_dir_png, path_to_single_panel_dir_pdf]:
        os.makedirs(single_panel_dir, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    significant: pd.DataFrame = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    if significant.empty:
        save_no_significant_tetrads_figure([path_to_ecdfs_png, path_to_ecdfs_pdf], statistical_test, significance_threshold)
        return

    n_panels = len(significant)
    n_cols_to_use = min(n_cols, n_panels)
    n_rows = math.ceil(n_panels / n_cols_to_use)

    fig, axs = plt.subplots(
        n_rows, n_cols_to_use,
        figsize=(n_cols_to_use * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )
    for ax, row in zip(axs.flat, significant.itertuples()):
        plot_ecdf(ax, mnc_object, row.source, row.receiver, row.interaction, row.p_adj)
    for ax in axs.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('Cumulative fraction of donors', fontsize=7)
    for ax in axs[-1, :]:
        ax.set_xlabel('Communication score', fontsize=7)
    axs.flat[0].legend(fontsize=6, frameon=False, loc='center right')

    fig.tight_layout()
    fig.savefig(path_to_ecdfs_png, dpi=300)
    fig.savefig(path_to_ecdfs_pdf, dpi=300)
    plt.close(fig)

    for row in significant.itertuples():
        fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
        plot_ecdf(ax, mnc_object, row.source, row.receiver, row.interaction, row.p_adj)
        ax.set_ylabel('Cumulative fraction of donors', fontsize=7)
        ax.set_xlabel('Communication score', fontsize=7)
        # Below the curves and above the p-value, the only area the ECDFs never cross
        ax.legend(fontsize=6, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.1), handlelength=1.2)
        save_single_panel(fig, path_to_single_panel_dir_png, path_to_single_panel_dir_pdf, tetrad_file_stem(row.source, row.receiver, row.interaction))


if __name__ == '__main__':
    main()
