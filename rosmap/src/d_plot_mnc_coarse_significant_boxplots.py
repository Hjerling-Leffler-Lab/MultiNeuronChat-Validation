import os

import math

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure, format_p_value, tetrad_file_stem, save_single_panel

parser = argparse.ArgumentParser(description='Plot NCI vs AD communication-score boxplots for every significant tetrad.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_boxplots_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_boxplots_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_significant_tetrads_csv',
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
    default=5.0,
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_boxplots_png: str = args.path_to_boxplots_png
path_to_boxplots_pdf: str = args.path_to_boxplots_pdf
path_to_significant_tetrads_csv: str = args.path_to_significant_tetrads_csv
path_to_single_panel_dir_png: str = args.path_to_single_panel_dir_png
path_to_single_panel_dir_pdf: str = args.path_to_single_panel_dir_pdf

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

n_cols: int = args.n_cols
width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor

def plot_tetrad(ax, mnc_object: MultiNeuronChatObject, source: str, receiver: str, interaction: str, p_adj: float):
    rng = np.random.default_rng(0)

    scores_per_condition: list[np.ndarray] = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_numpy()
        scores_per_condition.append(scores[~np.isnan(scores)])

    positions = np.arange(len(scores_per_condition))
    box = ax.boxplot(
        scores_per_condition,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.6},
        capprops={'linewidth': 0.6},
        boxprops={'linewidth': 0.6},
    )
    for patch, condition in zip(box['boxes'], mnc_object.condition_names):
        patch.set_facecolor(condition_colors[condition])
        patch.set_alpha(0.6)

    # Overlay individual donors so the shape of the distribution (which CVM tests) is visible
    for position, scores in zip(positions, scores_per_condition):
        jitter = rng.uniform(-0.18, 0.18, size=scores.shape[0])
        ax.scatter(position + jitter, scores, s=1.5, color='black', alpha=0.5, linewidths=0, zorder=3)

    # Significance bracket above both boxes
    y_max = max(scores.max() for scores in scores_per_condition)
    y_min = min(scores.min() for scores in scores_per_condition)
    y_range = y_max - y_min if y_max > y_min else 1.0
    bracket_y = y_max + 0.05 * y_range
    bracket_height = 0.03 * y_range
    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color='black', linewidth=0.6,
    )
    ax.text(
        positions.mean(), bracket_y + 1.5 * bracket_height, format_p_value(p_adj),
        ha='center', va='bottom', fontsize=6,
    )
    ax.set_ylim(y_min - 0.05 * y_range, bracket_y + 0.25 * y_range)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f'{condition}\n(n={len(scores)})' for condition, scores in zip(mnc_object.condition_names, scores_per_condition)],
        fontsize=6,
    )
    ax.tick_params(axis='y', labelsize=6)
    ax.set_title(
        f'{cell_type_to_short.get(source, source)} → {cell_type_to_short.get(receiver, receiver)}\n{interaction}',
        fontsize=7,
    )
    ax.spines[['top', 'right']].set_visible(False)


def main():
    for output_path in [path_to_boxplots_png, path_to_boxplots_pdf, path_to_significant_tetrads_csv]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for single_panel_dir in [path_to_single_panel_dir_png, path_to_single_panel_dir_pdf]:
        os.makedirs(single_panel_dir, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)

    significant: pd.DataFrame = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    significant.to_csv(path_to_significant_tetrads_csv, index=False)
    if significant.empty:
        save_no_significant_tetrads_figure([path_to_boxplots_png, path_to_boxplots_pdf], statistical_test, significance_threshold)
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
        plot_tetrad(ax, mnc_object, row.source, row.receiver, row.interaction, row.p_adj)
    for ax in axs.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('Communication score', fontsize=7)

    fig.tight_layout()
    fig.savefig(path_to_boxplots_png, dpi=300)
    fig.savefig(path_to_boxplots_pdf, dpi=300)
    plt.close(fig)

    for row in significant.itertuples():
        fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
        plot_tetrad(ax, mnc_object, row.source, row.receiver, row.interaction, row.p_adj)
        ax.set_ylabel('Communication score', fontsize=7)
        save_single_panel(fig, path_to_single_panel_dir_png, path_to_single_panel_dir_pdf, tetrad_file_stem(row.source, row.receiver, row.interaction))


if __name__ == '__main__':
    main()
