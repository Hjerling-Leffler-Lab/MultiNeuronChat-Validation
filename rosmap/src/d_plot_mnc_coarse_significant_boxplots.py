import os

import math

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import ScalarFormatter

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure, format_p_value

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
    default=4,
)
parser.add_argument(
    '--width_in_cm_per_panel',
    type=float,
    default=3.25,
)
parser.add_argument(
    '--height_in_cm_per_panel',
    type=float,
    default=4.6,
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_boxplots_png: str = args.path_to_boxplots_png
path_to_boxplots_pdf: str = args.path_to_boxplots_pdf
path_to_significant_tetrads_csv: str = args.path_to_significant_tetrads_csv

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

n_cols: int = args.n_cols
width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor

def get_scores_per_condition(mnc_object: MultiNeuronChatObject, source: str, receiver: str, interaction: str) -> list[np.ndarray]:
    scores_per_condition: list[np.ndarray] = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_numpy()
        scores_per_condition.append(scores[~np.isnan(scores)])
    return scores_per_condition


def plot_tetrad(ax, mnc_object: MultiNeuronChatObject, scores_per_condition: list[np.ndarray], source: str, receiver: str, interaction: str, p_adj: float):
    rng = np.random.default_rng(0)

    positions = np.array([-0.2, 0.2])
    box = ax.boxplot(
        scores_per_condition,
        positions=positions,
        widths=0.34,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.5},
        capprops={'linewidth': 0.5},
        boxprops={'linewidth': 0.5},
    )
    for patch, condition in zip(box['boxes'], mnc_object.condition_names):
        patch.set_facecolor(condition_colors[condition])
        patch.set_alpha(0.6)

    # Overlay individual donors so the shape of the distribution (which CVM tests) is visible
    for position, scores in zip(positions, scores_per_condition):
        jitter = rng.uniform(-0.09, 0.09, size=scores.shape[0])
        ax.scatter(position + jitter, scores, s=0.8, color='black', alpha=0.4, linewidths=0, zorder=3)

    # Significance bracket above both boxes; the number is the BH-adjusted p-value (stated in the caption)
    y_max = max(scores.max() for scores in scores_per_condition)
    y_min = min(scores.min() for scores in scores_per_condition)
    y_range = y_max - y_min if y_max > y_min else 1.0
    bracket_y = y_max + 0.05 * y_range
    bracket_height = 0.025 * y_range
    ax.plot(
        [positions[0], positions[0], positions[1], positions[1]],
        [bracket_y, bracket_y + bracket_height, bracket_y + bracket_height, bracket_y],
        color='black', linewidth=0.5,
    )
    ax.text(
        positions.mean(), bracket_y + 1.4 * bracket_height, format_p_value(p_adj, prefix=None),
        ha='center', va='bottom', fontsize=6,
    )
    ax.set_ylim(y_min - 0.04 * y_range, y_max + 0.22 * y_range)
    ax.set_xlim(-0.6, 0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(mnc_object.condition_names, fontsize=6)

    # Small scores get a shared x10^k offset instead of long tick labels like 0.0025
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_powerlimits((-2, 3))
    ax.yaxis.set_major_formatter(y_formatter)
    ax.yaxis.get_offset_text().set_fontsize(6)
    ax.tick_params(axis='y', labelsize=6)

    ax.set_title(
        f'{cell_type_to_short.get(source, source)} → {cell_type_to_short.get(receiver, receiver)}\n{interaction}',
        fontsize=6.5,
    )
    ax.spines[['top', 'right']].set_visible(False)


def main():
    for output_path in [path_to_boxplots_png, path_to_boxplots_pdf, path_to_significant_tetrads_csv]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

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
        layout='constrained',
        squeeze=False,
    )
    scores_per_tetrad: list[list[np.ndarray]] = []
    for ax, row in zip(axs.flat, significant.itertuples()):
        scores_per_condition = get_scores_per_condition(mnc_object, row.source, row.receiver, row.interaction)
        scores_per_tetrad.append(scores_per_condition)
        plot_tetrad(ax, mnc_object, scores_per_condition, row.source, row.receiver, row.interaction, row.p_adj)
    for ax in axs.flat[n_panels:]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel('Communication score', fontsize=7)

    # Donor counts go into one shared legend instead of under every panel; show a range if they differ between tetrads
    legend_handles = []
    for condition_index, condition in enumerate(mnc_object.condition_names):
        n_donors = sorted({len(scores[condition_index]) for scores in scores_per_tetrad})
        n_label = f'{n_donors[0]}' if len(n_donors) == 1 else f'{n_donors[0]}–{n_donors[-1]}'
        legend_handles.append(Patch(
            facecolor=condition_colors[condition], alpha=0.6, edgecolor='black', linewidth=0.5,
            label=f'{condition} (n={n_label})',
        ))
    fig.legend(handles=legend_handles, loc='outside lower center', ncols=len(legend_handles), fontsize=6, frameon=False)

    fig.savefig(path_to_boxplots_png, dpi=300)
    fig.savefig(path_to_boxplots_pdf, dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
