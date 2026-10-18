import os

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure, format_p_value, tetrad_file_stem, save_single_panel

parser = argparse.ArgumentParser(description='Decompose the communication score of every significant tetrad into ligand and target abundance.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_decomposition_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_decomposition_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_single_panel_dir_png',
    type=str,
    required=True,
    help='Directory receiving one ligand, target and score figure per significant tetrad, for assembling main and supplementary figures.'
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
path_to_decomposition_png: str = args.path_to_decomposition_png
path_to_decomposition_pdf: str = args.path_to_decomposition_pdf
path_to_single_panel_dir_png: str = args.path_to_single_panel_dir_png
path_to_single_panel_dir_pdf: str = args.path_to_single_panel_dir_pdf

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor


def plot_condition_boxes(ax, values_per_condition: list[np.ndarray], condition_names: tuple[str, str], title: str):
    rng = np.random.default_rng(0)
    positions = np.arange(len(values_per_condition))

    box = ax.boxplot(
        values_per_condition,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.6},
        capprops={'linewidth': 0.6},
        boxprops={'linewidth': 0.6},
    )
    for patch, condition in zip(box['boxes'], condition_names):
        patch.set_facecolor(condition_colors[condition])
        patch.set_alpha(0.6)

    for position, values in zip(positions, values_per_condition):
        jitter = rng.uniform(-0.18, 0.18, size=values.shape[0])
        ax.scatter(position + jitter, values, s=1.5, color='black', alpha=0.5, linewidths=0, zorder=3)

    # The fraction of donors at exactly zero is what separates the conditions for zero-inflated tetrads
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f'{condition}\n{np.mean(values == 0) * 100:.0f}% zero' for condition, values in zip(condition_names, values_per_condition)],
        fontsize=6,
    )
    ax.tick_params(axis='y', labelsize=6)
    ax.set_title(title, fontsize=7)
    ax.spines[['top', 'right']].set_visible(False)


def drop_nan(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def get_decomposition_panels(mnc_object: MultiNeuronChatObject, row) -> list[tuple[str, list[np.ndarray], str, str]]:
    """(name, values per condition, title, y label) of the ligand, target and score panel of one tetrad."""
    condition_names = mnc_object.condition_names
    source_short = cell_type_to_short.get(row.source, row.source)
    receiver_short = cell_type_to_short.get(row.receiver, row.receiver)
    ligand_values = [drop_nan(mnc_object.ligand_abundance_dict[c].loc[:, row.source, row.interaction].to_numpy()) for c in condition_names]
    target_values = [drop_nan(mnc_object.target_abundance_dict[c].loc[:, row.receiver, row.interaction].to_numpy()) for c in condition_names]
    score_values = [drop_nan(mnc_object.communication_score_dict[c].loc[:, row.source, row.receiver, row.interaction].to_numpy()) for c in condition_names]
    return [
        ('ligand', ligand_values, f'{row.interaction}\nLigand in {source_short}', 'Abundance'),
        ('target', target_values, f'{row.interaction}\nTarget in {receiver_short}', 'Abundance'),
        ('score', score_values, f'Score {source_short} → {receiver_short}\n{format_p_value(row.p_adj)}', 'Communication score'),
    ]


def main():
    for output_path in [path_to_decomposition_png, path_to_decomposition_pdf]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for single_panel_dir in [path_to_single_panel_dir_png, path_to_single_panel_dir_pdf]:
        os.makedirs(single_panel_dir, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    significant: pd.DataFrame = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    if significant.empty:
        save_no_significant_tetrads_figure([path_to_decomposition_png, path_to_decomposition_pdf], statistical_test, significance_threshold)
        return
    condition_names = mnc_object.condition_names

    n_rows = len(significant)
    fig, axs = plt.subplots(
        n_rows, 3,
        figsize=(3 * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )

    for row_axs, row in zip(axs, significant.itertuples()):
        for ax, (_, values, title, _) in zip(row_axs, get_decomposition_panels(mnc_object, row)):
            plot_condition_boxes(ax, values, condition_names, title)

        row_axs[0].set_ylabel('Abundance', fontsize=7)
        row_axs[2].set_ylabel('Communication score', fontsize=7)

    fig.tight_layout()
    fig.savefig(path_to_decomposition_png, dpi=300)
    fig.savefig(path_to_decomposition_pdf, dpi=300)
    plt.close(fig)

    for row in significant.itertuples():
        file_stem = tetrad_file_stem(row.source, row.receiver, row.interaction)
        for name, values, title, y_label in get_decomposition_panels(mnc_object, row):
            fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
            plot_condition_boxes(ax, values, condition_names, title)
            ax.set_ylabel(y_label, fontsize=7)
            save_single_panel(fig, path_to_single_panel_dir_png, path_to_single_panel_dir_pdf, f'{file_stem}__{name}')


if __name__ == '__main__':
    main()
