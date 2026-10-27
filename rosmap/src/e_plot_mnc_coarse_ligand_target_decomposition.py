import os

import numpy as np
import pandas as pd
from scipy import stats

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

# Embed fonts as TrueType so the text stays editable in Illustrator
plt.rcParams['pdf.fonttype'] = 42

import argparse

from plotting.utils import cell_type_to_short, condition_colors, format_p_value, get_significant_tetrads, save_no_significant_tetrads_figure, tetrad_file_stem, save_single_panel

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
    help='Directory receiving one ligand, target and score figure per significant tetrad, for assembling figures panel by panel.'
)
parser.add_argument(
    '--path_to_single_panel_dir_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_ligand_target_tests_csv',
    type=str,
    required=True,
    help='Raw and BH-adjusted p-values shown on the ligand and target panels.'
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
path_to_ligand_target_tests_csv: str = args.path_to_ligand_target_tests_csv

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor


def plot_condition_boxes(ax, values_per_condition: list[np.ndarray], condition_names: tuple[str, str], title: str, p_adj: float):
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

    # Bracket above both boxes, labelled with the BH-adjusted p-value (stated in the caption)
    if not np.isnan(p_adj):
        y_max = max(values.max() for values in values_per_condition)
        y_min = min(values.min() for values in values_per_condition)
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


def compare_conditions(values_per_condition: list[np.ndarray]) -> float:
    """Nominal p-value of the MNC run's test between the two conditions, applied to ligand or target abundance as MNC applies it to the score."""
    condition_a_values, condition_b_values = values_per_condition
    # Like MNC, skip the test if a condition has fewer than two donors or all values are identical
    if len(condition_a_values) < 2 or len(condition_b_values) < 2 or np.unique(np.hstack(values_per_condition)).shape[0] == 1:
        return np.nan
    if statistical_test == 'KS':
        return stats.ks_2samp(condition_a_values, condition_b_values).pvalue
    if statistical_test == 'CVM':
        return stats.cramervonmises_2samp(condition_a_values, condition_b_values, method='auto').pvalue
    if statistical_test == 'MannWhitneyU':
        return stats.mannwhitneyu(condition_a_values, condition_b_values, alternative='two-sided').pvalue
    raise ValueError(f'Unknown statistical test: {statistical_test}')


def get_abundance_values(mnc_object: MultiNeuronChatObject, side: str, cell_type: str, interaction: str) -> list[np.ndarray]:
    """Per-condition ligand abundance of a source, or target abundance of a receiver, for one interaction."""
    abundance_dict = mnc_object.ligand_abundance_dict if side == 'ligand' else mnc_object.target_abundance_dict
    return [drop_nan(abundance_dict[c].loc[:, cell_type, interaction].to_numpy()) for c in mnc_object.condition_names]


def test_ligands_and_targets(mnc_object: MultiNeuronChatObject, significant: pd.DataFrame) -> pd.DataFrame:
    """Test every distinct ligand and target abundance shown in the figure and BH-adjust across them.

    Tetrads sharing a source and interaction share their ligand test (likewise for receiver and target), so each test
    enters the correction once rather than once per tetrad.
    """
    tests = pd.concat([
        significant[['source', 'interaction']].rename(columns={'source': 'cell_type'}).assign(side='ligand'),
        significant[['receiver', 'interaction']].rename(columns={'receiver': 'cell_type'}).assign(side='target'),
    ]).drop_duplicates()[['side', 'cell_type', 'interaction']].reset_index(drop=True)
    tests['p'] = [
        compare_conditions(get_abundance_values(mnc_object, test.side, test.cell_type, test.interaction))
        for test in tests.itertuples()
    ]
    tests['p_adj'] = np.nan
    tested = tests['p'].notna()
    tests.loc[tested, 'p_adj'] = stats.false_discovery_control(tests.loc[tested, 'p'], method='bh')
    tests.insert(3, 'statistical_test', statistical_test)
    return tests


def get_decomposition_panels(mnc_object: MultiNeuronChatObject, row, tests: pd.DataFrame) -> list[tuple[str, list[np.ndarray], str, str, float]]:
    """(name, values per condition, title, y label, adjusted p-value) of the ligand, target and score panel of one tetrad."""
    condition_names = mnc_object.condition_names
    source_short = cell_type_to_short.get(row.source, row.source)
    receiver_short = cell_type_to_short.get(row.receiver, row.receiver)
    p_adj_per_test = tests.set_index(['side', 'cell_type', 'interaction'])['p_adj']
    score_values = [drop_nan(mnc_object.communication_score_dict[c].loc[:, row.source, row.receiver, row.interaction].to_numpy()) for c in condition_names]
    return [
        (
            'ligand', get_abundance_values(mnc_object, 'ligand', row.source, row.interaction),
            f'{row.interaction}\nLigand in {source_short}', 'Abundance', p_adj_per_test['ligand', row.source, row.interaction],
        ),
        (
            'target', get_abundance_values(mnc_object, 'target', row.receiver, row.interaction),
            f'{row.interaction}\nTarget in {receiver_short}', 'Abundance', p_adj_per_test['target', row.receiver, row.interaction],
        ),
        ('score', score_values, f'{row.interaction}\nScore {source_short} → {receiver_short}', 'Communication score', row.p_adj),
    ]


def main():
    for output_path in [path_to_decomposition_png, path_to_decomposition_pdf, path_to_ligand_target_tests_csv]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for single_panel_dir in [path_to_single_panel_dir_png, path_to_single_panel_dir_pdf]:
        os.makedirs(single_panel_dir, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    significant: pd.DataFrame = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    if significant.empty:
        save_no_significant_tetrads_figure([path_to_decomposition_png, path_to_decomposition_pdf], statistical_test, significance_threshold)
        pd.DataFrame(columns=['side', 'cell_type', 'interaction', 'statistical_test', 'p', 'p_adj']).to_csv(path_to_ligand_target_tests_csv, index=False)
        return
    condition_names = mnc_object.condition_names

    tests: pd.DataFrame = test_ligands_and_targets(mnc_object, significant)
    tests.to_csv(path_to_ligand_target_tests_csv, index=False)

    n_rows = len(significant)
    fig, axs = plt.subplots(
        n_rows, 3,
        figsize=(3 * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )

    for row_axs, row in zip(axs, significant.itertuples()):
        for ax, (_, values, title, _, p_adj) in zip(row_axs, get_decomposition_panels(mnc_object, row, tests)):
            plot_condition_boxes(ax, values, condition_names, title, p_adj)

        row_axs[0].set_ylabel('Abundance', fontsize=7)
        row_axs[2].set_ylabel('Communication score', fontsize=7)

    fig.tight_layout()
    fig.savefig(path_to_decomposition_png, dpi=300)
    fig.savefig(path_to_decomposition_pdf, dpi=300)
    plt.close(fig)

    # Every panel again as its own figure, named <source>__<receiver>__<interaction>__<ligand|target|score>
    for row in significant.itertuples():
        file_stem = tetrad_file_stem(row.source, row.receiver, row.interaction)
        for name, values, title, y_label, p_adj in get_decomposition_panels(mnc_object, row, tests):
            fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
            plot_condition_boxes(ax, values, condition_names, title, p_adj)
            ax.set_ylabel(y_label, fontsize=7)
            save_single_panel(fig, path_to_single_panel_dir_png, path_to_single_panel_dir_pdf, f'{file_stem}__{name}')


if __name__ == '__main__':
    main()
