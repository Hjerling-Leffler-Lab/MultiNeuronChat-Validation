import os

from typing import Callable

import numpy as np
import pandas as pd

import loompy

from scipy import stats

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure, format_p_value, tetrad_file_stem, save_single_panel

parser = argparse.ArgumentParser(description='Relate the communication score of every significant tetrad to disease severity and potential confounders.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_clinical_data',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_subject_wise_normalized_loom',
    type=str,
    required=True,
    help='Only the column attributes are read, to count cells per donor and cell type.'
)
parser.add_argument(
    '--cell_type_label_column',
    type=str,
    default='grouping.by',
    help='Column attribute holding the cell-type labels MNC was run on (grouping.by = coarse, cell.type = fine).'
)
parser.add_argument(
    '--path_to_severity_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_severity_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_confounders_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_confounders_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_single_panel_dir_png',
    type=str,
    required=True,
    help='Directory receiving one figure per significant tetrad and clinical variable, for assembling main and supplementary figures.'
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
path_to_clinical_data: str = args.path_to_clinical_data
path_to_subject_wise_normalized_loom: str = args.path_to_subject_wise_normalized_loom

path_to_severity_png: str = args.path_to_severity_png
path_to_severity_pdf: str = args.path_to_severity_pdf
path_to_confounders_png: str = args.path_to_confounders_png
path_to_confounders_pdf: str = args.path_to_confounders_pdf
path_to_single_panel_dir_png: str = args.path_to_single_panel_dir_png
path_to_single_panel_dir_pdf: str = args.path_to_single_panel_dir_pdf

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold

width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

cm: float = 1/2.54  # cm to inches conversion factor

subject_label_column: str = 'individualID'
cell_type_label_column: str = args.cell_type_label_column

# ROSMAP codings: CERAD 1 = definite AD ... 4 = no AD, msex 1 = male
braak_labels: dict[int, str] = {i: str(i) for i in range(7)}
cerad_labels: dict[int, str] = {1: 'Definite', 2: 'Probable', 3: 'Possible', 4: 'No AD'}
sex_labels: dict[int, str] = {0: 'Female', 1: 'Male'}


def load_clinical_data() -> pd.DataFrame:
    clinical_data: pd.DataFrame = pd.read_csv(path_to_clinical_data)
    clinical_data = clinical_data.drop_duplicates(subset=subject_label_column).set_index(subject_label_column)
    # age_death is censored at 90 and stored as '90+'
    clinical_data['age_death'] = pd.to_numeric(clinical_data['age_death'].astype(str).str.replace('+', '', regex=False), errors='coerce')
    return clinical_data


def load_cell_counts() -> pd.DataFrame:
    """Number of cells per donor (rows) and cell type (columns) in the data MNC was run on."""
    with loompy.connect(path_to_subject_wise_normalized_loom, 'r') as ds:
        cells = pd.DataFrame({
            'subject': ds.ca[subject_label_column],
            'cell_type': ds.ca[cell_type_label_column],
        })
    return cells.groupby(['subject', 'cell_type']).size().unstack(fill_value=0)


def get_scores(mnc_object: MultiNeuronChatObject, source: str, receiver: str, interaction: str) -> pd.DataFrame:
    frames = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_series().dropna()
        frames.append(pd.DataFrame({'score': scores, 'condition': condition}))
    return pd.concat(frames)


def scatter_by_condition(ax, df: pd.DataFrame, x_column: str, condition_names, jitter: float = 0.0):
    rng = np.random.default_rng(0)
    for condition in condition_names:
        condition_df = df[df['condition'] == condition]
        x = condition_df[x_column].to_numpy(dtype=float) + rng.uniform(-jitter, jitter, size=len(condition_df))
        ax.scatter(x, condition_df['score'], s=2, color=condition_colors[condition], alpha=0.7, linewidths=0, label=condition, zorder=3)


def boxes_per_level(ax, df: pd.DataFrame, x_column: str, levels: dict[int, str]):
    present = [level for level in levels if (df[x_column] == level).any()]
    ax.boxplot(
        [df.loc[df[x_column] == level, 'score'] for level in present],
        positions=present,
        widths=0.6,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.6},
        capprops={'linewidth': 0.6},
        boxprops={'linewidth': 0.6},
    )
    ax.set_xticks(list(levels))
    ax.set_xticklabels(list(levels.values()), fontsize=6, rotation=45 if max(len(v) for v in levels.values()) > 2 else 0)


def spearman_text(df: pd.DataFrame, x_column: str, label: str) -> str:
    df = df.dropna(subset=[x_column, 'score'])
    if len(df) < 3:
        return f'{label}: n<3'
    rho, p = stats.spearmanr(df[x_column], df['score'])
    return f'{label}: ρ = {rho:.2f}, {format_p_value(p, prefix="p")}'


def annotate(ax, lines: list[str]):
    ax.text(0.02, 0.98, '\n'.join(lines), transform=ax.transAxes, ha='left', va='top', fontsize=5)


def style_axis(ax, x_label: str):
    ax.set_xlabel(x_label, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.spines[['top', 'right']].set_visible(False)


def get_severity_panels(df: pd.DataFrame, condition_names) -> list[tuple[str, Callable]]:
    """(name, drawing function) of the Braak, CERAD and MMSE panel of one tetrad."""
    # Severity is confounded with diagnosis, so the within-AD trend is reported next to the pooled one
    ad_df = df[df['condition'] == condition_names[1]]

    def braak(ax):
        boxes_per_level(ax, df, 'braaksc', braak_labels)
        scatter_by_condition(ax, df, 'braaksc', condition_names, jitter=0.2)
        annotate(ax, [spearman_text(df, 'braaksc', 'All'), spearman_text(ad_df, 'braaksc', condition_names[1])])
        style_axis(ax, 'Braak stage')

    def cerad(ax):
        boxes_per_level(ax, df, 'ceradsc', cerad_labels)
        scatter_by_condition(ax, df, 'ceradsc', condition_names, jitter=0.2)
        annotate(ax, [spearman_text(df, 'ceradsc', 'All'), spearman_text(ad_df, 'ceradsc', condition_names[1])])
        style_axis(ax, 'CERAD score')

    def mmse(ax):
        scatter_by_condition(ax, df, 'cts_mmse30_lv', condition_names)
        annotate(ax, [spearman_text(df, 'cts_mmse30_lv', 'All'), spearman_text(ad_df, 'cts_mmse30_lv', condition_names[1])])
        style_axis(ax, 'MMSE (last visit)')

    return [('braak', braak), ('cerad', cerad), ('mmse', mmse)]


def get_confounder_panels(df: pd.DataFrame, condition_names, source_short: str, receiver_short: str) -> list[tuple[str, Callable]]:
    """(name, drawing function) of the age, PMI, sex and cell-count panels of one tetrad."""
    # Confounders are assessed within each condition: a pooled correlation would only re-detect the group difference
    def per_condition_spearman(x_column: str) -> list[str]:
        return [spearman_text(df[df['condition'] == c], x_column, c) for c in condition_names]

    def age(ax):
        scatter_by_condition(ax, df, 'age_death', condition_names, jitter=0.2)
        annotate(ax, per_condition_spearman('age_death'))
        style_axis(ax, 'Age at death (90 = 90+)')

    def pmi(ax):
        scatter_by_condition(ax, df, 'pmi', condition_names)
        annotate(ax, per_condition_spearman('pmi'))
        style_axis(ax, 'Post-mortem interval (h)')

    def sex(ax):
        boxes_per_level(ax, df, 'msex', sex_labels)
        scatter_by_condition(ax, df, 'msex', condition_names, jitter=0.2)
        sex_lines = []
        for condition in condition_names:
            condition_df = df[df['condition'] == condition]
            female, male = condition_df.loc[condition_df['msex'] == 0, 'score'], condition_df.loc[condition_df['msex'] == 1, 'score']
            if len(female) > 0 and len(male) > 0:
                sex_lines.append(f'{condition}: MWU {format_p_value(stats.mannwhitneyu(female, male).pvalue, prefix="p")}')
        annotate(ax, sex_lines)
        style_axis(ax, 'Sex')

    def n_source_cells(ax):
        scatter_by_condition(ax, df, 'n_source_cells', condition_names)
        annotate(ax, per_condition_spearman('n_source_cells'))
        ax.set_xscale('log')
        style_axis(ax, f'# {source_short} cells')

    def n_receiver_cells(ax):
        scatter_by_condition(ax, df, 'n_receiver_cells', condition_names)
        annotate(ax, per_condition_spearman('n_receiver_cells'))
        ax.set_xscale('log')
        style_axis(ax, f'# {receiver_short} cells')

    return [('age', age), ('pmi', pmi), ('sex', sex), ('n_source_cells', n_source_cells), ('n_receiver_cells', n_receiver_cells)]


def main():
    output_paths = [path_to_severity_png, path_to_severity_pdf, path_to_confounders_png, path_to_confounders_pdf]
    for output_path in output_paths:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    for single_panel_dir in [path_to_single_panel_dir_png, path_to_single_panel_dir_pdf]:
        os.makedirs(single_panel_dir, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    significant: pd.DataFrame = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    if significant.empty:
        save_no_significant_tetrads_figure(output_paths, statistical_test, significance_threshold)
        return
    condition_names = mnc_object.condition_names

    clinical_data = load_clinical_data()
    cell_counts = load_cell_counts()

    n_rows = len(significant)
    severity_fig, severity_axs = plt.subplots(
        n_rows, 3,
        figsize=(3 * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )
    confounders_fig, confounders_axs = plt.subplots(
        n_rows, 5,
        figsize=(5 * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )

    for severity_row_axs, confounders_row_axs, row in zip(severity_axs, confounders_axs, significant.itertuples()):
        source_short = cell_type_to_short.get(row.source, row.source)
        receiver_short = cell_type_to_short.get(row.receiver, row.receiver)

        df = get_scores(mnc_object, row.source, row.receiver, row.interaction)
        df = df.join(clinical_data[['braaksc', 'ceradsc', 'cts_mmse30_lv', 'age_death', 'pmi', 'msex']], how='left')
        df['n_source_cells'] = cell_counts[row.source].reindex(df.index).to_numpy()
        df['n_receiver_cells'] = cell_counts[row.receiver].reindex(df.index).to_numpy()

        n_missing_clinical = df['braaksc'].isna().sum()
        if n_missing_clinical > 0:
            print(f'Warning: {n_missing_clinical} donors without clinical data for {row.source} -> {row.receiver} {row.interaction}')

        severity_panels = get_severity_panels(df, condition_names)
        confounder_panels = get_confounder_panels(df, condition_names, source_short, receiver_short)
        for ax, (_, draw) in zip(severity_row_axs, severity_panels):
            draw(ax)
        for ax, (_, draw) in zip(confounders_row_axs, confounder_panels):
            draw(ax)

        row_label = f'{source_short} → {receiver_short}\n{row.interaction}\n{format_p_value(row.p_adj)}'
        severity_row_axs[0].set_ylabel(f'{row_label}\n\nCommunication score', fontsize=6)
        confounders_row_axs[0].set_ylabel(f'{row_label}\n\nCommunication score', fontsize=6)

        file_stem = tetrad_file_stem(row.source, row.receiver, row.interaction)
        for name, draw in severity_panels + confounder_panels:
            fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
            draw(ax)
            ax.set_title(f'{source_short} → {receiver_short}\n{row.interaction}', fontsize=7)
            ax.set_ylabel('Communication score', fontsize=7)
            # Headroom so the statistics text does not cover donors; no legend, the NCI/AD colours match the boxplot and ECDF panels
            y_min, y_max = ax.get_ylim()
            ax.set_ylim(y_min, y_max + 0.3 * (y_max - y_min))
            save_single_panel(fig, path_to_single_panel_dir_png, path_to_single_panel_dir_pdf, f'{file_stem}__{name}')

    severity_axs.flat[-1].legend(fontsize=6, frameon=False, loc='lower right', markerscale=2)
    confounders_axs.flat[-1].legend(fontsize=6, frameon=False, loc='lower right', markerscale=2)

    for fig, png, pdf in [
        (severity_fig, path_to_severity_png, path_to_severity_pdf),
        (confounders_fig, path_to_confounders_png, path_to_confounders_pdf),
    ]:
        fig.tight_layout()
        fig.savefig(png, dpi=300)
        fig.savefig(pdf, dpi=300)
        plt.close(fig)


if __name__ == '__main__':
    main()
