import os

import numpy as np
import pandas as pd
import xarray as xr

from scipy.cluster import hierarchy

from multineuronchat import MultiNeuronChatObject

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import argparse

from plotting.utils import cell_type_to_short, get_significant_tetrads, save_no_significant_tetrads_figure

parser = argparse.ArgumentParser(description='Hierarchically cluster the donors of one condition and the tetrads on their communication scores and plot them as a heatmap.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_clustermap_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_clustermap_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_donor_order_csv',
    type=str,
    required=True,
)
parser.add_argument(
    '--statistical_test',
    type=str,
    required=True,
)
parser.add_argument(
    '--selection',
    type=str,
    required=True,
    help="Which tetrads to show: 'significant' (adj. p < significance_threshold) or 'top_<N>' (N smallest raw p-values)."
)
parser.add_argument(
    '--significance_threshold',
    type=float,
    default=0.1,
)
parser.add_argument(
    '--condition',
    type=str,
    default='AD',
    help='Condition whose donors are clustered; z-scores are computed within this condition.'
)
parser.add_argument(
    '--path_to_clinical_data',
    type=str,
    default=None,
    help='Optional ROSMAP clinical table; if given, Braak, CERAD, MMSE, age and sex are shown as colour bars above the donors.'
)
parser.add_argument(
    '--width_in_cm',
    type=float,
    default=18.0,
)
parser.add_argument(
    '--height_in_cm_per_tetrad',
    type=float,
    default=0.45,
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_clustermap_png: str = args.path_to_clustermap_png
path_to_clustermap_pdf: str = args.path_to_clustermap_pdf
path_to_donor_order_csv: str = args.path_to_donor_order_csv

statistical_test: str = args.statistical_test
selection: str = args.selection
significance_threshold: float = args.significance_threshold
condition: str = args.condition
path_to_clinical_data: str | None = args.path_to_clinical_data

width_in_cm: float = args.width_in_cm
height_in_cm_per_tetrad: float = args.height_in_cm_per_tetrad

cm: float = 1/2.54  # cm to inches conversion factor

# z-scores beyond this are clipped in the colour scale only, not in the clustering
z_score_limit: float = 3.0

subject_label_column: str = 'individualID'

# Clinical covariates shown above the donors: column -> (label, colour map); msex is categorical (1 = male)
clinical_covariates: dict[str, tuple[str, str]] = {
    'braaksc': ('Braak', 'Purples'),
    'ceradsc': ('CERAD', 'Purples_r'),  # 1 = definite AD ... 4 = no AD, so reversed to make darker = more pathology
    'cts_mmse30_lv': ('MMSE', 'Greens'),
    'age_death': ('Age', 'Greys'),
    'msex': ('Sex', None),
}
sex_colors: dict[int, str] = {0: '#E5A3A3', 1: '#8DB3D6'}


def select_tetrads(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    """Return (source, receiver, interaction, p, p_adj) of the tetrads to show, sorted by raw p-value."""
    p_values = mnc_object.p_values[statistical_test].to_dataframe('p').dropna()
    p_values_adj = mnc_object.p_values_adj[statistical_test].to_dataframe('p_adj').dropna()
    tetrads = p_values.join(p_values_adj).sort_values(['p', 'p_adj']).reset_index()

    if selection == 'significant':
        significant = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
        keys = ['source', 'receiver', 'interaction']
        return tetrads.merge(significant[keys], on=keys)
    if selection.startswith('top_'):
        return tetrads.head(int(selection.removeprefix('top_')))
    raise ValueError(f"Unknown selection '{selection}', expected 'significant' or 'top_<N>'")


def tetrad_label(source: str, receiver: str, interaction: str) -> str:
    short = lambda cell_type: cell_type_to_short.get(cell_type, cell_type).replace(' \n ', ' ')
    return f'{short(source)} → {short(receiver)}: {interaction}'


def get_score_matrix(mnc_object: MultiNeuronChatObject, tetrads: pd.DataFrame) -> pd.DataFrame:
    """Communication scores of the donors of the selected condition as a tetrads x donors matrix."""
    scores: xr.DataArray = mnc_object.communication_score_dict[condition]
    rows = {
        tetrad_label(row.source, row.receiver, row.interaction): scores.loc[:, row.source, row.receiver, row.interaction].to_numpy()
        for row in tetrads.itertuples()
    }
    return pd.DataFrame(rows, index=scores['subject'].to_numpy()).T


def load_clinical_data(subjects: pd.Index) -> pd.DataFrame:
    """Clinical covariates of the given donors, numeric, NaN where unknown."""
    clinical_data: pd.DataFrame = pd.read_csv(path_to_clinical_data)
    clinical_data = clinical_data.drop_duplicates(subset=subject_label_column).set_index(subject_label_column)
    # age_death is censored at 90 and stored as '90+'
    clinical_data['age_death'] = clinical_data['age_death'].astype(str).str.replace('+', '', regex=False)
    clinical_data = clinical_data[list(clinical_covariates)].apply(pd.to_numeric, errors='coerce')
    return clinical_data.reindex(subjects)


def get_clinical_colors(clinical_data: pd.DataFrame) -> tuple[pd.DataFrame, list[Patch]]:
    """One colour bar per covariate (white = unknown), plus legend handles with the lowest and highest value."""
    colors = pd.DataFrame(index=clinical_data.index)
    handles = []
    for column, (label, cmap_name) in clinical_covariates.items():
        values = clinical_data[column]
        if cmap_name is None:
            colors[label] = values.map(sex_colors).fillna('white')
            handles += [Patch(color=sex_colors[0], label='Female'), Patch(color=sex_colors[1], label='Male')]
            continue
        cmap = plt.get_cmap(cmap_name)
        low, high = values.min(), values.max()
        scaled = (values - low) / (high - low) if high > low else values * 0.0
        colors[label] = [matplotlib.colors.to_hex(cmap(0.15 + 0.85 * value)) if pd.notna(value) else 'white' for value in scaled]
        handles += [Patch(color=cmap(0.15), label=f'{label} {low:g}'), Patch(color=cmap(1.0), label=f'{label} {high:g}')]
    return colors, handles


def main():
    for output_path in [path_to_clustermap_png, path_to_clustermap_pdf, path_to_donor_order_csv]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)
    if condition not in mnc_object.condition_names:
        raise ValueError(f"Unknown condition '{condition}', expected one of {list(mnc_object.condition_names)}")

    tetrads: pd.DataFrame = select_tetrads(mnc_object)
    if tetrads.empty:
        save_no_significant_tetrads_figure([path_to_clustermap_png, path_to_clustermap_pdf], statistical_test, significance_threshold)
        pd.DataFrame(columns=['subject', 'dendrogram_position']).to_csv(path_to_donor_order_csv, index=False)
        return

    scores: pd.DataFrame = get_score_matrix(mnc_object, tetrads)

    # Scores differ by orders of magnitude between tetrads, so standardise every tetrad across the donors of the condition.
    # Donors lacking a cell type have no score; they are set to the tetrad mean (z = 0) for clustering and drawn grey.
    z_scores = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1), axis=0)
    missing = z_scores.isna()
    z_scores_filled = z_scores.fillna(0.0)

    donor_linkage = hierarchy.linkage(z_scores_filled.T.to_numpy(), method='ward')
    tetrad_linkage = hierarchy.linkage(z_scores_filled.to_numpy(), method='ward') if len(z_scores) > 1 else None

    # Donors in the left-to-right order of the dendrogram leaves
    donor_order = z_scores.columns[hierarchy.leaves_list(donor_linkage)]
    donor_table = pd.DataFrame({'subject': donor_order, 'dendrogram_position': np.arange(len(donor_order))})

    col_colors = None
    legend_handles: list[Patch] = []
    if path_to_clinical_data is not None:
        clinical_data = load_clinical_data(donor_order)
        donor_table = donor_table.join(clinical_data.reset_index(drop=True))
        col_colors, legend_handles = get_clinical_colors(clinical_data)

    donor_table.to_csv(path_to_donor_order_csv, index=False)

    height_in_cm = 5.0 + height_in_cm_per_tetrad * len(z_scores) + (0.3 * col_colors.shape[1] if col_colors is not None else 0.0)
    grid = sns.clustermap(
        z_scores_filled,
        mask=missing,
        row_linkage=tetrad_linkage,
        col_linkage=donor_linkage,
        row_cluster=tetrad_linkage is not None,
        col_colors=col_colors,
        cmap='RdBu_r',
        center=0,
        vmin=-z_score_limit,
        vmax=z_score_limit,
        xticklabels=False,
        yticklabels=True,
        figsize=(width_in_cm * cm, height_in_cm * cm),
        dendrogram_ratio=(0.08, 2.0 / height_in_cm),
        colors_ratio=(0.02, 0.3 / height_in_cm),
        cbar_pos=(0.01, 0.82, 0.015, 0.12),
        tree_kws={'linewidths': 0.4},
    )
    grid.ax_heatmap.set_facecolor('lightgrey')
    grid.ax_heatmap.set_xlabel(f'{condition} donors (n = {z_scores.shape[1]})', fontsize=7)
    grid.ax_heatmap.set_ylabel('')
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0)
    if col_colors is not None:
        grid.ax_col_colors.tick_params(axis='y', labelsize=6, length=0)

    grid.ax_cbar.set_yticks([-z_score_limit, 0, z_score_limit])
    grid.ax_cbar.set_yticklabels([f'≤ {-z_score_limit:g}', '0', f'≥ {z_score_limit:g}'])
    grid.ax_cbar.tick_params(labelsize=5)
    grid.ax_cbar.set_title('z-score', fontsize=6)

    if legend_handles:
        grid.figure.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.0, 0.8), fontsize=5, frameon=False)  # left of the tetrad dendrogram

    selection_label = f'adj. p < {significance_threshold}' if selection == 'significant' else f'top {len(z_scores)} by p-value'
    grid.figure.suptitle(f'{statistical_test}: communication scores, {selection_label}, {condition} donors only', fontsize=7, y=1.0, va='bottom')

    grid.savefig(path_to_clustermap_png, dpi=300, bbox_inches='tight')
    grid.savefig(path_to_clustermap_pdf, dpi=300, bbox_inches='tight')
    plt.close(grid.figure)


if __name__ == '__main__':
    main()
