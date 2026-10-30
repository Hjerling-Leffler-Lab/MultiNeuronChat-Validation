import os

import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.cluster import hierarchy

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

from plotting.utils import cell_type_to_short, save_no_significant_tetrads_figure

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
    '--p_value_correction',
    type=str,
    default='bh',
    choices=['bh', 'by', 'stored'],
    help="Re-adjust the raw p-values over all tested tetrads ('bh'/'by'), or use the object's stored adjusted p ('stored')",
)
parser.add_argument(
    '--condition',
    type=str,
    default='AD',
    help='Condition whose donors are clustered; z-scores are computed within this condition.'
)
parser.add_argument(
    '--width_in_cm',
    type=float,
    default=10.0,
)
parser.add_argument(
    '--height_in_cm',
    type=float,
    default=10.0,
    help='Figure height; with the 10 cm default width the clustermap matches the clinical covariate summary heatmap.'
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_clustermap_png: str = args.path_to_clustermap_png
path_to_clustermap_pdf: str = args.path_to_clustermap_pdf
path_to_donor_order_csv: str = args.path_to_donor_order_csv

statistical_test: str = args.statistical_test
selection: str = args.selection
significance_threshold: float = args.significance_threshold
p_value_correction: str = args.p_value_correction
condition: str = args.condition

width_in_cm: float = args.width_in_cm
height_in_cm: float = args.height_in_cm

cm: float = 1/2.54  # cm to inches conversion factor

# z-scores beyond this are clipped in the colour scale only, not in the clustering
z_score_limit: float = 3.0

def get_adjusted_p_values(mnc_object: MultiNeuronChatObject) -> xr.DataArray:
    """Adjusted p-values: re-adjusted from the raw p (same as j_plot_mnc_coarse_clinical_covariates.py) or the stored ones."""
    if p_value_correction == 'stored':
        return mnc_object.p_values_adj[statistical_test]
    p_values = mnc_object.p_values[statistical_test]
    tested = np.isfinite(p_values.values)
    adjusted = np.full(p_values.shape, np.nan)
    adjusted[tested] = stats.false_discovery_control(p_values.values[tested], method=p_value_correction)
    return p_values.copy(data=adjusted)


def select_tetrads(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    """Return (source, receiver, interaction, p, p_adj) of the tetrads to show, sorted by raw p-value."""
    p_values = mnc_object.p_values[statistical_test].to_dataframe('p').dropna()
    p_values_adj = get_adjusted_p_values(mnc_object).to_dataframe('p_adj').dropna()
    tetrads = p_values.join(p_values_adj).sort_values(['p', 'p_adj']).reset_index()
    print(f'{(tetrads["p_adj"] < significance_threshold).sum()} of {len(tetrads)} tetrads with {statistical_test} '
          f'{p_value_correction.upper()}-adjusted p < {significance_threshold}')

    if selection == 'significant':
        return tetrads[tetrads['p_adj'] < significance_threshold]
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

    donor_table.to_csv(path_to_donor_order_csv, index=False)

    grid = sns.clustermap(
        z_scores_filled,
        mask=missing,
        row_linkage=tetrad_linkage,
        col_linkage=donor_linkage,
        row_cluster=tetrad_linkage is not None,
        cmap='RdBu_r',
        center=0,
        vmin=-z_score_limit,
        vmax=z_score_limit,
        xticklabels=False,
        yticklabels=True,
        figsize=(width_in_cm * cm, height_in_cm * cm),
        dendrogram_ratio=(0.08, 2.0 / height_in_cm),
        cbar_pos=(0.01, 0.82, 0.015, 0.12),
        tree_kws={'linewidths': 0.4},
    )
    grid.ax_heatmap.set_facecolor('lightgrey')
    grid.ax_heatmap.set_xlabel(f'{condition} donors (n = {z_scores.shape[1]})', fontsize=7)
    grid.ax_heatmap.set_ylabel('')
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0)

    grid.ax_cbar.set_yticks([-z_score_limit, 0, z_score_limit])
    grid.ax_cbar.set_yticklabels([f'≤ {-z_score_limit:g}', '0', f'≥ {z_score_limit:g}'])
    grid.ax_cbar.tick_params(labelsize=5)
    grid.ax_cbar.set_title('z-score', fontsize=6)

    # The title goes inside the figure, so the saved file is exactly width_in_cm x height_in_cm
    title_height: float = 0.6 * cm / grid.figure.get_figheight()
    grid.figure.tight_layout(h_pad=0.02, w_pad=0.02, rect=(0.0, 0.0, 1.0, 1.0 - title_height))
    grid.ax_cbar.set_position((0.035, 0.82 - title_height, 0.015, 0.12))  # far enough right for the "z-score" title

    selection_label = f'{p_value_correction.upper()}-adj. p < {significance_threshold}' if selection == 'significant' else f'top {len(z_scores)} by p-value'
    grid.figure.suptitle(f'{statistical_test}: communication scores, {selection_label}, {condition} donors only', fontsize=7, y=0.995, va='top')

    grid.figure.savefig(path_to_clustermap_png, dpi=300)  # not grid.savefig, which defaults to bbox_inches="tight"
    grid.figure.savefig(path_to_clustermap_pdf, dpi=300)
    plt.close(grid.figure)


if __name__ == '__main__':
    main()
