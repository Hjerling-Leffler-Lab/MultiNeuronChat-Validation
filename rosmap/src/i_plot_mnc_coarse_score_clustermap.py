import os

import numpy as np
import pandas as pd
import xarray as xr

from scipy import stats
from scipy.cluster import hierarchy

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns

import argparse

from plotting.utils import cell_type_to_short, condition_colors, get_significant_tetrads, save_no_significant_tetrads_figure

parser = argparse.ArgumentParser(description='Cluster donors and tetrads on their communication scores and plot them as a heatmap.')

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
    '--path_to_donor_clusters_csv',
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
    '--n_donor_clusters',
    type=int,
    default=2,
    help='Number of clusters the donor dendrogram is cut into to compare against the condition labels.'
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
path_to_donor_clusters_csv: str = args.path_to_donor_clusters_csv

statistical_test: str = args.statistical_test
selection: str = args.selection
significance_threshold: float = args.significance_threshold
n_donor_clusters: int = args.n_donor_clusters

width_in_cm: float = args.width_in_cm
height_in_cm_per_tetrad: float = args.height_in_cm_per_tetrad

cm: float = 1/2.54  # cm to inches conversion factor

# z-scores beyond this are clipped in the colour scale only, not in the clustering
z_score_limit: float = 3.0


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


def get_score_matrix(mnc_object: MultiNeuronChatObject, tetrads: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Communication scores as a tetrads x donors matrix, plus the condition of every donor."""
    scores = xr.concat([mnc_object.communication_score_dict[condition] for condition in mnc_object.condition_names], dim='subject')
    conditions = pd.Series(
        np.concatenate([[condition] * mnc_object.communication_score_dict[condition].sizes['subject'] for condition in mnc_object.condition_names]),
        index=scores['subject'].to_numpy(),
    )

    rows = {
        tetrad_label(row.source, row.receiver, row.interaction): scores.loc[:, row.source, row.receiver, row.interaction].to_numpy()
        for row in tetrads.itertuples()
    }
    return pd.DataFrame(rows, index=conditions.index).T, conditions


def main():
    for output_path in [path_to_clustermap_png, path_to_clustermap_pdf, path_to_donor_clusters_csv]:
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)

    tetrads: pd.DataFrame = select_tetrads(mnc_object)
    if tetrads.empty:
        save_no_significant_tetrads_figure([path_to_clustermap_png, path_to_clustermap_pdf], statistical_test, significance_threshold)
        pd.DataFrame(columns=['subject', 'condition', 'cluster']).to_csv(path_to_donor_clusters_csv, index=False)
        return

    scores, conditions = get_score_matrix(mnc_object, tetrads)

    # Scores differ by orders of magnitude between tetrads, so standardise every tetrad across donors.
    # Donors lacking a cell type have no score; they are set to the tetrad mean (z = 0) for clustering and drawn grey.
    z_scores = scores.sub(scores.mean(axis=1), axis=0).div(scores.std(axis=1), axis=0)
    missing = z_scores.isna()
    z_scores_filled = z_scores.fillna(0.0)

    donor_linkage = hierarchy.linkage(z_scores_filled.T.to_numpy(), method='ward')
    tetrad_linkage = hierarchy.linkage(z_scores_filled.to_numpy(), method='ward') if len(z_scores) > 1 else None

    # How well do the unsupervised donor clusters recover NCI vs AD?
    donor_clusters = pd.Series(hierarchy.fcluster(donor_linkage, t=n_donor_clusters, criterion='maxclust'), index=z_scores.columns)
    contingency = pd.crosstab(donor_clusters, conditions)
    _, p_chi2, _, _ = stats.chi2_contingency(contingency.to_numpy())
    pd.DataFrame({'subject': donor_clusters.index, 'condition': conditions[donor_clusters.index].to_numpy(), 'cluster': donor_clusters.to_numpy()}) \
        .to_csv(path_to_donor_clusters_csv, index=False)

    cluster_summary = ', '.join(
        f'C{cluster}: ' + '/'.join(f'{contingency.loc[cluster, condition]} {condition}' for condition in mnc_object.condition_names)
        for cluster in contingency.index
    )

    height_in_cm = 5.0 + height_in_cm_per_tetrad * len(z_scores)
    grid = sns.clustermap(
        z_scores_filled,
        mask=missing,
        row_linkage=tetrad_linkage,
        col_linkage=donor_linkage,
        row_cluster=tetrad_linkage is not None,
        col_colors=conditions.map(condition_colors).rename('Condition'),
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
    grid.ax_heatmap.set_xlabel(f'Donors (n = {z_scores.shape[1]})', fontsize=7)
    grid.ax_heatmap.set_ylabel('')
    grid.ax_heatmap.tick_params(axis='y', labelsize=6, length=0)
    grid.ax_col_colors.tick_params(axis='y', labelsize=6, length=0)

    grid.ax_cbar.set_yticks([-z_score_limit, 0, z_score_limit])
    grid.ax_cbar.set_yticklabels([f'≤ {-z_score_limit:g}', '0', f'≥ {z_score_limit:g}'])
    grid.ax_cbar.tick_params(labelsize=5)
    grid.ax_cbar.set_title('z-score', fontsize=6)

    grid.figure.legend(
        handles=[Patch(color=condition_colors[condition], label=condition) for condition in mnc_object.condition_names],
        loc='upper left', bbox_to_anchor=(0.0, 0.8), fontsize=6, frameon=False,
    )

    selection_label = f'adj. p < {significance_threshold}' if selection == 'significant' else f'top {len(z_scores)} by p-value'
    grid.figure.suptitle(
        f'{statistical_test}: communication scores, {selection_label}\n'
        f'{n_donor_clusters} donor clusters vs condition: {cluster_summary} (χ² p = {p_chi2:.2g})',
        fontsize=7, y=1.02,
    )

    grid.savefig(path_to_clustermap_png, dpi=300, bbox_inches='tight')
    grid.savefig(path_to_clustermap_pdf, dpi=300, bbox_inches='tight')
    plt.close(grid.figure)


if __name__ == '__main__':
    main()
