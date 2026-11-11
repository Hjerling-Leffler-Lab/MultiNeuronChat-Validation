"""NCI vs AD histograms of the communication score for every significant tetrad, with the donors without communication
(score exactly 0) as a separate bar.

Run by the Snakefile rule plot_score_histograms (part of plot_all). Reads the per-donor scores, so it runs on Bianca.

Usage (inside the validation container, from code/validation/rosmap):
    python src/l_plot_mnc_coarse_score_histograms.py \
        --path_to_mnc_object results/coarse/mnc_grouping_by.pkl \
        --path_to_output_dir results/figs/coarse/CVM/padj_0.05

Each panel has two parts sharing the y-axis (fraction of the donors of a condition, so per condition all bars sum to 1):
    left   one bar per condition: donors with score 0, labelled with the percentage (scale of the scores > 0 in the x label)
    right  histogram of the scores > 0, NCI and AD side by side in each bin (bins shared by both conditions)
Outputs, under --path_to_output_dir:
    score_zero_fractions.csv                                     n, n with score 0 and fraction per tetrad and condition
    {png,pdf}/score_histograms.{png,pdf}                         all significant tetrads in one grid
    {png,pdf}/single_panels/score_histograms/<tetrad>.{png,pdf}  one figure per tetrad
"""
import os

import argparse

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

from multineuronchat import MultiNeuronChatObject

from plotting.utils import cm, cell_type_to_short, condition_colors, get_significant_tetrads, format_p_value, tetrad_file_stem

matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in Illustrator / Inkscape

parser = argparse.ArgumentParser(description='NCI vs AD communication-score histograms with a separate zero bar for the significant ROSMAP tetrads.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_output_dir',
    type=str,
    required=True,
    help='Plot directory of one test/threshold, e.g. results/figs/coarse/CVM/padj_0.05.'
)
parser.add_argument(
    '--statistical_test',
    type=str,
    default='CVM',
)
parser.add_argument(
    '--significance_threshold',
    type=float,
    default=0.05,
)
parser.add_argument(
    '--n_bins',
    type=int,
    default=15,
)
parser.add_argument(
    '--n_columns',
    type=int,
    default=4,
)
parser.add_argument(
    '--width_in_cm_per_panel',
    type=float,
    default=5.5,
)
parser.add_argument(
    '--height_in_cm_per_panel',
    type=float,
    default=4.5,
)
args = parser.parse_args()

path_to_output_dir: str = args.path_to_output_dir
path_to_zero_fractions_csv: str = os.path.join(path_to_output_dir, 'score_zero_fractions.csv')
single_panel_dir_png: str = os.path.join(path_to_output_dir, 'png', 'single_panels', 'score_histograms')
single_panel_dir_pdf: str = os.path.join(path_to_output_dir, 'pdf', 'single_panels', 'score_histograms')

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold
n_bins: int = args.n_bins
width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel


def tetrad_title(source: str, receiver: str, interaction: str) -> str:
    short = lambda cell_type: cell_type_to_short.get(cell_type, cell_type)
    return f'{short(source)} → {short(receiver)}  {interaction.replace("_", "–")}'


def get_scores(mnc_object, source: str, receiver: str, interaction: str) -> dict[str, np.ndarray]:
    """Scores of the donors of each condition (donors without a score dropped)."""
    scores = {}
    for condition in mnc_object.condition_names:
        values = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_numpy()
        scores[condition] = values[~np.isnan(values)]
    return scores


def add_condition_legend(fig, scores: dict[str, np.ndarray]):
    fig.legend(
        handles=[Patch(facecolor=condition_colors.get(condition, 'grey'), label=f'{condition} (n={values.shape[0]})')
                 for condition, values in scores.items()],
        loc='outside lower center', ncol=len(scores), fontsize=6, frameon=False,
    )


def plot_score_histogram(subfig, scores: dict[str, np.ndarray], title: str, p_adj: float):
    """Zero bar (left) and histogram of the scores > 0 (right) in one sub-figure, y = fraction of donors."""
    ax_zero, ax_hist = subfig.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [1.5, 4], 'wspace': 0.08})
    conditions = list(scores)
    bar_width = 0.8 / len(conditions)

    # Small scores (e.g. CCK-CCKBR, ~1e-3) are shown as multiples of a power of ten named in the axis label
    positive = np.concatenate([values[values > 0] for values in scores.values()])
    exponent = int(np.floor(np.log10(positive.max()))) if positive.size > 0 else 0
    scale = 10.0 ** -exponent if exponent < -1 else 1.0

    for i, condition in enumerate(conditions):
        values = scores[condition]
        frac_zero = np.mean(values == 0)
        x = (i - (len(conditions) - 1) / 2) * 1.3 * bar_width
        ax_zero.bar(x, frac_zero, width=0.85 * bar_width, color=condition_colors.get(condition, 'grey'), linewidth=0)
        ax_zero.text(x, frac_zero, f'{100 * frac_zero:.0f}%', ha='center', va='bottom', fontsize=5)

    edges = np.histogram_bin_edges(positive * scale, bins=n_bins) if positive.size > 0 else np.linspace(0, 1, n_bins + 1)
    bin_width = edges[1] - edges[0]
    for i, condition in enumerate(conditions):
        values = scores[condition]
        counts, _ = np.histogram(values[values > 0] * scale, bins=edges)
        offset = (i - (len(conditions) - 1) / 2) * bin_width * bar_width
        ax_hist.bar(edges[:-1] + bin_width / 2 + offset, counts / values.shape[0], width=bin_width * bar_width,
                    color=condition_colors.get(condition, 'grey'), linewidth=0)

    ax_zero.set_xlim(-0.85, 0.85)
    ax_zero.set_xticks([0])
    ax_zero.set_xticklabels(['0'])
    ax_zero.set_ylabel('Donors', fontsize=7)
    ax_zero.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax_zero.spines[['top', 'right']].set_visible(False)

    scale_label = f' (×10$^{{{exponent}}}$)' if scale != 1.0 else ''
    subfig.supxlabel(f'Communication score{scale_label}', fontsize=7)
    ax_hist.spines[['top', 'right', 'left']].set_visible(False)
    ax_hist.tick_params(axis='y', left=False)
    ax_hist.text(0.97, 0.97, format_p_value(p_adj), transform=ax_hist.transAxes, ha='right', va='top', fontsize=6)

    for ax in (ax_zero, ax_hist):
        ax.tick_params(axis='both', labelsize=6)
    ax_zero.set_ylim(0, 1.12 * max(ax_zero.get_ylim()[1], 0.05))
    subfig.suptitle(title, fontsize=7)


def main():
    for directory in [os.path.join(path_to_output_dir, 'png'), os.path.join(path_to_output_dir, 'pdf'), single_panel_dir_png, single_panel_dir_pdf]:
        os.makedirs(directory, exist_ok=True)

    mnc_object = MultiNeuronChatObject.load(args.path_to_mnc_object)
    significant = get_significant_tetrads(mnc_object, statistical_test, significance_threshold)
    print(f'{len(significant)} tetrads with {statistical_test} BH-adjusted p < {significance_threshold}')
    if len(significant) == 0:
        return

    titles = [tetrad_title(row.source, row.receiver, row.interaction) for row in significant.itertuples()]
    scores_per_tetrad = {
        title: get_scores(mnc_object, row.source, row.receiver, row.interaction)
        for title, row in zip(titles, significant.itertuples())
    }

    # Aggregated counts only, no donor IDs
    zero_fractions = pd.DataFrame([
        {'tetrad': title, 'condition': condition, 'n_donors': values.shape[0], 'n_zero': int(np.sum(values == 0)),
         'frac_zero': np.mean(values == 0)}
        for title, scores in scores_per_tetrad.items() for condition, values in scores.items()
    ])
    zero_fractions.to_csv(path_to_zero_fractions_csv, index=False)
    print(zero_fractions.to_string(index=False))

    # All tetrads in one grid, one legend underneath
    n_columns = min(args.n_columns, len(titles))
    n_rows = int(np.ceil(len(titles) / n_columns))
    fig = plt.figure(
        figsize=(n_columns * width_in_cm_per_panel * cm, (n_rows * height_in_cm_per_panel + 0.8) * cm),
        layout='constrained',
    )
    subfigs = np.atleast_2d(fig.subfigures(n_rows, n_columns, wspace=0.02, hspace=0.04))
    for subfig, title, row in zip(subfigs.flat, titles, significant.itertuples()):
        plot_score_histogram(subfig, scores_per_tetrad[title], title, row.p_adj)
    add_condition_legend(fig, next(iter(scores_per_tetrad.values())))
    for extension in ('png', 'pdf'):
        fig.savefig(os.path.join(path_to_output_dir, extension, f'score_histograms.{extension}'), dpi=300)
    plt.close(fig)

    # One figure per tetrad
    for title, row in zip(titles, significant.itertuples()):
        fig = plt.figure(figsize=(width_in_cm_per_panel * cm, (height_in_cm_per_panel + 0.6) * cm), layout='constrained')
        plot_score_histogram(fig.subfigures(1, 1), scores_per_tetrad[title], title, row.p_adj)
        add_condition_legend(fig, scores_per_tetrad[title])
        file_stem = tetrad_file_stem(row.source, row.receiver, row.interaction)
        fig.savefig(os.path.join(single_panel_dir_png, f'{file_stem}.png'), dpi=300)
        fig.savefig(os.path.join(single_panel_dir_pdf, f'{file_stem}.pdf'), dpi=300)
        plt.close(fig)

    print(f'Results written to {path_to_output_dir}')


if __name__ == '__main__':
    main()
