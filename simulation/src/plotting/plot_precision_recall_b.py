import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import cm, statistical_test_colors, filter_display_names, make_test_label

import argparse


def load_joined_df(paths_to_dfs: list[str]) -> pd.DataFrame:
    """
    This method helps to join a list of dataframes into a single dataframe.

    :param paths_to_dfs: list of paths to dataframes
    :return: pd.DataFrame
    """
    if not all(map(lambda x: os.path.exists(x), paths_to_dfs)):
        raise FileNotFoundError(
            "Please make sure that all the files exist!"
        )

    return pd.concat(
        [pd.read_pickle(path) for path in paths_to_dfs],
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--paths_to_summary_dfs',
        nargs='+',
        help='Path to summary file',
        required=True
    )
    parser.add_argument(
        '--mean_type_to_plot',
        type=str,
        help='Mean type to plot',
        required=True
    )
    parser.add_argument(
        '--case_to_plot',
        type=str,
        help='Case to plot',
        required=True
    )
    parser.add_argument(
        '--proportions_to_plot',
        nargs='+',
        type=float,
        help='Proportions to plot',
        required=True
    )
    parser.add_argument(
        '--samples_in_dataset',
        type=int,
        help='Number of samples to plot',
        required=True
    )
    parser.add_argument(
        '--statistical_tests',
        nargs='+',
        type=str,
        help='Statistical tests to plot',
        required=True
    )
    parser.add_argument(
        '--width_of_figure_in_cm',
        type=float,
        help='Width of figure in cm',
        default=8.5
    )
    parser.add_argument(
        '--height_of_figure_in_cm',
        type=float,
        help='Height of figure in cm',
        default=8.2 / 2
    )
    parser.add_argument(
        '--dpi',
        type=float,
        help='DPI',
        default=300
    )
    parser.add_argument(
        '--font_size',
        type=float,
        help='Font size',
        default=8
    )
    parser.add_argument(
        '--label_font_size',
        type=float,
        help='Label font size',
        default=8
    )
    parser.add_argument(
        '--path_to_output_figure',
        type=str,
        help='Path to output figure',
        required=True
    )
    parser.add_argument(
        '--show_figure',
        action='store_true',
        help='Show figure',
        default=False
    )
    args = parser.parse_args()

    paths_to_summary_dfs: list[str] = args.paths_to_summary_dfs
    path_to_output_figure: str = args.path_to_output_figure

    mean_type_to_plot: str = args.mean_type_to_plot
    case_to_plot: str = args.case_to_plot
    proportions_to_plot: list[float] = sorted(args.proportions_to_plot)
    samples_in_dataset: int = args.samples_in_dataset
    statistical_tests: list[str] = args.statistical_tests

    width_of_figure_in_cm: float = args.width_of_figure_in_cm
    height_of_figure_in_cm: float = args.height_of_figure_in_cm

    dpi: float = args.dpi

    font_size: float = args.font_size
    label_font_size: float = args.label_font_size

    show_figure: bool = args.show_figure

    n_proportions: int = len(proportions_to_plot)

    # Join the dfs together
    precision_recall_df = load_joined_df(paths_to_summary_dfs)

    # Facet by prefilter: one column per filter present in the data, in the canonical order from
    # utils.filter_display_names. Within each facet the statistical test(s) are drawn across proportions
    # and coloured by statistical_test_colors. The top row shows precision, the bottom row recall.
    present_labels: set[str] = set(precision_recall_df['Statistical Test'])
    present_filters: list[str] = [
        filter_key for filter_key in filter_display_names
        if any(make_test_label(filter_key, test) in present_labels for test in statistical_tests)
    ]
    n_filters: int = len(present_filters)

    # The passed width describes a roughly two-facet figure (the previous precision|recall layout), so we
    # scale the total width by the number of filter columns to keep each facet legible, and double the
    # height for the two metric rows.
    per_facet_width_cm: float = width_of_figure_in_cm / 2
    fig_width_cm: float = per_facet_width_cm * max(1, n_filters)
    fig_height_cm: float = height_of_figure_in_cm * 2

    precision_recall_fig, axes = plt.subplots(
        nrows=2,
        ncols=n_filters,
        figsize=(fig_width_cm * cm, fig_height_cm * cm),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    x_positions: np.ndarray = np.arange(n_proportions)

    def mean_std_per_proportion(label: str, metric: str) -> tuple[list[float], list[float]]:
        """Mean and std of a metric across datasets for one label, ordered by proportions_to_plot."""
        means: list[float] = []
        stds: list[float] = []
        for proportion in proportions_to_plot:
            values: list[float] = precision_recall_df[
                (precision_recall_df['Mean Type'] == mean_type_to_plot) &
                (precision_recall_df['Case'] == case_to_plot) &
                (precision_recall_df['Proportion'] == proportion) &
                (precision_recall_df['Statistical Test'] == label)
            ][metric].values.tolist()
            means.append(np.mean(values).item() if len(values) > 0 else np.nan)
            stds.append(np.std(values).item() if len(values) > 0 else np.nan)
        return means, stds

    for column, filter_key in enumerate(present_filters):
        precision_ax = axes[0][column]
        recall_ax = axes[1][column]

        for metric, metric_ax in (('Precision', precision_ax), ('Recall', recall_ax)):
            for statistical_test in statistical_tests:
                label: str = make_test_label(filter_key, statistical_test)
                means, stds = mean_std_per_proportion(label, metric)

                metric_ax.errorbar(
                    x=x_positions,
                    y=means,
                    yerr=stds,
                    label=statistical_test,
                    color=statistical_test_colors[statistical_test],
                    fmt='-o',
                    capsize=1.5,
                    capthick=1,
                    linewidth=1,
                    markersize=3,
                )

            metric_ax.set_yticks(np.arange(0, 1.1, 0.2))
            metric_ax.set_ylim(0, 1.05)
            metric_ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
            metric_ax.yaxis.grid(False)
            metric_ax.tick_params(axis='x', labelsize=label_font_size)
            metric_ax.tick_params(axis='y', labelsize=label_font_size)

            for x_position in x_positions[:-1]:
                metric_ax.axvline(x_position + 0.5, color='black', linewidth=0.3, linestyle='--', alpha=0.3)

        recall_ax.set_xticks(x_positions)
        recall_ax.set_xticklabels(
            [int(samples_in_dataset * proportion) for proportion in proportions_to_plot]
        )

        precision_ax.set_title(filter_display_names[filter_key], fontsize=font_size)

    # Row labels only on the leftmost column.
    axes[0][0].set_ylabel('Precision', fontsize=font_size)
    axes[1][0].set_ylabel('Recall', fontsize=font_size)

    handles = [
        plt.Line2D([0], [0], color=statistical_test_colors[statistical_test], lw=1, label=statistical_test)
        for statistical_test in statistical_tests
    ]
    precision_recall_fig.legend(
        handles=handles,
        fontsize=font_size,
        loc='upper center',
        ncol=len(statistical_tests),
    )

    precision_recall_fig.tight_layout(rect=(0, 0, 1, 0.93))

    precision_recall_fig.savefig(
        path_to_output_figure,
        dpi=dpi
    )

    if show_figure:
        plt.show()
    else:
        plt.close(precision_recall_fig)


if __name__ == '__main__':
    main()