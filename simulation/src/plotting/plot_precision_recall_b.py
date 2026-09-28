import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import (
    cm,
    filter_display_names,
    filter_colors,
    filter_line_styles,
    filter_markers,
    filter_hollow_markers,
    make_test_label,
)

import argparse


# The prefilter families compared one at a time against the unfiltered baseline. Each family is drawn together
# with all of its retentions (e.g. Variance top 10% and > 0).
filter_families: dict[str, list[str]] = {
    'Wasserstein': ['Wasserstein'],
    'Variance': ['Variance-top10', 'Variance-min0'],
    'Abundance': ['Abundance-top10', 'Abundance-min0'],
}


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


def plot_filters(
        precision_recall_df: pd.DataFrame,
        filter_keys: list[str],
        statistical_test: str,
        mean_type_to_plot: str,
        case_to_plot: str,
        proportions_to_plot: list[float],
        samples_in_dataset: int,
        width_of_figure_in_cm: float,
        height_of_figure_in_cm: float,
        font_size: float,
        label_font_size: float,
        dpi: float,
        path_to_output_figure: str,
        show_figure: bool,
) -> None:
    """
    Draw precision (left) and recall (right) against the number of affected donors, with one line per prefilter
    in the same panel. Lines are dodged horizontally, so that filters with identical results (e.g. the '> 0'
    filters and the unfiltered baseline) stay visible.

    :param precision_recall_df: joined summary dataframes of plot_precision_recall_a.py
    :param filter_keys: keys of filter_display_names to draw, in drawing order
    :param statistical_test: e.g. 'KS'
    :param mean_type_to_plot: e.g. 'tri_mean'
    :param case_to_plot: e.g. 'CASE_1'
    :param proportions_to_plot: sorted proportions of affected donors
    :param samples_in_dataset: number of case donors, used to convert proportions to donor numbers
    :param path_to_output_figure: path of the figure to write
    """
    x_positions: np.ndarray = np.arange(len(proportions_to_plot))
    dodge_step: float = 0.06
    offsets: np.ndarray = (np.arange(len(filter_keys)) - (len(filter_keys) - 1) / 2) * dodge_step

    fig, (precision_ax, recall_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(width_of_figure_in_cm * cm, height_of_figure_in_cm * cm),
        sharex=True,
        sharey=True,
    )

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
            # Precision is NaN for datasets without any significant call; average over the remaining datasets.
            finite_values: np.ndarray = np.asarray(values, dtype=float)
            finite_values = finite_values[np.isfinite(finite_values)]
            means.append(np.mean(finite_values).item() if len(finite_values) > 0 else np.nan)
            stds.append(np.std(finite_values).item() if len(finite_values) > 0 else np.nan)
        return means, stds

    for filter_key, offset in zip(filter_keys, offsets):
        label: str = make_test_label(filter_key, statistical_test)
        color: str = filter_colors[filter_key]

        for metric, metric_ax in (('Precision', precision_ax), ('Recall', recall_ax)):
            means, stds = mean_std_per_proportion(label, metric)

            metric_ax.errorbar(
                x=x_positions + offset,
                y=means,
                yerr=stds,
                label=filter_display_names[filter_key],
                color=color,
                linestyle=filter_line_styles[filter_key],
                marker=filter_markers[filter_key],
                markerfacecolor='white' if filter_key in filter_hollow_markers else color,
                markeredgecolor=color,
                markeredgewidth=0.8,
                capsize=1.5,
                capthick=0.8,
                elinewidth=0.8,
                linewidth=1,
                markersize=3.5,
            )

    for metric, metric_ax in (('Precision', precision_ax), ('Recall', recall_ax)):
        metric_ax.set_title(f'{metric} ({statistical_test})', fontsize=font_size)
        metric_ax.set_yticks(np.arange(0, 1.1, 0.2))
        metric_ax.set_ylim(0, 1.05)
        metric_ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
        metric_ax.set_xticks(x_positions)
        metric_ax.set_xticklabels(
            [int(samples_in_dataset * proportion) for proportion in proportions_to_plot]
        )
        metric_ax.set_xlabel(f'Affected donors (of {samples_in_dataset})', fontsize=font_size)
        metric_ax.tick_params(axis='both', labelsize=label_font_size)
        metric_ax.spines[['top', 'right']].set_visible(False)

        for x_position in x_positions[:-1]:
            metric_ax.axvline(x_position + 0.5, color='black', linewidth=0.3, linestyle='--', alpha=0.3)

    # Legend above the panels. With few lines they share one row; otherwise each column holds one group
    # (baseline + Wasserstein, the variance filters, the abundance filters), padded with invisible entries.
    handles, labels = precision_ax.get_legend_handles_labels()
    handle_by_filter: dict[str, object] = dict(zip(filter_keys, handles))
    if len(filter_keys) <= 3:
        legend_columns: list[list[str]] = [[filter_key] for filter_key in filter_keys]
    else:
        legend_groups: list[list[str]] = [
            ['None', 'Wasserstein'],
            filter_families['Variance'],
            filter_families['Abundance'],
        ]
        legend_columns = [
            [filter_key for filter_key in group if filter_key in filter_keys] for group in legend_groups
        ]
        legend_columns = [column for column in legend_columns if len(column) > 0]
    n_legend_rows: int = max(len(column) for column in legend_columns)
    blank_handle = plt.Line2D([], [], linestyle='none', marker='none')
    legend_handles: list[object] = []
    legend_labels: list[str] = []
    for column in legend_columns:
        for row in range(n_legend_rows):
            if row < len(column):
                legend_handles.append(handle_by_filter[column[row]])
                legend_labels.append(filter_display_names[column[row]])
            else:
                legend_handles.append(blank_handle)
                legend_labels.append('')

    legend_height: float = 0.07 * n_legend_rows + 0.02
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        fontsize=font_size,
        loc='upper center',
        ncol=len(legend_columns),
        frameon=False,
        columnspacing=1.0,
        handlelength=2.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 1 - legend_height))

    fig.savefig(
        path_to_output_figure,
        dpi=dpi
    )

    if show_figure:
        plt.show()
    else:
        plt.close(fig)


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
        '--statistical_test',
        type=str,
        help='Statistical test to plot',
        required=True
    )
    parser.add_argument(
        '--width_of_figure_in_cm',
        type=float,
        help='Width of figure in cm',
        default=14
    )
    parser.add_argument(
        '--height_of_figure_in_cm',
        type=float,
        help='Height of figure in cm',
        default=6.5
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
        '--path_to_output_figure_all_filters',
        type=str,
        help='Path to the figure with the unfiltered baseline and all prefilters',
        required=True
    )
    parser.add_argument(
        '--path_to_output_figure_without_wasserstein',
        type=str,
        help='Path to the figure with the unfiltered baseline and all prefilters except Wasserstein',
        required=True
    )
    parser.add_argument(
        '--path_to_output_figure_wasserstein',
        type=str,
        help='Path to the figure with the unfiltered baseline and the Wasserstein prefilter',
        required=True
    )
    parser.add_argument(
        '--path_to_output_figure_variance',
        type=str,
        help='Path to the figure with the unfiltered baseline and the variance prefilters',
        required=True
    )
    parser.add_argument(
        '--path_to_output_figure_abundance',
        type=str,
        help='Path to the figure with the unfiltered baseline and the abundance prefilters',
        required=True
    )
    parser.add_argument(
        '--show_figure',
        action='store_true',
        help='Show figure',
        default=False
    )
    args = parser.parse_args()

    precision_recall_df = load_joined_df(args.paths_to_summary_dfs)
    statistical_test: str = args.statistical_test

    # Only draw the filters present in the summary dataframes, in the canonical order of filter_display_names.
    present_labels: set[str] = set(precision_recall_df['Statistical Test'])
    present_filters: list[str] = [
        filter_key for filter_key in filter_display_names
        if make_test_label(filter_key, statistical_test) in present_labels
    ]

    def present(filter_keys: list[str]) -> list[str]:
        return [filter_key for filter_key in filter_keys if filter_key in present_filters]

    figures: dict[str, list[str]] = {
        args.path_to_output_figure_all_filters: present_filters,
        args.path_to_output_figure_without_wasserstein: [
            filter_key for filter_key in present_filters if filter_key != 'Wasserstein'
        ],
        args.path_to_output_figure_wasserstein: present(['None'] + filter_families['Wasserstein']),
        args.path_to_output_figure_variance: present(['None'] + filter_families['Variance']),
        args.path_to_output_figure_abundance: present(['None'] + filter_families['Abundance']),
    }

    for path_to_output_figure, filter_keys in figures.items():
        os.makedirs(os.path.dirname(path_to_output_figure), exist_ok=True)
        plot_filters(
            precision_recall_df=precision_recall_df,
            filter_keys=filter_keys,
            statistical_test=statistical_test,
            mean_type_to_plot=args.mean_type_to_plot,
            case_to_plot=args.case_to_plot,
            proportions_to_plot=sorted(args.proportions_to_plot),
            samples_in_dataset=args.samples_in_dataset,
            width_of_figure_in_cm=args.width_of_figure_in_cm,
            height_of_figure_in_cm=args.height_of_figure_in_cm,
            font_size=args.font_size,
            label_font_size=args.label_font_size,
            dpi=args.dpi,
            path_to_output_figure=path_to_output_figure,
            show_figure=args.show_figure,
        )


if __name__ == '__main__':
    main()
