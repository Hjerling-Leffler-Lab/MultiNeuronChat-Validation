import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from prefilter_utils import (
    filter_display_names,
    filter_colors,
    filter_line_styles,
    filter_markers,
    filter_hollow_markers,
    parse_test_label,
)
from utils import (
    cm,
    statistical_test_short,
    draw_beeswarm,
    median_and_iqr_errors,
    with_points_path,
)
from plot_n_sweep_precision_recall_b import finite_values_per_n, summarise, style_axis, mark_no_calls

import argparse


statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']

n_sweep_filter_keys: list[str] = [
    'None',
    'Variance-top10',
    'Variance-min0',
    'Abundance-top10',
    'Abundance-min0',
]


def load_n_sweep_df(paths_to_summary_dfs: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_pickle(path) for path in paths_to_summary_dfs], ignore_index=True)


def load_full_cohort_df(paths_to_summary_dfs: list[str], n_per_group: int) -> pd.DataFrame:
    """
    Load the prefilter precision/recall summaries of the full-cohort benchmark (summarise_precision_recall_by_prefilter.py)
    at proportion 1.0 and bring them into the layout of the donor-count sweep, so they can be drawn as its largest
    cohort. The sweep sub-samples donors from the same simulated patient pool, so the full cohort is its natural end
    point.
    """
    df: pd.DataFrame = pd.concat([pd.read_pickle(path) for path in paths_to_summary_dfs], ignore_index=True)
    df = df[df['Proportion'] == 1.0].copy()

    parsed: list[tuple[str, str]] = [parse_test_label(label) for label in df['Statistical Test']]
    df['Filter'] = [filter_key for filter_key, _ in parsed]
    df['Test'] = [statistical_test for _, statistical_test in parsed]
    df['N Donors Per Group'] = n_per_group

    return df[df['Filter'].isin(n_sweep_filter_keys)]


def plot_filters_vs_n(
        df: pd.DataFrame,
        statistical_test: str,
        case: str,
        n_per_group: list[int],
        width_in_cm: float,
        height_in_cm: float,
        font_size: float,
        path_to_output_figure: str,
        show_points: bool = False,
) -> None:
    """
    A3: precision (left) and recall (right) at BH-adjusted p < 0.05 against the cohort size, one line per prefilter.
    Markers show the median over datasets, bars the interquartile range.
    With show_points, the value of every dataset is overlaid as a beeswarm and the lines are dodged further apart.
    """
    x_positions: np.ndarray = np.arange(len(n_per_group))
    dodge_step: float = 0.7 / len(n_sweep_filter_keys) if show_points else 0.06
    offsets: np.ndarray = (np.arange(len(n_sweep_filter_keys)) - (len(n_sweep_filter_keys) - 1) / 2) * dodge_step

    fig, (precision_ax, recall_ax) = plt.subplots(
        1, 2, figsize=(width_in_cm * cm, height_in_cm * cm), sharex=True, sharey=True,
    )

    test_df: pd.DataFrame = df[(df['Test'] == statistical_test) & (df['Case'] == case)]

    # (axis, x, values, color) of every swarm, drawn once the layout is final.
    swarms: list[tuple[plt.Axes, float, np.ndarray, str]] = []

    for filter_key, offset in zip(n_sweep_filter_keys, offsets):
        filter_df: pd.DataFrame = test_df[test_df['Filter'] == filter_key]
        color: str = filter_colors[filter_key]

        for metric, ax in (('Precision', precision_ax), ('Recall', recall_ax)):
            values_per_n: list[np.ndarray] = finite_values_per_n(filter_df, metric, n_per_group)
            medians, iqr_errors = median_and_iqr_errors(values_per_n)
            swarms.extend((ax, x, values, color) for x, values in zip(x_positions + offset, values_per_n))
            ax.errorbar(
                x=x_positions + offset,
                y=medians,
                yerr=iqr_errors,
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
                zorder=2,
            )

    # The unfiltered baseline decides where precision is undefined for every filter it contains.
    _, _, n_finite = summarise(test_df[test_df['Filter'] == 'None'], 'Precision', n_per_group)
    mark_no_calls(precision_ax, x_positions, n_finite, font_size)

    for metric, ax in (('Precision', precision_ax), ('Recall', recall_ax)):
        style_axis(ax, x_positions, n_per_group, font_size)
        ax.set_title(f'{metric} ({statistical_test_short[statistical_test]}, BH 5%)', fontsize=font_size)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Donors per group', fontsize=font_size)

    # Legend above the panels, one column per filter family (as in plot_precision_recall_b.py); the legend fills
    # column by column, so the baseline column is padded with an invisible entry.
    handles, _ = precision_ax.get_legend_handles_labels()
    handle_by_filter: dict[str, object] = dict(zip(n_sweep_filter_keys, handles))
    blank_handle = plt.Line2D([], [], linestyle='none', marker='none')
    legend_order: list[str | None] = ['None', None, 'Variance-top10', 'Variance-min0', 'Abundance-top10', 'Abundance-min0']
    fig.legend(
        handles=[blank_handle if key is None else handle_by_filter[key] for key in legend_order],
        labels=['' if key is None else filter_display_names[key] for key in legend_order],
        fontsize=font_size, loc='upper center', ncol=3, frameon=False, columnspacing=1.0, handlelength=2.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.8))

    if show_points:
        for ax, x, values, color in swarms:
            draw_beeswarm(ax, x, values, color=color, max_half_width=dodge_step * 0.45, marker_size=1.8)

    fig.savefig(path_to_output_figure)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_n_sweep_summary_dfs', nargs='+', required=True)
    parser.add_argument(
        '--paths_to_full_cohort_summary_dfs', nargs='+', required=True,
        help='summarise_precision_recall_by_prefilter.py summaries of the full-cohort benchmark (all tests, the plotted case)',
    )
    parser.add_argument('--mean_type', type=str, required=True, help='e.g. tri_mean')
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--n_donors', nargs='+', type=int, required=True, help='Donors per sex per group of the sweep')
    parser.add_argument('--full_cohort_n_per_group', type=int, required=True)
    parser.add_argument('--paths_to_output_figure_filters', nargs='+', required=True, help='One per test, KS CVM MWU')
    parser.add_argument('--width_of_filters_figure_in_cm', type=float, default=14)
    parser.add_argument('--height_of_figure_in_cm', type=float, default=6.5)
    parser.add_argument('--font_size', type=float, default=8)

    args = parser.parse_args()

    n_sweep_df: pd.DataFrame = load_n_sweep_df(args.paths_to_n_sweep_summary_dfs)
    full_cohort_df: pd.DataFrame = load_full_cohort_df(
        args.paths_to_full_cohort_summary_dfs, n_per_group=args.full_cohort_n_per_group,
    )
    df: pd.DataFrame = pd.concat([n_sweep_df, full_cohort_df], ignore_index=True)
    df = df[df['Mean Type'] == args.mean_type]

    n_per_group: list[int] = [2 * n for n in sorted(args.n_donors)] + [args.full_cohort_n_per_group]

    for statistical_test, path_to_output_figure in zip(statistical_tests, args.paths_to_output_figure_filters):
        os.makedirs(os.path.dirname(path_to_output_figure), exist_ok=True)
        for path, show_points in ((path_to_output_figure, False), (with_points_path(path_to_output_figure), True)):
            plot_filters_vs_n(
                df=df,
                statistical_test=statistical_test,
                case=args.case,
                n_per_group=n_per_group,
                width_in_cm=args.width_of_filters_figure_in_cm,
                height_in_cm=args.height_of_figure_in_cm,
                font_size=args.font_size,
                path_to_output_figure=path,
                show_points=show_points,
            )


if __name__ == '__main__':
    main()
