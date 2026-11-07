import os

import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import (
    cm,
    statistical_test_colors,
    statistical_test_short,
    convert_cases_l2fc_label,
    draw_beeswarm,
    median_and_iqr_errors,
    with_points_path,
)

import argparse


statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']
test_markers: dict[str, str] = {'KS': 'o', 'CVM': 's', 'MannWhitneyU': '^'}


def load_n_sweep_df(paths_to_summary_dfs: list[str]) -> pd.DataFrame:
    return pd.concat([pd.read_pickle(path) for path in paths_to_summary_dfs], ignore_index=True)


def load_full_cohort_df(paths_to_summary_dfs: list[str], n_per_group: int) -> pd.DataFrame:
    """
    Load the precision/recall summaries of the full-cohort benchmark (plot_precision_recall_a.py) at proportion 1.0
    and bring them into the layout of the donor-count sweep, so they can be drawn as its largest cohort. The sweep
    sub-samples donors from the same simulated patient pool, so the full cohort is its natural end point.
    """
    df: pd.DataFrame = pd.concat([pd.read_pickle(path) for path in paths_to_summary_dfs], ignore_index=True)
    df = df[df['Proportion'] == 1.0].copy()
    df['N Donors Per Group'] = n_per_group

    return df


def finite_values_per_n(df: pd.DataFrame, metric: str, n_per_group: list[int]) -> list[np.ndarray]:
    """The finite per-dataset values of a metric, one array per cohort size."""
    values_per_n: list[np.ndarray] = []
    for n in n_per_group:
        values: np.ndarray = df.loc[df['N Donors Per Group'] == n, metric].to_numpy(dtype=float)
        values_per_n.append(values[np.isfinite(values)])
    return values_per_n


def summarise(df: pd.DataFrame, metric: str, n_per_group: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean and std of a metric per cohort size, plus the number of datasets with a finite value. Precision is NaN for a
    dataset without any significant call, so it is averaged over the remaining datasets only.
    """
    means: list[float] = []
    stds: list[float] = []
    n_finite: list[int] = []
    for values in finite_values_per_n(df, metric, n_per_group):
        means.append(np.mean(values) if len(values) > 0 else np.nan)
        stds.append(np.std(values) if len(values) > 0 else np.nan)
        n_finite.append(len(values))
    return np.array(means), np.array(stds), np.array(n_finite)


def style_axis(ax: plt.Axes, x_positions: np.ndarray, n_per_group: list[int], font_size: float) -> None:
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(n_per_group)
    ax.tick_params(axis='both', labelsize=font_size)
    ax.spines[['top', 'right']].set_visible(False)


def mark_no_calls(ax: plt.Axes, x_positions: np.ndarray, n_finite: np.ndarray, font_size: float) -> None:
    """Label the cohort sizes at which no dataset produced a significant call (precision undefined)."""
    for x, n in zip(x_positions, n_finite):
        if n == 0:
            ax.text(x, 0.03, 'no calls', rotation=90, ha='center', va='bottom', fontsize=font_size - 1, color='#6b6b6b')


def plot_tests_vs_n(
        df: pd.DataFrame,
        pr_aucs: dict[int, dict[str, list[float]]],
        case: str,
        n_per_group: list[int],
        width_in_cm: float,
        height_in_cm: float,
        font_size: float,
        path_to_output_figure: str,
        alpha: float = 0.05,
        show_points: bool = False,
) -> None:
    """
    Per test against the cohort size:
    (a) PR AUC (a ranking metric, no significance threshold),
    (b) recall and (c) precision of the calls at BH-adjusted p < alpha.
    Markers show the median over datasets, bars the interquartile range.
    With show_points, the value of every dataset is overlaid as a beeswarm and the tests are dodged
    further apart.
    """
    x_positions: np.ndarray = np.arange(len(n_per_group))
    dodge_step: float = 0.25 if show_points else 0.12
    offsets: np.ndarray = (np.arange(len(statistical_tests)) - (len(statistical_tests) - 1) / 2) * dodge_step

    fig, (auc_ax, recall_ax, precision_ax) = plt.subplots(
        1, 3, figsize=(width_in_cm * cm, height_in_cm * cm),
    )

    case_df: pd.DataFrame = df[df['Case'] == case]

    # (axis, x, values, color) of every swarm, drawn once the layout is final.
    swarms: list[tuple[plt.Axes, float, np.ndarray, str]] = []

    for statistical_test, offset in zip(statistical_tests, offsets):
        color: str = statistical_test_colors[statistical_test]
        test_df: pd.DataFrame = case_df[case_df['Test'] == statistical_test]
        line_kwargs = dict(
            color=color, marker=test_markers[statistical_test], markersize=3, linewidth=1, capsize=1.5,
            capthick=0.8, elinewidth=0.8, label=statistical_test_short[statistical_test], zorder=2,
        )

        auc_values: list[np.ndarray] = [np.asarray(pr_aucs[n][statistical_test], dtype=float) for n in n_per_group]
        auc_medians, auc_iqr_errors = median_and_iqr_errors(auc_values)
        auc_ax.errorbar(x_positions + offset, auc_medians, yerr=auc_iqr_errors, **line_kwargs)
        swarms.extend((auc_ax, x, values, color) for x, values in zip(x_positions + offset, auc_values))

        for metric, ax in (('Recall', recall_ax), ('Precision', precision_ax)):
            values_per_n = finite_values_per_n(test_df, metric, n_per_group)
            medians, iqr_errors = median_and_iqr_errors(values_per_n)
            ax.errorbar(x_positions + offset, medians, yerr=iqr_errors, **line_kwargs)
            swarms.extend((ax, x, values, color) for x, values in zip(x_positions + offset, values_per_n))

    _, _, n_finite = summarise(case_df[case_df['Test'] == 'KS'], 'Precision', n_per_group)
    mark_no_calls(precision_ax, x_positions, n_finite, font_size)

    auc_ax.set_title('PR AUC', fontsize=font_size)
    recall_ax.set_title(f'Recall (BH {alpha:.0%})', fontsize=font_size)
    precision_ax.set_title(f'Precision (BH {alpha:.0%})', fontsize=font_size)
    for ax in (auc_ax, recall_ax, precision_ax):
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    for ax in (auc_ax, recall_ax, precision_ax):
        style_axis(ax, x_positions, n_per_group, font_size)

    handles, labels = auc_ax.get_legend_handles_labels()
    fig.legend(
        handles=handles, labels=labels, fontsize=font_size, loc='upper center', ncol=3, frameon=False, columnspacing=1.2,
    )
    fig.supxlabel('Donors per group (cases = controls)', fontsize=font_size)
    fig.text(0.01, 0.99, convert_cases_l2fc_label(case), ha='left', va='top', fontsize=font_size)
    fig.tight_layout(rect=(0, 0, 1, 0.88))

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
        help='plot_precision_recall_a.py summaries of the full-cohort benchmark (all tests, the plotted case)',
    )
    parser.add_argument('--path_to_n_sweep_pr_aucs', type=str, required=True)
    parser.add_argument('--path_to_full_cohort_pr_aucs', type=str, required=True)
    parser.add_argument('--mean_type', type=str, required=True, help='e.g. tri_mean')
    parser.add_argument('--trim_mean_fraction', type=str, required=True, help='e.g. 0')
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--n_donors', nargs='+', type=int, required=True, help='Donors per sex per group of the sweep')
    parser.add_argument('--full_cohort_n_per_group', type=int, required=True)
    parser.add_argument('--path_to_output_figure_tests', type=str, required=True)
    parser.add_argument('--width_of_tests_figure_in_cm', type=float, default=13.5)
    parser.add_argument('--height_of_figure_in_cm', type=float, default=6.5)
    parser.add_argument('--font_size', type=float, default=8)

    args = parser.parse_args()

    mean_type_label: str = f'{args.mean_type}_{args.trim_mean_fraction}'

    n_sweep_df: pd.DataFrame = load_n_sweep_df(args.paths_to_n_sweep_summary_dfs)
    full_cohort_df: pd.DataFrame = load_full_cohort_df(
        args.paths_to_full_cohort_summary_dfs, n_per_group=args.full_cohort_n_per_group,
    )
    df: pd.DataFrame = pd.concat([n_sweep_df, full_cohort_df], ignore_index=True)
    df = df[df['Mean Type'] == args.mean_type]

    n_per_group: list[int] = [2 * n for n in sorted(args.n_donors)] + [args.full_cohort_n_per_group]

    pr_aucs: dict[int, dict[str, list[float]]] = {}
    for n in sorted(args.n_donors):
        with open(os.path.join(args.path_to_n_sweep_pr_aucs, f'{mean_type_label}_{args.case}_{n}.pkl'), 'rb') as f:
            pr_aucs[2 * n] = pickle.load(f)
    with open(os.path.join(args.path_to_full_cohort_pr_aucs, f'{mean_type_label}_{args.case}_1.0.pkl'), 'rb') as f:
        pr_aucs[args.full_cohort_n_per_group] = pickle.load(f)

    os.makedirs(os.path.dirname(args.path_to_output_figure_tests), exist_ok=True)
    for path, show_points in (
            (args.path_to_output_figure_tests, False), (with_points_path(args.path_to_output_figure_tests), True),
    ):
        plot_tests_vs_n(
            df=df,
            pr_aucs=pr_aucs,
            case=args.case,
            n_per_group=n_per_group,
            width_in_cm=args.width_of_tests_figure_in_cm,
            height_in_cm=args.height_of_figure_in_cm,
            font_size=args.font_size,
            path_to_output_figure=path,
            show_points=show_points,
        )


if __name__ == '__main__':
    main()
