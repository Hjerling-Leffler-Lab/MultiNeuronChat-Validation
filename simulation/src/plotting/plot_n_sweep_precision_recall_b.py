import os

import pickle

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator

from utils import (
    cm,
    statistical_test_colors,
    statistical_test_short,
    filter_display_names,
    filter_colors,
    filter_line_styles,
    filter_markers,
    filter_hollow_markers,
    parse_test_label,
    convert_cases_l2fc_label,
)

import argparse


statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']
test_markers: dict[str, str] = {'KS': 'o', 'CVM': 's', 'MannWhitneyU': '^'}

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
    Load the precision/recall summaries of the full-cohort benchmark (plot_precision_recall_a.py) at proportion 1.0
    and bring them into the layout of the donor-count sweep, so they can be drawn as its largest cohort. The sweep
    sub-samples donors from the same simulated patient pool, so the full cohort is its natural end point.
    """
    df: pd.DataFrame = pd.concat([pd.read_pickle(path) for path in paths_to_summary_dfs], ignore_index=True)
    df = df[df['Proportion'] == 1.0].copy()

    parsed: list[tuple[str, str]] = [parse_test_label(label) for label in df['Statistical Test']]
    df['Filter'] = [filter_key for filter_key, _ in parsed]
    df['Test'] = [statistical_test for _, statistical_test in parsed]
    df['N Donors Per Group'] = n_per_group

    return df[df['Filter'].isin(n_sweep_filter_keys)]


def summarise(df: pd.DataFrame, metric: str, n_per_group: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mean and std of a metric per cohort size, plus the number of datasets with a finite value. Precision is NaN for a
    dataset without any significant call, so it is averaged over the remaining datasets only.
    """
    means: list[float] = []
    stds: list[float] = []
    n_finite: list[int] = []
    for n in n_per_group:
        values: np.ndarray = df.loc[df['N Donors Per Group'] == n, metric].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
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


def plot_filters_vs_n(
        df: pd.DataFrame,
        statistical_test: str,
        case: str,
        n_per_group: list[int],
        width_in_cm: float,
        height_in_cm: float,
        font_size: float,
        path_to_output_figure: str,
) -> None:
    """
    A3: precision (left) and recall (right) at BY-adjusted p < 0.05 against the cohort size, one line per prefilter.
    """
    x_positions: np.ndarray = np.arange(len(n_per_group))
    dodge_step: float = 0.06
    offsets: np.ndarray = (np.arange(len(n_sweep_filter_keys)) - (len(n_sweep_filter_keys) - 1) / 2) * dodge_step

    fig, (precision_ax, recall_ax) = plt.subplots(
        1, 2, figsize=(width_in_cm * cm, height_in_cm * cm), sharex=True, sharey=True,
    )

    test_df: pd.DataFrame = df[(df['Test'] == statistical_test) & (df['Case'] == case)]

    for filter_key, offset in zip(n_sweep_filter_keys, offsets):
        filter_df: pd.DataFrame = test_df[test_df['Filter'] == filter_key]
        color: str = filter_colors[filter_key]

        for metric, ax in (('Precision', precision_ax), ('Recall', recall_ax)):
            means, stds, _ = summarise(filter_df, metric, n_per_group)
            ax.errorbar(
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

    # The unfiltered baseline decides where precision is undefined for every filter it contains.
    _, _, n_finite = summarise(test_df[test_df['Filter'] == 'None'], 'Precision', n_per_group)
    mark_no_calls(precision_ax, x_positions, n_finite, font_size)

    for metric, ax in (('Precision', precision_ax), ('Recall', recall_ax)):
        style_axis(ax, x_positions, n_per_group, font_size)
        ax.set_title(f'{metric} ({statistical_test_short[statistical_test]}, BY 5%)', fontsize=font_size)
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
    fig.savefig(path_to_output_figure)
    plt.close(fig)


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
) -> None:
    """
    B1: without prefiltering, per test against the cohort size:
    (a) PR AUC (a ranking metric, no significance threshold),
    (b) recall and (c) precision of the calls at BY-adjusted p < alpha,
    (d) the detectability floor: the smallest number of triples that must reach the minimal attainable p-value
        together before BY can call anything (line), against the number of triples that actually reach it (markers).
    """
    x_positions: np.ndarray = np.arange(len(n_per_group))
    dodge_step: float = 0.12
    offsets: np.ndarray = (np.arange(len(statistical_tests)) - (len(statistical_tests) - 1) / 2) * dodge_step
    rng = np.random.default_rng(0)

    fig, (auc_ax, recall_ax, precision_ax, floor_ax) = plt.subplots(
        1, 4, figsize=(width_in_cm * cm, height_in_cm * cm),
    )

    case_df: pd.DataFrame = df[(df['Case'] == case) & (df['Filter'] == 'None')]

    for statistical_test, offset in zip(statistical_tests, offsets):
        color: str = statistical_test_colors[statistical_test]
        test_df: pd.DataFrame = case_df[case_df['Test'] == statistical_test]
        line_kwargs = dict(
            color=color, marker=test_markers[statistical_test], markersize=3, linewidth=1, capsize=1.5,
            capthick=0.8, elinewidth=0.8, label=statistical_test_short[statistical_test],
        )

        auc_values: list[np.ndarray] = [np.asarray(pr_aucs[n][statistical_test], dtype=float) for n in n_per_group]
        auc_ax.errorbar(
            x_positions + offset, [v.mean() for v in auc_values], yerr=[v.std() for v in auc_values], **line_kwargs,
        )
        for x, values in zip(x_positions + offset, auc_values):
            jitter: np.ndarray = (rng.random(len(values)) - 0.5) * dodge_step * 0.6
            auc_ax.scatter(x + jitter, values, color=color, s=2, alpha=0.4, linewidths=0, zorder=1)

        for metric, ax in (('Recall', recall_ax), ('Precision', precision_ax)):
            means, stds, _ = summarise(test_df, metric, n_per_group)
            ax.errorbar(x_positions + offset, means, yerr=stds, **line_kwargs)

        # Observed triples at (or below) the floor, only where the floor binds (more than one triple needed).
        # Where one triple suffices, the floor is far below any realised p-value and the count carries no meaning.
        needed, _, _ = summarise(test_df, 'N Needed At Floor', n_per_group)
        at_floor, _, _ = summarise(test_df, 'N At Floor', n_per_group)
        binds: np.ndarray = needed > 1
        floor_ax.plot(
            (x_positions + offset)[binds], at_floor[binds], linestyle='none', marker=test_markers[statistical_test],
            color=color, markersize=3.5, label=f'{statistical_test_short[statistical_test]}: at floor',
        )

    _, _, n_finite = summarise(case_df[case_df['Test'] == 'KS'], 'Precision', n_per_group)
    mark_no_calls(precision_ax, x_positions, n_finite, font_size)

    # Theoretical requirement (the same for every rank test) from the measured number of tested hypotheses.
    needed, _, _ = summarise(case_df, 'N Needed At Floor', n_per_group)
    floor_ax.plot(x_positions, needed, color='#3d3d3a', linewidth=1, marker='_', markersize=6, label='needed for any call')
    n_ground_truth: float = case_df['N Ground Truth Tested'].median()
    floor_ax.axhline(n_ground_truth, color='#6b6b6b', linewidth=0.8, linestyle=':')
    floor_ax.text(
        x_positions[-1] + 0.4, n_ground_truth * 1.15, f'true tetrads tested ({n_ground_truth:.0f})',
        ha='right', va='bottom', fontsize=font_size - 1, color='#6b6b6b',
    )
    floor_ax.set_yscale('log')
    floor_ax.set_ylim(0.7, 1500)
    floor_ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    floor_ax.yaxis.set_minor_locator(NullLocator())
    floor_ax.set_yticklabels(['1', '10', '100', '1000'])
    floor_ax.set_title('Triples at the p-value floor', fontsize=font_size)

    auc_ax.set_title('PR AUC', fontsize=font_size)
    recall_ax.set_title(f'Recall (BY {alpha:.0%})', fontsize=font_size)
    precision_ax.set_title(f'Precision (BY {alpha:.0%})', fontsize=font_size)
    for ax in (auc_ax, recall_ax, precision_ax):
        ax.set_ylim(0, 1.05)
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    for ax in (auc_ax, recall_ax, precision_ax, floor_ax):
        style_axis(ax, x_positions, n_per_group, font_size)

    handles, labels = auc_ax.get_legend_handles_labels()
    floor_handles, _ = floor_ax.get_legend_handles_labels()
    fig.legend(
        handles=handles + floor_handles[-1:], labels=labels + ['floor: triples needed for any call'],
        fontsize=font_size, loc='upper center', ncol=4, frameon=False, columnspacing=1.2,
    )
    fig.supxlabel('Donors per group (cases = controls)', fontsize=font_size)
    fig.text(0.01, 0.99, convert_cases_l2fc_label(case), ha='left', va='top', fontsize=font_size)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
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
    parser.add_argument('--paths_to_output_figure_filters', nargs='+', required=True, help='One per test, KS CVM MWU')
    parser.add_argument('--width_of_tests_figure_in_cm', type=float, default=18)
    parser.add_argument('--width_of_filters_figure_in_cm', type=float, default=14)
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
    plot_tests_vs_n(
        df=df,
        pr_aucs=pr_aucs,
        case=args.case,
        n_per_group=n_per_group,
        width_in_cm=args.width_of_tests_figure_in_cm,
        height_in_cm=args.height_of_figure_in_cm,
        font_size=args.font_size,
        path_to_output_figure=args.path_to_output_figure_tests,
    )

    for statistical_test, path_to_output_figure in zip(statistical_tests, args.paths_to_output_figure_filters):
        os.makedirs(os.path.dirname(path_to_output_figure), exist_ok=True)
        plot_filters_vs_n(
            df=df,
            statistical_test=statistical_test,
            case=args.case,
            n_per_group=n_per_group,
            width_in_cm=args.width_of_filters_figure_in_cm,
            height_in_cm=args.height_of_figure_in_cm,
            font_size=args.font_size,
            path_to_output_figure=path_to_output_figure,
        )


if __name__ == '__main__':
    main()
