import os
import pickle

import argparse

import numpy as np

import scipy

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from utils import (
    cm, statistical_test_colors, statistical_test_short, convert_mean_type_label, convert_p_value_string_to_stars,
    draw_beeswarm, with_points_path,
)


case_l2fc_tick_labels: dict[str, str] = {
    'CASE_0.3': '0.3',
    'CASE_0.5': '0.5',
    'CASE_1': '1.0',
    'CASE_1.5': '1.5',
}


def load_pr_aucs(
        path_to_pr_aucs: str,
        mean_type: str,
        cases: list[str],
        proportion: str,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Load the per-replicate PR AUCs of one mean type and one proportion for every case.

    :return: {case: {statistical_test: array of PR AUCs, one per dataset replicate}}
    """
    pr_auc_data_dict: dict[str, dict[str, np.ndarray]] = {}

    for case in cases:
        path_to_pr_auc: str = os.path.join(path_to_pr_aucs, f'{mean_type}_{case}_{proportion}.pkl')

        with open(path_to_pr_auc, 'rb') as f:
            pr_aucs: dict[str, list[float]] = pickle.load(f)

        pr_auc_data_dict[case] = {test: np.asarray(values) for test, values in pr_aucs.items()}

    return pr_auc_data_dict


def paired_wilcoxon_p_value(x: np.ndarray, y: np.ndarray) -> float:
    # Wilcoxon is undefined when every paired difference is zero (e.g. both tests reach PR AUC 1 on every
    # replicate); there is no difference to test then.
    if np.all(x - y == 0):
        return 1.0
    return scipy.stats.wilcoxon(x, y).pvalue


def draw_boxes(
        ax: Axes,
        pr_auc_data_dict: dict[str, dict[str, np.ndarray]],
        cases: list[str],
        statistical_tests: list[str],
        x_positions: np.ndarray,
        x_offset_per_test: list[float],
        box_width: float,
        show_fliers: bool = True,
):
    for test_idx, test in enumerate(statistical_tests):
        ax.boxplot(
            [pr_auc_data_dict[case][test] for case in cases],
            positions=x_positions + x_offset_per_test[test_idx],
            widths=box_width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=statistical_test_colors[test], edgecolor='black', linewidth=0.3),
            medianprops=dict(color='black', linewidth=0.6),
            whiskerprops=dict(color='black', linewidth=0.3),
            capprops=dict(color='black', linewidth=0.3),
            flierprops=dict(marker='o', markersize=1.5, markerfacecolor='black', markeredgecolor='none'),
            showfliers=show_fliers,
        )

    ax.set_xlim(-0.5, len(cases) - 0.5)
    ax.grid(axis='y', color='black', linewidth=0.3, alpha=0.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)


def draw_significance_brackets(
        ax: Axes,
        pr_auc_data_dict: dict[str, dict[str, np.ndarray]],
        cases: list[str],
        statistical_tests: list[str],
        x_positions: np.ndarray,
        x_offset_per_test: list[float],
        bracket_y_positions: list[float],
        font_size: float,
):
    # Paired over the dataset replicates (the same simulated datasets are scored by every test);
    # BH across the pairwise comparisons at each log2FC.
    n_tests: int = len(statistical_tests)
    test_pairs: list[tuple[int, int]] = [(i, j) for i in range(n_tests) for j in range(i + 1, n_tests)]

    for case, x in zip(cases, x_positions):
        p_values = [
            paired_wilcoxon_p_value(
                pr_auc_data_dict[case][statistical_tests[i]],
                pr_auc_data_dict[case][statistical_tests[j]],
            )
            for i, j in test_pairs
        ]
        p_values = scipy.stats.false_discovery_control(p_values, method='bh')

        for (i, j), p_value, y in zip(test_pairs, p_values, bracket_y_positions):
            if p_value < 0.05:
                x_0 = x + x_offset_per_test[i]
                x_1 = x + x_offset_per_test[j]

                ax.plot([x_0, x_1], [y, y], color='black', linewidth=0.5)
                ax.text(
                    (x_0 + x_1) / 2, y, convert_p_value_string_to_stars(p_value),
                    ha='center', va='bottom', fontsize=font_size * 0.8
                )


def draw_axis_break_marks(ax_upper: Axes, ax_lower: Axes):
    # Short diagonal ticks on the left spine where the y-axis is cut.
    marker_kwargs = dict(
        marker=[(-1, -0.5), (1, 0.5)], markersize=6, linestyle='none', color='black', mec='black', mew=0.8,
        clip_on=False,
    )
    ax_upper.plot([0], [0], transform=ax_upper.transAxes, **marker_kwargs)
    ax_lower.plot([0], [1], transform=ax_lower.transAxes, **marker_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description='PR AUC vs case log2FC at a fixed proportion of affected donors, one box per statistical test.'
    )

    parser.add_argument('--path_to_pr_aucs', type=str, required=True)
    parser.add_argument('--path_to_output_figure', type=str, required=True)
    # Kept as a string so it matches the pickle filenames exactly ('1.0', not '1').
    parser.add_argument('--proportion', type=str, default='1.0')

    parser.add_argument('--mean_type', type=str, default='tri_mean_0')
    parser.add_argument('--cases', nargs='+', type=str, default=['CASE_0.3', 'CASE_0.5', 'CASE_1', 'CASE_1.5'])
    parser.add_argument('--statistical_tests', nargs='+', type=str, default=['KS', 'CVM', 'MannWhitneyU'])
    parser.add_argument('--n_donors_per_group', type=int, default=80)

    parser.add_argument('--width_in_cm', type=float, default=8)
    parser.add_argument('--height_in_cm', type=float, default=7)
    parser.add_argument('--font_size', type=float, default=8)

    # Broken y-axis: a short lower panel anchored at 0 and a taller upper panel zoomed on the high PR AUCs,
    # with the range in between cut out.
    parser.add_argument('--broken_y_axis', action='store_true')
    parser.add_argument('--broken_y_lower_max', type=float, default=0.2)
    parser.add_argument('--broken_y_upper_min', type=float, default=0.6)
    parser.add_argument('--broken_y_height_ratio', type=float, default=4,
                        help='Height of the upper panel relative to the lower panel.')

    parser.add_argument('--plot_significant_difference', action='store_true')
    parser.add_argument('--hide_legend', action='store_true')

    args = parser.parse_args()

    font_size: float = args.font_size
    cases: list[str] = args.cases
    statistical_tests: list[str] = args.statistical_tests

    pr_auc_data_dict = load_pr_aucs(
        path_to_pr_aucs=args.path_to_pr_aucs,
        mean_type=args.mean_type,
        cases=cases,
        proportion=args.proportion,
    )

    if args.broken_y_axis:
        # A value inside the cut would silently vanish from the figure.
        all_values: np.ndarray = np.concatenate([values for case in cases for values in pr_auc_data_dict[case].values()])
        in_gap: np.ndarray = (all_values > args.broken_y_lower_max) & (all_values < args.broken_y_upper_min)
        if np.any(in_gap):
            raise ValueError(
                f'{in_gap.sum()} PR AUCs fall inside the cut ({args.broken_y_lower_max}, {args.broken_y_upper_min}) '
                f'at proportion {args.proportion}; widen the panels or drop --broken_y_axis.'
            )

    n_tests: int = len(statistical_tests)
    x_positions: np.ndarray = np.arange(len(cases))
    box_width: float = 0.8 / n_tests
    # Centre the per-test boxes on each log2FC so the group stays symmetric for any number of tests.
    x_offset_per_test: list[float] = [-0.4 + box_width * (i + 0.5) for i in range(n_tests)]

    # Drawn plain and with the PR AUC of every dataset replicate as a beeswarm over the boxes (fliers hidden, as
    # they would be drawn twice).
    for path_to_output_figure, show_points in (
            (args.path_to_output_figure, False), (with_points_path(args.path_to_output_figure), True),
    ):
        fig_size = (args.width_in_cm * cm, args.height_in_cm * cm)
        if args.broken_y_axis:
            fig, (ax_upper, ax_lower) = plt.subplots(
                2, 1, sharex=True, figsize=fig_size, layout='constrained',
                gridspec_kw=dict(height_ratios=[args.broken_y_height_ratio, 1]),
            )
            fig.get_layout_engine().set(hspace=0.04)
            axes: list[Axes] = [ax_upper, ax_lower]
        else:
            fig, ax_upper = plt.subplots(1, 1, figsize=fig_size)
            ax_lower = ax_upper
            axes = [ax_upper]

        for ax in axes:
            draw_boxes(
                ax, pr_auc_data_dict, cases, statistical_tests, x_positions, x_offset_per_test, box_width,
                show_fliers=not show_points,
            )

        y_max: float = 1.0
        if args.plot_significant_difference:
            n_pairs: int = n_tests * (n_tests - 1) // 2
            # Bracket spacing in data units, scaled to the visible y-range of the panel they sit in.
            y_min_upper: float = args.broken_y_upper_min if args.broken_y_axis else 0.0
            step: float = 0.08 * (1.0 - y_min_upper)
            bracket_y_positions: list[float] = [1.0 + 0.4 * step + step * k for k in range(n_pairs)]
            y_max = bracket_y_positions[-1] + step

            draw_significance_brackets(
                ax_upper, pr_auc_data_dict, cases, statistical_tests, x_positions, x_offset_per_test,
                bracket_y_positions, font_size,
            )

        if args.broken_y_axis:
            ax_upper.set_ylim(args.broken_y_upper_min, y_max)
            upper_ticks: np.ndarray = np.round(np.arange(args.broken_y_upper_min, 1.0 + 1e-9, 0.1), 1)
            ax_upper.set_yticks(upper_ticks)
            ax_upper.set_yticklabels(upper_ticks, fontsize=font_size)

            ax_lower.set_ylim(0, args.broken_y_lower_max)
            ax_lower.set_yticks([0, args.broken_y_lower_max])
            ax_lower.set_yticklabels([0, args.broken_y_lower_max], fontsize=font_size)

            ax_upper.spines['bottom'].set_visible(False)
            ax_upper.tick_params(axis='x', bottom=False)
            draw_axis_break_marks(ax_upper, ax_lower)
        else:
            ax_upper.set_ylim(0, y_max)
            ax_upper.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax_upper.set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size)

        ax_lower.set_xticks(x_positions)
        ax_lower.set_xticklabels([case_l2fc_tick_labels.get(case, case) for case in cases], fontsize=font_size)
        ax_lower.set_xlabel('Case log2FC', fontsize=font_size)

        affected_donors: int = round(float(args.proportion) * args.n_donors_per_group)
        ax_upper.set_title(
            f'{convert_mean_type_label(args.mean_type)} · affected donors: {affected_donors}/{args.n_donors_per_group}',
            fontsize=font_size
        )

        if not args.hide_legend:
            ax_upper.legend(
                handles=[
                    Patch(facecolor=statistical_test_colors[test], edgecolor='black', linewidth=0.3,
                          label=statistical_test_short[test])
                    for test in statistical_tests
                ],
                loc='lower right',
                fontsize=font_size,
                frameon=False,
            )

        if args.broken_y_axis:
            fig.supylabel('PR AUC', fontsize=font_size)
        else:
            ax_upper.set_ylabel('PR AUC', fontsize=font_size)
            fig.tight_layout()

        if show_points:
            for ax in axes:
                for test_idx, test in enumerate(statistical_tests):
                    for case, x in zip(cases, x_positions):
                        draw_beeswarm(
                            ax, x + x_offset_per_test[test_idx], pr_auc_data_dict[case][test], color='black',
                            max_half_width=box_width * 0.4, marker_size=1.8, alpha=0.7, edgecolor='white', zorder=3,
                        )

        os.makedirs(os.path.dirname(path_to_output_figure) or '.', exist_ok=True)
        fig.savefig(path_to_output_figure)
        plt.close(fig)


if __name__ == '__main__':
    main()
