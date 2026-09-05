import os

import numpy as np

import scipy

import matplotlib.pyplot as plt

from utils import cm, statistical_test_colors, load_curve_data, convert_mean_type_label, convert_p_value_string_to_stars, convert_cases_l2fc_label

import argparse


def load_curve_data_dict(
        path_to_curves: str,
        path_to_pr_aucs: str,
        path_to_error_bands_dict: str,
        mean_types: list[str],
        case: str

):
    curve_data_dict = {}
    pr_auc_data_dict = {}
    error_bands_data_dict = {}

    for mean_type in mean_types:
        curve_data_dict[mean_type] = {}
        pr_auc_data_dict[mean_type] = {}
        error_bands_data_dict[mean_type] = {}

        for proportion in [0.1, 0.3, 0.5, 0.8, 1.0]:
            path_to_curve: str = os.path.join(path_to_curves, f'{mean_type}_{case}_{proportion}.pkl')
            path_to_pr_auc: str = os.path.join(path_to_pr_aucs, f'{mean_type}_{case}_{proportion}.pkl')
            path_to_error_band_dict: str = os.path.join(path_to_error_bands_dict, f'{mean_type}_{case}_{proportion}.pkl')

            curves, pr_aucs, error_bands = load_curve_data(
                path_to_curves=path_to_curve,
                path_to_pr_aucs=path_to_pr_auc,
                path_to_error_bands_dict=path_to_error_band_dict,
            )

            curve_data_dict[mean_type][proportion] = curves
            pr_auc_data_dict[mean_type][proportion] = pr_aucs
            error_bands_data_dict[mean_type][proportion] = error_bands
    return curve_data_dict, pr_auc_data_dict, error_bands_data_dict

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_curves',
        type=str,
        required=True
    )
    parser.add_argument(
        '--path_to_pr_aucs',
        type=str,
        required=True
    )
    parser.add_argument(
        '--path_to_error_bands_dict',
        type=str,
        required=True
    )

    parser.add_argument(
        '--path_to_output_figure',
        type=str,
        required=True
    )


    parser.add_argument(
        '--case',
        type=str,
        required=True
    )

    parser.add_argument(
        '--mean_types',
        nargs='+',
        default=['mean_0', 'trim_mean_0.05', 'trim_mean_0.1', 'tri_mean_0']
    )
    parser.add_argument(
        '--proportions',
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.8, 1.0]
    )

    parser.add_argument(
        '--width_in_cm',
        type=float,
        default=20
    )
    parser.add_argument(
        '--height_in_cm',
        type=float,
        default=10
    )

    parser.add_argument(
        '--font_size',
        type=float,
        default=8
    )

    parser.add_argument(
        '--plot_with_offset',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--plot_case_title',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--plot_significant_difference',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--plot_vertical_separators',
        type=bool,
        default=True
    )
    parser.add_argument(
        '--plot_legend',
        type=bool,
        default=True
    )

    args = parser.parse_args()

    path_to_curves: str = args.path_to_curves
    path_to_pr_aucs: str = args.path_to_pr_aucs
    path_to_error_bands_dict: str = args.path_to_error_bands_dict

    path_to_output_figure: str = args.path_to_output_figure

    mean_types: list[str] = args.mean_types
    proportions: list[float] = args.proportions
    case: str = args.case

    fig_size = (args.width_in_cm * cm, args.height_in_cm * cm)
    font_size = args.font_size

    plot_with_offset: bool = args.plot_with_offset
    plot_case_title: bool = args.plot_case_title
    plot_significant_difference: bool = args.plot_significant_difference
    plot_vertical_separators: bool = args.plot_vertical_separators
    plot_legend: bool = args.plot_legend

    case_l2fc_label = convert_cases_l2fc_label(case)

    n_tests: int = 3
    n_proportions: int = len(proportions)

    colors = [statistical_test_colors[statistical_test] for statistical_test in ['KS', 'CVM', 'MannWhitneyU']]

    # Three tests (KS, CVM, MannWhitneyU) → three pairwise comparisons. Indices are into the test order
    # ['KS', 'CVM', 'MannWhitneyU'] used for `colors` and `x_offset_per_test`.
    p_value_order: list[tuple[str, str]] = [('KS', 'CVM'), ('KS', 'MannWhitneyU'), ('CVM', 'MannWhitneyU')]
    p_value_order_idx: list[tuple[int, int]] = [(0, 1), (0, 2), (1, 2)]
    p_value_y_positions: list[float] = [0.78, 0.82, 0.86]

    curves_data_dict: dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]
    pr_auc_data_dict: dict[str, dict[str, dict[str, list[float] | np.array]]]
    error_bands_data_dict: dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]

    curves_data_dict, pr_auc_data_dict, error_bands_data_dict = load_curve_data_dict(
        path_to_curves=path_to_curves,
        path_to_pr_aucs=path_to_pr_aucs,
        path_to_error_bands_dict=path_to_error_bands_dict,
        mean_types=mean_types,
        case=case,
    )

    fig, axs = plt.subplots(1, len(mean_types), figsize=fig_size)

    box_width = 0.8 / n_tests
    x_positions: np.ndarray = np.arange(n_proportions) * 1.5
    if plot_with_offset:
        # Centre the per-test markers on each proportion so the trio stays symmetric for any n_tests.
        x_offset_per_test: list[float] = [-0.4 + box_width * (i + 0.5) for i in range(n_tests)]
    else:
        x_offset_per_test = [0] * n_tests

    for i, mean_type in enumerate(mean_types):
        mean_type_label: str = convert_mean_type_label(mean_type)

        mean_aucs_per_statistical_test: dict[str, list[float]] = {}
        std_aucs_per_statistical_test: dict[str, list[float]] = {}

        # This dictionary contains the p-values for paired comparisons between the statistical tests
        # The order is as follows: KS-CVM, KS-MannWhitney, CVM-MannWhitney
        p_values: dict[str, list[float]] = {}

        for proportion in proportions:
            # Compute the mean and std of the PR AUCs for each statistical test
            for statistical_test in pr_auc_data_dict[mean_type][proportion].keys():
                # Initialize the lists in the dictionaries
                if statistical_test not in mean_aucs_per_statistical_test:
                    mean_aucs_per_statistical_test[statistical_test] = []
                    std_aucs_per_statistical_test[statistical_test] = []

                mean_aucs_per_statistical_test[statistical_test].append(
                    np.mean(pr_auc_data_dict[mean_type][proportion][statistical_test])
                )
                std_aucs_per_statistical_test[statistical_test].append(
                    np.std(pr_auc_data_dict[mean_type][proportion][statistical_test])
                )

            # Compute whether the difference is significant
            # The measurments can be considered to be paired
            p_values[proportion] = []
            if plot_significant_difference:
                for statistical_test1, statistical_test2 in p_value_order:
                    res = scipy.stats.wilcoxon(
                        pr_auc_data_dict[mean_type][proportion][statistical_test1],
                        pr_auc_data_dict[mean_type][proportion][statistical_test2],
                    )
                    p_value = res.pvalue
                    p_values[proportion].append(p_value)

                # Correct the p-values for multiple comparisons
                p_values[proportion] = scipy.stats.false_discovery_control(p_values[proportion], method='bh')

        for j, statistical_test in enumerate(mean_aucs_per_statistical_test.keys()):
            axs[i].errorbar(
                x=[x + x_offset_per_test[j] for x in x_positions],
                y=mean_aucs_per_statistical_test[statistical_test],
                yerr=std_aucs_per_statistical_test[statistical_test],
                label=statistical_test,
                color=colors[j],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
            )

        axs[i].set_xticks(x_positions)
        axs[i].set_xticklabels([int(80 * float(x)) for x in proportions], fontsize=font_size)

        axs[i].set_ylim([0, 1])

        axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        if i == 0:
            axs[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=font_size)
        else:
            axs[i].set_yticklabels([])

        axs[i].set_title(mean_type_label, fontsize=font_size)

        if i == 0:
            if plot_legend:
                axs[i].legend(loc='upper left', fontsize=font_size)

            axs[i].set_ylabel('PR AUC', fontsize=font_size)

        if plot_significant_difference:
            for proportion_i, (proportion, p_values_list) in enumerate(p_values.items()):
                for j, p_value in enumerate(p_values_list):
                    if p_value < 0.05:
                        p_value_text = convert_p_value_string_to_stars(p_value)

                        # draw a line for the tests
                        x_0 = x_positions[proportion_i] + x_offset_per_test[p_value_order_idx[j][0]]
                        x_1 = x_positions[proportion_i] + x_offset_per_test[p_value_order_idx[j][1]]
                        y_0 = y_1 = p_value_y_positions[j]

                        axs[i].plot([x_0, x_1], [y_0, y_1], color='black')
                        axs[i].text((x_0 + x_1) / 2, y_0, p_value_text, ha='center')

        if plot_vertical_separators:
            axs[i].axvline(x=x_positions[1] - 0.75, color='black', linestyle='--', linewidth=0.3, alpha=0.3)
            axs[i].axvline(x=x_positions[2] - 0.75, color='black', linestyle='--', linewidth=0.3, alpha=0.3)
            axs[i].axvline(x=x_positions[3] - 0.75, color='black', linestyle='--', linewidth=0.3, alpha=0.3)
            axs[i].axvline(x=x_positions[4] - 0.75, color='black', linestyle='--', linewidth=0.3, alpha=0.3)

    if plot_case_title:
        # fig.suptitle(case_l2fc_label, fontsize=font_size)
        fig.text(0.5, 1 - 0.04, case_l2fc_label, ha='center', va='center', fontsize=font_size)

    fig.text(0.5, 0.04, 'Affected donors (out of 80)', ha='center', va='center', fontsize=font_size)

    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.14, bottom=0.14, hspace=0, wspace=0.05)

    plt.tight_layout()
    plt.savefig(path_to_output_figure)
    plt.close(fig)
    
if __name__ == '__main__':
    main()