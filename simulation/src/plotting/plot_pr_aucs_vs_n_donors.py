import os

import numpy as np

import matplotlib.pyplot as plt

from utils import cm, statistical_test_colors, load_curve_data, convert_mean_type_label, convert_cases_l2fc_label

import argparse


def load_curve_data_dict(
        path_to_curves: str,
        path_to_pr_aucs: str,
        path_to_error_bands_dict: str,
        mean_types: list[str],
        n_donors: list[int],
        case: str
):
    curve_data_dict = {}
    pr_auc_data_dict = {}
    error_bands_data_dict = {}

    for mean_type in mean_types:
        curve_data_dict[mean_type] = {}
        pr_auc_data_dict[mean_type] = {}
        error_bands_data_dict[mean_type] = {}

        for n in n_donors:
            path_to_curve: str = os.path.join(path_to_curves, f'{mean_type}_{case}_{n}.pkl')
            path_to_pr_auc: str = os.path.join(path_to_pr_aucs, f'{mean_type}_{case}_{n}.pkl')
            path_to_error_band_dict: str = os.path.join(path_to_error_bands_dict, f'{mean_type}_{case}_{n}.pkl')

            curves, pr_aucs, error_bands = load_curve_data(
                path_to_curves=path_to_curve,
                path_to_pr_aucs=path_to_pr_auc,
                path_to_error_bands_dict=path_to_error_band_dict,
            )

            curve_data_dict[mean_type][n] = curves
            pr_auc_data_dict[mean_type][n] = pr_aucs
            error_bands_data_dict[mean_type][n] = error_bands
    return curve_data_dict, pr_auc_data_dict, error_bands_data_dict


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_curves', type=str, required=True)
    parser.add_argument('--path_to_pr_aucs', type=str, required=True)
    parser.add_argument('--path_to_error_bands_dict', type=str, required=True)
    parser.add_argument('--path_to_output_figure', type=str, required=True)
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--mean_types', nargs='+', default=['mean_0', 'tri_mean_0'])
    parser.add_argument('--n_donors', nargs='+', type=int, default=[2, 3, 5, 8, 10, 20])
    parser.add_argument('--width_in_cm', type=float, default=14)
    parser.add_argument('--height_in_cm', type=float, default=8)
    parser.add_argument('--font_size', type=float, default=8)
    parser.add_argument('--plot_with_offset', type=bool, default=True)
    parser.add_argument('--plot_case_title', type=bool, default=True)
    parser.add_argument('--plot_legend', type=bool, default=True)

    args = parser.parse_args()

    path_to_curves: str = args.path_to_curves
    path_to_pr_aucs: str = args.path_to_pr_aucs
    path_to_error_bands_dict: str = args.path_to_error_bands_dict
    path_to_output_figure: str = args.path_to_output_figure

    mean_types: list[str] = args.mean_types
    n_donors: list[int] = args.n_donors
    case: str = args.case

    fig_size = (args.width_in_cm * cm, args.height_in_cm * cm)
    font_size = args.font_size

    plot_with_offset: bool = args.plot_with_offset
    plot_case_title: bool = args.plot_case_title
    plot_legend: bool = args.plot_legend

    case_l2fc_label = convert_cases_l2fc_label(case)

    statistical_tests: list[str] = ['KS', 'Anderson', 'CVM', 'MannWhitneyU']
    n_tests: int = len(statistical_tests)
    n_points: int = len(n_donors)

    colors = [statistical_test_colors[statistical_test] for statistical_test in statistical_tests]

    _, pr_auc_data_dict, _ = load_curve_data_dict(
        path_to_curves=path_to_curves,
        path_to_pr_aucs=path_to_pr_aucs,
        path_to_error_bands_dict=path_to_error_bands_dict,
        mean_types=mean_types,
        n_donors=n_donors,
        case=case,
    )

    fig, axs = plt.subplots(1, len(mean_types), figsize=fig_size, squeeze=False)
    axs = axs[0]

    box_width = 0.8 / n_tests
    x_positions: np.ndarray = np.arange(n_points) * 1.5
    if plot_with_offset:
        x_offset_per_test: list[float] = [-0.25 + i * box_width for i in range(n_tests)]
    else:
        x_offset_per_test = [0] * n_tests

    rng = np.random.default_rng(0)

    for i, mean_type in enumerate(mean_types):
        mean_type_label: str = convert_mean_type_label(mean_type)

        for j, statistical_test in enumerate(statistical_tests):
            mean_aucs: list[float] = []
            std_aucs: list[float] = []

            for n in n_donors:
                replicate_aucs = pr_auc_data_dict[mean_type][n][statistical_test]
                mean_aucs.append(np.mean(replicate_aucs))
                std_aucs.append(np.std(replicate_aucs))

            x_for_test = np.array([x + x_offset_per_test[j] for x in x_positions])

            axs[i].errorbar(
                x=x_for_test,
                y=mean_aucs,
                yerr=std_aucs,
                label=statistical_test,
                color=colors[j],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
                zorder=2,
            )

            # Overlay individual replicate PR AUCs (editorial requirement: show data distribution)
            for point_idx, n in enumerate(n_donors):
                replicate_aucs = pr_auc_data_dict[mean_type][n][statistical_test]
                jitter = (rng.random(len(replicate_aucs)) - 0.5) * box_width * 0.6
                axs[i].scatter(
                    x_for_test[point_idx] + jitter,
                    replicate_aucs,
                    color=colors[j],
                    s=2,
                    alpha=0.4,
                    linewidths=0,
                    zorder=1,
                )

        axs[i].set_xticks(x_positions)
        # Label ticks with the per-group donor count (n_donors covers both sexes).
        axs[i].set_xticklabels([int(n * 2) for n in n_donors], fontsize=font_size)

        axs[i].set_ylim([0, 1])
        axs[i].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1])
        if i == 0:
            axs[i].set_yticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=font_size)
        else:
            axs[i].set_yticklabels([])

        axs[i].set_title(mean_type_label, fontsize=font_size)

        if i == 0:
            if plot_legend:
                axs[i].legend(loc='lower right', fontsize=font_size)
            axs[i].set_ylabel('PR AUC', fontsize=font_size)

    if plot_case_title:
        fig.text(0.5, 1 - 0.04, case_l2fc_label, ha='center', va='center', fontsize=font_size)

    fig.text(0.5, 0.04, 'Donors per group (cases = controls)', ha='center', va='center', fontsize=font_size)

    fig.subplots_adjust(left=0.07, right=0.99, top=1 - 0.14, bottom=0.14, hspace=0, wspace=0.05)

    plt.tight_layout()
    plt.savefig(path_to_output_figure)
    plt.close(fig)


if __name__ == '__main__':
    main()
