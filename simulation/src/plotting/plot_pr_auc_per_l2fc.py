import os

import numpy as np

import scipy

from utils import cm, load_curve_data, convert_p_value_string_to_stars

import matplotlib.pyplot as plt


def load_curves_pr_auc_and_error_bands(
        path_to_curves: str,
        path_to_aucs: str,
        path_to_error_bands_dict: str,
        mean_types: list[str] = ['mean_0', 'tri_mean_0', 'trim_mean_0.05', 'trim_mean_0.1'],
        cases_l2fcs: list[str] = ['CASE_0.3', 'CASE_0.5', 'CASE_1', 'CASE_1.5'],
        proportions: list[float] = [0.1, 0.3, 0.5, 0.8, 1.0]
):
    curves_data_dict: dict[str, dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]] = {}
    pr_auc_data_dict: dict[str, dict[str, dict[str, dict[str, list[float] | np.array]]]] = {}
    error_bands_data_dict: dict[str, dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]] = {}

    for mean_type in mean_types:
        curves_data_dict[mean_type] = {}
        pr_auc_data_dict[mean_type] = {}
        error_bands_data_dict[mean_type] = {}

        for cases_l2fc in cases_l2fcs:
            curves_data_dict[mean_type][cases_l2fc] = {}
            pr_auc_data_dict[mean_type][cases_l2fc] = {}
            error_bands_data_dict[mean_type][cases_l2fc] = {}

            for proportion in proportions:
                path_to_curve: str = os.path.join(path_to_curves, f'{mean_type}_{cases_l2fc}_{proportion}.pkl')
                path_to_auc: str = os.path.join(path_to_aucs, f'{mean_type}_{cases_l2fc}_{proportion}.pkl')
                path_to_error_band_dict: str = os.path.join(path_to_error_bands_dict, f'{mean_type}_{cases_l2fc}_{proportion}.pkl')

                curve, auc, error_bands_dict = load_curve_data(
                    path_to_curves=path_to_curve,
                    path_to_pr_aucs=path_to_auc,
                    path_to_error_bands_dict=path_to_error_band_dict,
                )

                curves_data_dict[mean_type][cases_l2fc][proportion] = curve
                pr_auc_data_dict[mean_type][cases_l2fc][proportion] = auc
                error_bands_data_dict[mean_type][cases_l2fc][proportion] = error_bands_dict

    return curves_data_dict, pr_auc_data_dict, error_bands_data_dict


def main():
    plot_significant_difference: bool = True
    plot_legend: bool = False

    path_to_curves: str = '/Users/gianluca.volkmer/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/Projects/MultiNeuronChat/Code_Clean/validation/simulation/results/figs/pr_curves/summary_data/curves'
    path_to_aucs: str = '/Users/gianluca.volkmer/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/Projects/MultiNeuronChat/Code_Clean/validation/simulation/results/figs/pr_curves/summary_data/pr_aucs'
    path_to_error_bands_dict: str = '/Users/gianluca.volkmer/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/Projects/MultiNeuronChat/Code_Clean/validation/simulation/results/figs/pr_curves/summary_data/error_bands'

    path_to_figs: str = '/Users/gianluca.volkmer/Library/CloudStorage/OneDrive-KarolinskaInstitutet/Documents/Projects/MultiNeuronChat/Code_Clean/validation/simulation/results/figs/pr_aucs_per_l2fc'
    os.makedirs(path_to_figs, exist_ok=True)

    width_in_cm: float = 7
    height_in_cm: float = 7
    font_size: float = 8

    colors = ['#ca0020', '#f4a582', '#92c5de', '#0571b0']

    p_value_order: list[tuple[str, str]] = [
        ('KS', 'Anderson'), ('CVM', 'MannWhitneyU'), ('Anderson', 'CVM'),
        ('KS', 'CVM'), ('Anderson', 'MannWhitneyU'), ('KS', 'MannWhitneyU')
    ]
    p_value_order_idx: list[tuple[int, int]] = [(0, 1), (2, 3), (1, 2), (0, 2), (1, 3), (0, 3)]
    p_value_y_positions: list[float] = [0.9, 0.9, 0.93, 0.96, 0.99, 1.02]

    fig_size = (width_in_cm * cm, height_in_cm * cm)

    mean_types: list[str] = ['mean_0', 'tri_mean_0', 'trim_mean_0.05', 'trim_mean_0.1']
    cases_l2fcs: list[str] = ['CASE_0.3', 'CASE_0.5', 'CASE_1', 'CASE_1.5']
    proportions: list[float] = [0.1, 0.3, 0.5, 0.8, 1.0]

    curves_data_dict: dict[str, dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]]
    pr_auc_data_dict: dict[str, dict[str, dict[str, dict[str, list[float] | np.array]]]]
    error_bands_data_dict: dict[str, dict[str, dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]]]]

    curves_data_dict, pr_auc_data_dict, error_bands_data_dict = load_curves_pr_auc_and_error_bands(
        path_to_curves=path_to_curves,
        path_to_aucs=path_to_aucs,
        path_to_error_bands_dict=path_to_error_bands_dict,
    )

    tests: list[str] = list(pr_auc_data_dict[mean_types[0]][cases_l2fcs[0]][proportions[0]].keys())

    n_dots_per_box: int = 4
    x_positions = np.arange(4)
    box_width = 0.8 / n_dots_per_box
    x_offset_per_test = [-0.25 + i * box_width for i in range(n_dots_per_box)]

    for mean_type in mean_types:
        path_to_figs_mean_type = os.path.join(path_to_figs, mean_type)
        os.makedirs(path_to_figs_mean_type, exist_ok=True)

        for proportions_idx, proportion in enumerate(proportions):
            fig, axs = plt.subplots(1, 1, figsize=fig_size)

            for case_idx, case_l2fc in enumerate(cases_l2fcs):

                for test_idx, test in enumerate(tests):
                    boxplot = axs.boxplot(
                        [pr_auc_data_dict[mean_type][case_l2fc][proportion][test]],
                        positions=[x_positions[case_idx] + x_offset_per_test[test_idx]],
                        widths=[box_width],
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[test_idx], color='black', linewidth=0.3),
                        medianprops=dict(color='black', linewidth=0.3),
                        whiskerprops=dict(color='black', linewidth=0.3),
                        capprops=dict(color='black', linewidth=0.3),
                        flierprops=dict(marker='o', markersize=1, markerfacecolor='black', markeredgecolor='black'),
                    )

                    for patch in boxplot['boxes']:
                        patch.set_facecolor(colors[test_idx])

                p_values = []
                if plot_significant_difference:
                    for statistical_test1, statistical_test2 in p_value_order:
                        res = scipy.stats.wilcoxon(
                            pr_auc_data_dict[mean_type][case_l2fc][proportion][statistical_test1],
                            pr_auc_data_dict[mean_type][case_l2fc][proportion][statistical_test2],
                        )
                        p_value = res.pvalue
                        p_values.append(p_value)

                    # Correct the p-values for multiple comparisons
                    p_values = scipy.stats.false_discovery_control(p_values, method='bh')

                for p_value_idx, p_value in enumerate(p_values):
                    if p_value < 0.05:
                        x_0 = x_positions[case_idx] + x_offset_per_test[p_value_order_idx[p_value_idx][0]]
                        x_1 = x_positions[case_idx] + x_offset_per_test[p_value_order_idx[p_value_idx][1]]

                        y = p_value_y_positions[p_value_idx]

                        stars = convert_p_value_string_to_stars(p_value)

                        axs.plot([x_0, x_1], [y, y], color='black', linewidth=0.5)
                        axs.text((x_0 + x_1) / 2, y, stars, ha='center', va='center', fontsize=font_size * 0.8)

            axs.set_xticks(x_positions)
            axs.set_xticklabels(['0.3', '0.5', '1.0', '1.5'], fontsize=font_size)

            axs.text(
                x_positions[1] + (x_positions[2] - x_positions[1]) / 2,
                -0.15,
                "log2FC",
                ha='center',
                va='center',
                fontsize=font_size
            )

            axs.set_ylabel('PR AUC', fontsize=font_size)
            axs.set_ylim(0, 1.1)
            axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            axs.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=font_size)

            affected_donors: int = int(float(proportion) * 80)

            axs.set_title(f'Affected donors: {affected_donors}/80', fontsize=font_size)

            if plot_legend:
                axs.legend(
                    [f'{test}' for test in tests],
                    loc='upper left',
                    fontsize=font_size
                )

            # fig.subplots_adjust(left=0.1, right=0.99, top=0.95, bottom=0.1)
            fig.tight_layout()

            fig.savefig(os.path.join(path_to_figs_mean_type, f'pr_auc_errorbars_{proportion}.pdf'))
            plt.close(fig)


if __name__ == '__main__':
    main()