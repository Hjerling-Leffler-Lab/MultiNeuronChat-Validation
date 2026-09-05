import numpy as np
import pandas as pd

from utils import cm, statistical_test_colors, load_curve_data

from scipy import stats

import matplotlib.pyplot as plt

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_curves',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_aucs',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_error_bands_dict',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--path_to_summary_pr_curves_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_curves_KS_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_curves_CVM_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_curves_MannWhitneyU_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_aucs_with_significance_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_pr_aucs_fig',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--pr_curves_width_of_figure_in_cm',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--pr_curves_height_of_figure_in_cm',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--pr_aucs_width_of_figure_in_cm',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--pr_aucs_height_of_figure_in_cm',
        type=float,
        required=True,
    )

    parser.add_argument(
        '--font_size',
        type=float,
        required=True,
    )

    args = parser.parse_args()

    path_to_curves: str = args.path_to_curves
    path_to_pr_aucs: str = args.path_to_pr_aucs
    path_to_error_bands_dict: str = args.path_to_error_bands_dict

    path_to_summary_pr_curves_fig: str = args.path_to_summary_pr_curves_fig

    path_to_pr_curves_KS_fig: str = args.path_to_pr_curves_KS_fig
    path_to_pr_curves_CVM_fig: str = args.path_to_pr_curves_CVM_fig
    path_to_pr_curves_MannWhitneyU_fig: str = args.path_to_pr_curves_MannWhitneyU_fig

    path_to_pr_aucs_with_significance_fig: str = args.path_to_pr_aucs_with_significance_fig
    path_to_pr_aucs_fig: str = args.path_to_pr_aucs_fig

    pr_curves_width_of_figure_in_cm: float = args.pr_curves_width_of_figure_in_cm
    pr_curves_height_of_figure_in_cm: float = args.pr_curves_height_of_figure_in_cm

    pr_aucs_width_of_figure_in_cm: float = args.pr_aucs_width_of_figure_in_cm
    pr_aucs_height_of_figure_in_cm: float = args.pr_aucs_height_of_figure_in_cm

    font_size: float = args.font_size

    curve_label_to_fig_path: dict[str, str] = {
        'KS': path_to_pr_curves_KS_fig,
        'CVM': path_to_pr_curves_CVM_fig,
        'MannWhitneyU': path_to_pr_curves_MannWhitneyU_fig,
    }

    colors = [statistical_test_colors[statistical_test] for statistical_test in ['KS', 'CVM', 'MannWhitneyU']]

    pr_curves_fig_size = (pr_curves_width_of_figure_in_cm*cm, pr_curves_height_of_figure_in_cm*cm)
    pr_auc_fig_size = (pr_aucs_width_of_figure_in_cm*cm, pr_aucs_height_of_figure_in_cm*cm)

    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
    pr_aucs: dict[str, list[float] | np.array]
    error_bands_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    curves, pr_aucs, error_bands_dict = load_curve_data(
        path_to_curves=path_to_curves,
        path_to_pr_aucs=path_to_pr_aucs,
        path_to_error_bands_dict=path_to_error_bands_dict,
    )

    curve_names: list[str] = list(curves.keys())
    mean_precisions: list[np.ndarray] = [error_bands_dict[label][0] for label in curve_names]
    mean_recalls: list[np.ndarray] = [error_bands_dict[label][1] for label in curve_names]
    std_precisions: list[np.ndarray] = [error_bands_dict[label][2] for label in curve_names]
    std_recalls: list[np.ndarray] = [error_bands_dict[label][3] for label in curve_names]

    fig, axs = plt.subplots(1, 1, figsize=pr_curves_fig_size)

    for mean_precision, mean_recall, std_precision, std_recall, color, label in zip(
            mean_precisions, mean_recalls, std_precisions, std_recalls, colors, curve_names
    ):
        auc_mean = np.mean(pr_aucs[label])
        auc_std = np.std(pr_aucs[label])

        axs.plot(mean_recall, mean_precision, label=f'{label} (AUC = {auc_mean:.2f} $\pm$ {auc_std:.2f})', color=color)
        axs.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.3,
                         color=color)

    axs.set_xlabel('Recall', fontsize=font_size * 1.5)
    axs.set_ylabel('Precision', fontsize=font_size * 1.5)

    # set ticks with precision 2
    axs.set_xticks(np.round(np.arange(0, 1.01, 0.1), 2))
    axs.set_yticks(np.round(np.arange(0, 1.01, 0.1), 2))
    axs.set_xticklabels(axs.get_xticks(), fontsize=font_size)
    axs.set_yticklabels(axs.get_yticks(), fontsize=font_size)

    axs.legend(fontsize=font_size * 0.9)

    axs.set_xlim([0.0, 1.01])
    axs.set_ylim([0.0, 1.01])

    # if title_composite_plot is not None:
    #    axs.set_title(title_composite_plot, fontsize=font_size * 1.5)

    plt.tight_layout()
    plt.savefig(path_to_summary_pr_curves_fig)
    plt.close(fig)

    for mean_precision, mean_recall, std_precision, std_recall, color, label in zip(
            mean_precisions, mean_recalls, std_precisions, std_recalls, colors, curve_names
    ):
        fig, axs = plt.subplots(1, 1, figsize=pr_curves_fig_size)

        auc_mean = np.mean(pr_aucs[label])
        auc_std = np.std(pr_aucs[label])

        axs.plot(mean_recall, mean_precision, label=f'{label} (AUC = {auc_mean:.2f} $\pm$ {auc_std:.2f})', color=color)
        axs.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, alpha=0.3,
                         color=color)
        # if plot_std2_single_plot:
        #     axs.fill_between(mean_recall, mean_precision - 2 * std_precision, mean_precision + 2 * std_precision,
        #                      alpha=0.1, color=color)

        for curve in curves[label]:
            precision, recall, _ = curve
            axs.plot(recall, precision, color='grey', alpha=0.3)

        axs.set_xlabel('Recall', fontsize=font_size * 1.5)
        axs.set_ylabel('Precision', fontsize=font_size * 1.5)

        # set ticks with precision 2
        axs.set_xticks(np.round(np.arange(0, 1.01, 0.1), 2))
        axs.set_yticks(np.round(np.arange(0, 1.01, 0.1), 2))
        axs.set_xticklabels(axs.get_xticks(), fontsize=font_size)
        axs.set_yticklabels(axs.get_yticks(), fontsize=font_size)

        axs.legend(fontsize=font_size * 0.9)

        axs.set_xlim([0.0, 1.01])
        axs.set_ylim([0.0, 1.01])

        # if title_single_plot is not None:
        #    axs.set_title(f'{title_single_plot} - {label}', fontsize=font_size * 1.5)

        plt.tight_layout()
        plt.savefig(curve_label_to_fig_path[label])
        plt.close(fig)

    colors = colors[0:len(curve_names)]

    # AUC box plots
    fig, axs = plt.subplots(1, 1, figsize=pr_auc_fig_size)
    pr_aucs_values = [pr_aucs[label] for label in curve_names]

    # create dataframe
    pr_df = {
        'Method': [],
        'AUC': []
    }
    for label in curve_names:
        for x in pr_aucs[label]:
            short_label = label

            if label == 'MannWhitneyU':
                short_label = 'MWU'

            pr_df['Method'].append(short_label)
            pr_df['AUC'].append(x)
    pr_df = pd.DataFrame(pr_df)
    # sns.boxplot(x='Method', y='AUC', hue='Method', data=pr_df, ax=axs, palette=colors)

    bplot = axs.boxplot(
        pr_aucs_values,
        labels=['KS', 'CVM', 'MWU'],
        patch_artist=True,
        medianprops=dict(color='black')
    )

    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    axs.set_ylabel('AUC', fontsize=font_size * 1.5)
    axs.set_xlabel('')

    # axs.set_yticklabels(axs.get_yticks(), fontsize=font_size)
    # axs.set_xticks(['KS', 'And', 'CVM', 'Wil'])

    axs.tick_params(axis='both', which='major', labelsize=font_size)

    # if title_auc_plot is not None:
    #     axs.set_title(title_auc_plot, fontsize=font_size * 1.5)

    # Add significance
    for label in curve_names:
        pr_aucs[label] = np.array(pr_aucs[label])

    if not np.any(pr_aucs['KS'] - pr_aucs['CVM']):
        ks_cvm = 1.0
    else:
        ks_cvm = stats.wilcoxon(pr_aucs['KS'], pr_aucs['CVM'], zero_method='wilcox',
                                method=stats.PermutationMethod()).pvalue
    if not np.any(pr_aucs['KS'] - pr_aucs['MannWhitneyU']):
        ks_mannwhitneyu = 1.0
    else:
        ks_mannwhitneyu = stats.wilcoxon(pr_aucs['KS'], pr_aucs['MannWhitneyU'], zero_method='wilcox',
                                         method=stats.PermutationMethod()).pvalue
    if not np.any(pr_aucs['CVM'] - pr_aucs['MannWhitneyU']):
        cvm_mannwhitneyu = 1.0
    else:
        cvm_mannwhitneyu = stats.wilcoxon(pr_aucs['CVM'], pr_aucs['MannWhitneyU'], zero_method='wilcox',
                                          method=stats.PermutationMethod()).pvalue

    # if nan set to 1
    ks_cvm = 1.0 if np.isnan(ks_cvm) else ks_cvm
    ks_mannwhitneyu = 1.0 if np.isnan(ks_mannwhitneyu) else ks_mannwhitneyu
    cvm_mannwhitneyu = 1.0 if np.isnan(cvm_mannwhitneyu) else cvm_mannwhitneyu

    # adjust p-values for multiple testing
    [ks_cvm, ks_mannwhitneyu, cvm_mannwhitneyu] = stats.false_discovery_control(
        [ks_cvm, ks_mannwhitneyu, cvm_mannwhitneyu],
        method='bh'
    )

    ks_cvm_text = ['n.s.', '*', '**', '***'][int(ks_cvm < 0.05) + int(ks_cvm < 0.01) + int(ks_cvm < 0.001)]
    ks_mannwhitneyu_text = ['n.s.', '*', '**', '***'][
        int(ks_mannwhitneyu < 0.05) + int(ks_mannwhitneyu < 0.01) + int(ks_mannwhitneyu < 0.001)]
    cvm_mannwhitneyu_text = ['n.s.', '*', '**', '***'][
        int(cvm_mannwhitneyu < 0.05) + int(cvm_mannwhitneyu < 0.01) + int(cvm_mannwhitneyu < 0.001)]

    # get x_tick locations (one per test: KS, CVM, MWU)
    x_ticks = axs.get_xticks()

    x1 = x_ticks[0]
    x2 = x_ticks[1]
    x3 = x_ticks[2]

    y = 1
    h = 0.07
    col = 'black'

    axs.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(path_to_pr_aucs_fig)

    # Significance ladder: adjacent pairs on the first rung, the KS–MWU span above them.
    axs.plot([x1, x2], [y + h, y + h], lw=1.5, c=col)
    axs.text((x1 + x2) * .5, y + h, ks_cvm_text, ha='center', va='bottom', color=col,
             fontsize=font_size * 1)

    axs.plot([x2, x3], [y + h, y + h], lw=1.5, c=col)
    axs.text((x2 + x3) * .5, y + h, cvm_mannwhitneyu_text, ha='center', va='bottom', color=col,
             fontsize=font_size * 1)

    axs.plot([x1, x3], [y + 2 * h, y + 2 * h], lw=1.5, c=col)
    axs.text((x1 + x3) * .5, y + 2 * h, ks_mannwhitneyu_text, ha='center', va='bottom', color=col,
             fontsize=font_size * 1)

    axs.set_ylim([0, y + 3 * h])
    axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.tight_layout()
    plt.savefig(path_to_pr_aucs_with_significance_fig)
    plt.close(fig)


if __name__ == '__main__':
    main()