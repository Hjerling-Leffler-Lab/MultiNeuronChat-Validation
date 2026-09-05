import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns

from utils import cm

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_all_timings',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_timing_fig',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_legend_fig',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--width_in_cm',
        type=float,
        default=17,
    )
    parser.add_argument(
        '--height_in_cm',
        type=float,
        default=5.3,
    )

    parser.add_argument(
        '--width_in_cm_legend',
        type=float,
        default=4,
    )
    parser.add_argument(
        '--height_in_cm_legend',
        type=float,
        default=3,
    )

    parser.add_argument(
        '--fontsize',
        type=float,
        default=8,
    )

    args = parser.parse_args()

    path_to_all_timings: str = args.path_to_all_timings
    path_to_timing_fig: str = args.path_to_timing_fig
    path_to_legend_fig: str = args.path_to_legend_fig

    width_in_cm: float = args.width_in_cm
    height_in_cm: float = args.height_in_cm

    width_in_cm_legend: float = args.width_in_cm_legend
    height_in_cm_legend: float = args.height_in_cm_legend

    fontsize: float = args.fontsize

    os.makedirs(os.path.dirname(path_to_timing_fig), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_legend_fig), exist_ok=True)

    means: list[str] = ['mean_0', 'tri_mean_0', 'trim_mean_0.1', 'trim_mean_0.05']
    cases: list[str] = ['CASE_0.3', 'CASE_0.5', 'CASE_1', 'CASE_1.5']
    proportions: list[str] = ['0.1', '0.3', '0.5', '0.8', '1.0']

    timing_results: dict[str, dict[str, dict[str, dict[str, list[float]]]]] = {
        mean: {
            case: {
                proportion: {

                }
                for proportion in proportions
            }
            for case in cases
        }
        for mean in means
    }
    # Convert to dataframe
    df_dict: dict[str, list[str | float]] = {
        'mean': [],
        'case': [],
        'proportion': [],
        'timing_type': [],
        'timing_result': [],
    }

    for mean in means:
        for case in cases:
            for proportion in proportions:
                path_to_timings: str = os.path.join(path_to_all_timings, mean, case, proportion)

                # Extract all timing files
                pkl_files: list[str] = [os.path.join(path_to_timings, x) for x in os.listdir(path_to_timings) if
                                        x.endswith('.pkl')]

                for pkl_file in pkl_files:
                    df: pd.DataFrame = pd.read_pickle(pkl_file)

                    total_time: float = 0.0
                    total_time_KS: float = 0.0
                    total_time_CvM: float = 0.0
                    total_time_MwU: float = 0.0

                    for key in df:
                        start, end = df[key]
                        diff = end - start

                        if key not in timing_results[mean][case][proportion].keys():
                            timing_results[mean][case][proportion][key] = []

                        timing_results[mean][case][proportion][key].append(diff)

                        df_dict['mean'].append(mean)
                        df_dict['case'].append(case)
                        df_dict['proportion'].append(proportion)
                        df_dict['timing_type'].append(key)
                        df_dict['timing_result'].append(diff)

                        total_time += diff

                        if 'significance' in key or 'correction' in key:
                            test = str(key).split('_')[-1]
                            if test == 'KS':
                                total_time_KS += diff
                            elif test == 'CVM':
                                total_time_CvM += diff
                            elif test == 'MannWhitneyU':
                                total_time_MwU += diff

                    total_timings: list[tuple[str, float]] = [
                        ('total_time', total_time),
                        ('total_time_KS', total_time_KS),
                        ('total_time_CVM', total_time_CvM),
                        ('total_time_MwU', total_time_MwU),
                    ]

                    for type, time in total_timings:
                        df_dict['mean'].append(mean)
                        df_dict['case'].append(case)
                        df_dict['proportion'].append(proportion)
                        df_dict['timing_type'].append(type)
                        df_dict['timing_result'].append(time)

    all_timings_df: pd.DataFrame = pd.DataFrame(df_dict)

    # Remove the total_time
    all_timings_df = all_timings_df[all_timings_df['timing_type'] != 'total_time']

    # Explicit left-to-right ordering of the timing events, grouped into: setup (communication scores),
    # the three label-blind prefilter masks, per-test significance, per-test FDR correction, and the
    # per-test total. Passing `order=` makes the x-axis self-consistent regardless of dict/file order,
    # and any timing key not listed here (e.g. a legacy Anderson column) is simply not drawn.
    category_order: list[str] = [
        'computation_of_communication_scores',
        'wasserstein_mask',
        'variance_mask',
        'abundance_mask',
        'significance_KS',
        'significance_CVM',
        'significance_MannWhitneyU',
        'correction_KS',
        'correction_CVM',
        'correction_MannWhitneyU',
        'total_time_KS',
        'total_time_CVM',
        'total_time_MwU',
    ]
    category_labels: list[str] = [
        'Com.\nScores',
        'Wass.\nMask',
        'Var.\nMask',
        'Abund.\nMask',
        'KS',
        'CVM',
        'MWU',
        'FDR\nKS',
        'FDR\nCVM',
        'FDR\nMWU',
        'Total\nKS',
        'Total\nCVM',
        'Total\nMWU',
    ]

    all_means_fig, all_means_axs = plt.subplots(1, 1, figsize=(width_in_cm * cm, height_in_cm * cm))

    color_palette = sns.color_palette('husl', len(means))

    # TODO:
    # 1. Change size of outlier + color them in the color of the mean type

    sns.boxplot(
        x='timing_type',
        y='timing_result',
        hue='mean',
        data=all_timings_df,
        order=category_order,
        ax=all_means_axs,
        linewidth=0.5,
        flierprops=dict(marker='o', markersize=3, markeredgewidth=0.5),
        palette=color_palette,
        saturation=1,
    )

    # Step 1: get box patches (in visual order)
    boxes = [patch for patch in all_means_axs.patches if isinstance(patch, mpatches.PathPatch)]

    # Step 2: get fliers
    fliers = [child for child in all_means_axs.get_children()
              if isinstance(child, mlines.Line2D) and child.get_marker() == 'o']

    # Step 3: pair them by position
    # Note: every box usually has 1 associated flier (sometimes none)
    # So we check len match or zip by min length
    for box, flier in zip(boxes, fliers):
        facecolor = box.get_facecolor()
        flier.set_markerfacecolor(facecolor)
        flier.set_markeredgecolor('black')  # optional for contrast

    all_means_axs.set_xlabel('Timing Event', fontsize=fontsize)
    all_means_axs.set_ylabel('Time in seconds', fontsize=fontsize)

    all_means_axs.set_xticks(list(range(len(category_order))))
    all_means_axs.set_xticklabels(category_labels, fontsize=fontsize * 0.8)

    max_y = np.max(all_timings_df['timing_result'])
    max_y_lim = max_y * 1.2

    # Group boundaries (in category-index space): Com.Scores | masks | significance | correction | total.
    # Dividers sit halfway between the last index of one group and the first of the next.
    divider_positions: list[float] = [0.5, 3.5, 6.5, 9.5]
    for divider in divider_positions:
        alpha = 0.8 if divider == 9.5 else 0.5
        all_means_axs.vlines(divider, 0, max_y_lim, color='black', linestyle='--', alpha=alpha)

    # Section headers centred over each group's index span.
    section_headers: list[tuple[float, float, str]] = [
        (1, 3, 'Prefilter Masks'),
        (4, 6, 'Computation of Significance'),
        (7, 9, 'Correction of P-Values'),
        (10, 12, 'Total Time per Test'),
    ]
    for start_idx, end_idx, header in section_headers:
        all_means_axs.text((start_idx + end_idx) / 2, max_y_lim * 0.95, header, ha='center', va='top',
                           fontsize=fontsize * 0.8)

    all_means_axs.set_xlim([-0.5, len(category_order) - 0.5])
    all_means_axs.set_ylim([-50, max_y_lim])
    all_means_axs.set_title('MultiNeuronChat Timings', fontsize=fontsize)

    # Disable legend
    all_means_axs.legend_.remove()

    all_means_fig.tight_layout()

    all_means_fig.savefig(path_to_timing_fig, dpi=300, transparent=True)

    plt.show()

    # Plot the legend separately
    legend_fig, legend_axs = plt.subplots(1, 1, figsize=(width_in_cm_legend * cm, height_in_cm_legend * cm))

    # Change title of legend
    handles, labels = all_means_axs.get_legend_handles_labels()
    new_labels = []
    for label in labels:
        if label == 'mean_0':
            new_labels.append('Mean')
        elif label == 'tri_mean_0':
            new_labels.append('Tri Mean')
        elif label == 'trim_mean_0.1':
            new_labels.append('Trim Mean 0.1')
        elif label == 'trim_mean_0.05':
            new_labels.append('Trim Mean 0.05')
    # Center the legend
    legend_axs.legend(handles, new_labels, fontsize=fontsize * 0.8, loc='center', frameon=True,
                      bbox_to_anchor=(0.5, 0.5))

    legend_axs.axis('off')

    # Turn off the grid
    legend_axs.grid(False)

    legend_axs.set_xlim([-0.5, 1.5])
    legend_axs.set_ylim([-0.5, 1.5])
    legend_fig.tight_layout()
    legend_fig.savefig(path_to_legend_fig, dpi=300, transparent=True)
    plt.show()

if __name__ == "__main__":
    main()