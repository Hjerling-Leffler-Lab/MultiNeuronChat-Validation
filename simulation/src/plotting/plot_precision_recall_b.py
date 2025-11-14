import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from utils import cm, statistical_test_colors, wasserstein_statistical_test_colors

import argparse


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
        '--statistical_tests',
        nargs='+',
        type=str,
        help='Statistical tests to plot',
        required=True
    )
    parser.add_argument(
        '--width_of_figure_in_cm',
        type=float,
        help='Width of figure in cm',
        default=8.5
    )
    parser.add_argument(
        '--height_of_figure_in_cm',
        type=float,
        help='Height of figure in cm',
        default=8.2 / 2
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
        '--path_to_output_figure',
        type=str,
        help='Path to output figure',
        required=True
    )
    parser.add_argument(
        '--show_figure',
        action='store_true',
        help='Show figure',
        default=False
    )
    args = parser.parse_args()

    paths_to_summary_dfs: list[str] = args.paths_to_summary_dfs
    path_to_output_figure: str = args.path_to_output_figure

    mean_type_to_plot: str = args.mean_type_to_plot
    case_to_plot: str = args.case_to_plot
    proportions_to_plot: list[float] = sorted(args.proportions_to_plot)
    samples_in_dataset: int = args.samples_in_dataset
    statistical_tests: list[str] = args.statistical_tests

    width_of_figure_in_cm: float = args.width_of_figure_in_cm
    height_of_figure_in_cm: float = args.height_of_figure_in_cm

    dpi: float = args.dpi

    font_size: float = args.font_size
    label_font_size: float = args.label_font_size

    show_figure: bool = args.show_figure

    n_proportions: int = len(proportions_to_plot)
    n_tests: int = len(statistical_tests)

    # Join the dfs together
    precision_recall_df = load_joined_df(paths_to_summary_dfs)

    precision_recall_fig, (precision_ax, recall_ax) = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(width_of_figure_in_cm * cm, height_of_figure_in_cm * cm)
    )

    width_to_plot = 0.9
    x_positions: np.ndarray = np.arange(n_proportions)
    x_offset_per_test: list[float] = [
        -(width_to_plot / 2) + i * (width_to_plot / (n_tests * 2)) for i in range(n_tests * 2)
    ]

    mean_precision_and_recall_dict: dict[str, dict[str, list[float]]] = {}
    for statistical_test in statistical_tests:
        mean_precision_and_recall_dict[f'{statistical_test}_adj'] = {
            'Precision': [],
            'Recall': [],
        }
        mean_precision_and_recall_dict[f'Wasserstein_{statistical_test}_adj'] = {
            'Precision': [],
            'Recall': [],
        }

    for i, proportion in enumerate(proportions_to_plot):
        x_position = x_positions[i]

        precision_recall_df_subset: pd.DataFrame = precision_recall_df[
            (precision_recall_df['Mean Type'] == mean_type_to_plot) &
            (precision_recall_df['Case'] == case_to_plot) &
            (precision_recall_df['Proportion'] == proportion)
        ]

        for j, statistical_test in enumerate(statistical_tests):
            x_offset: float = x_offset_per_test[(j * 2)]
            x_offset_2: float = x_offset_per_test[(j * 2) + 1]

            precision_values: list[float] = precision_recall_df_subset[
                precision_recall_df_subset['Statistical Test'] == f'{statistical_test}_adj'
                ]['Precision'].values.tolist()
            recall_values: list[float] = precision_recall_df_subset[
                precision_recall_df_subset['Statistical Test'] == f'{statistical_test}_adj'
                ]['Recall'].values.tolist()

            precision_with_wasserstein_values: list[float] = precision_recall_df_subset[
                precision_recall_df_subset['Statistical Test'] == f'Wasserstein_{statistical_test}_adj'
                ]['Precision'].values.tolist()
            recall_with_wasserstein_values: list[float] = precision_recall_df_subset[
                precision_recall_df_subset['Statistical Test'] == f'Wasserstein_{statistical_test}_adj'
                ]['Recall'].values.tolist()

            mean_precision: float = np.mean(precision_values).item()
            std_precision: float = np.std(precision_values).item()

            mean_recall: float = np.mean(recall_values).item()
            std_recall: float = np.std(recall_values).item()

            mean_precision_with_wasserstein: float = np.mean(precision_with_wasserstein_values).item()
            std_precision_with_wasserstein: float = np.std(precision_with_wasserstein_values).item()

            mean_recall_with_wasserstein: float = np.mean(recall_with_wasserstein_values).item()
            std_recall_with_wasserstein: float = np.std(recall_with_wasserstein_values).item()

            mean_precision_and_recall_dict[f'{statistical_test}_adj']['Precision'].append(mean_precision)
            mean_precision_and_recall_dict[f'{statistical_test}_adj']['Recall'].append(mean_recall)

            mean_precision_and_recall_dict[f'Wasserstein_{statistical_test}_adj']['Precision'].append(
                mean_precision_with_wasserstein
            )
            mean_precision_and_recall_dict[f'Wasserstein_{statistical_test}_adj']['Recall'].append(
                mean_recall_with_wasserstein
            )

            precision_ax.errorbar(
                x=[x_position + x_offset],
                y=[mean_precision],
                yerr=[std_precision],
                label=statistical_test,
                color=statistical_test_colors[statistical_test],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
            )

            precision_ax.errorbar(
                x=[x_position + x_offset_2],
                y=[mean_precision_with_wasserstein],
                yerr=[std_precision_with_wasserstein],
                label=f'Wasserstein_{statistical_test}',
                color=wasserstein_statistical_test_colors[statistical_test],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
            )

            ### Plot Recall ###
            recall_ax.errorbar(
                x=[x_position + x_offset],
                y=[mean_recall],
                yerr=[std_recall],
                label=statistical_test,
                color=statistical_test_colors[statistical_test],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
            )

            recall_ax.errorbar(
                x=[x_position + x_offset_2],
                y=[mean_recall_with_wasserstein],
                yerr=[std_recall_with_wasserstein],
                label=f'Wasserstein_{statistical_test}',
                color=wasserstein_statistical_test_colors[statistical_test],
                fmt='-o',
                capsize=1.5,
                capthick=1,
                linewidth=1,
                markersize=3,
            )

    for j, statistical_test in enumerate(statistical_tests):
        x_offset: float = x_offset_per_test[(j * 2)]
        x_offset_2: float = x_offset_per_test[(j * 2) + 1]

        precision_ax.plot(
            [x_position + x_offset for x_position in x_positions],
            mean_precision_and_recall_dict[f'{statistical_test}_adj']['Precision'],
            color=statistical_test_colors[statistical_test],
            linewidth=0.5,
            linestyle='--'
        )

        precision_ax.plot(
            [x_position + x_offset_2 for x_position in x_positions],
            mean_precision_and_recall_dict[f'Wasserstein_{statistical_test}_adj']['Precision'],
            color=wasserstein_statistical_test_colors[statistical_test],
            linewidth=0.5,
            linestyle='-.'
        )

        recall_ax.plot(
            [x_position + x_offset for x_position in x_positions],
            mean_precision_and_recall_dict[f'{statistical_test}_adj']['Recall'],
            color=statistical_test_colors[statistical_test],
            linewidth=0.5,
            linestyle='--'
        )

        recall_ax.plot(
            [x_position + x_offset_2 for x_position in x_positions],
            mean_precision_and_recall_dict[f'Wasserstein_{statistical_test}_adj']['Recall'],
            color=wasserstein_statistical_test_colors[statistical_test],
            linewidth=0.5,
            linestyle='-.'
        )

    precision_ax.set_xticks(x_positions)
    precision_ax.set_xticklabels(
        [int(samples_in_dataset*proportion) for proportion in proportions_to_plot]
    )

    precision_ax.set_yticks(np.arange(0, 1.1, 0.2))
    precision_ax.yaxis.grid(False)

    precision_ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    precision_ax.set_ylim(0, 1.05)

    precision_ax.set_ylabel('Precision', fontsize=font_size)

    recall_ax.set_xticks(x_positions)
    recall_ax.set_xticklabels(
        [int(samples_in_dataset*proportion) for proportion in proportions_to_plot]
    )

    recall_ax.set_yticks(np.arange(0, 1.1, 0.2))
    recall_ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    recall_ax.yaxis.grid(False)
    recall_ax.set_ylim(0, 1.05)
    recall_ax.set_ylabel('Recall', fontsize=font_size)

    precision_ax.tick_params(axis='x', labelsize=label_font_size)
    precision_ax.tick_params(axis='y', labelsize=label_font_size)
    recall_ax.tick_params(axis='x', labelsize=label_font_size)
    recall_ax.tick_params(axis='y', labelsize=label_font_size)

    for x_position in x_positions[:-1]:
        precision_ax.axvline(x_position + 0.5, color='black', linewidth=0.3, linestyle='--', alpha=0.3)
        recall_ax.axvline(x_position + 0.5, color='black', linewidth=0.3, linestyle='--', alpha=0.3)

    handles = []
    for j, statistical_test in enumerate(statistical_tests):
        handles.append(plt.Line2D([0], [0], color=statistical_test_colors[statistical_test], lw=1, label=statistical_test))
        handles.append(plt.Line2D([0], [0], color=wasserstein_statistical_test_colors[statistical_test], lw=1, label=f'WS+{statistical_test}'))

    precision_ax.legend(
        handles=handles,
        fontsize=font_size,
    )

    precision_recall_fig.tight_layout()

    precision_recall_fig.savefig(
        path_to_output_figure,
        dpi=dpi
    )

    if show_figure:
        plt.show()
    else:
        plt.close(precision_recall_fig)


if __name__ == '__main__':
    main()