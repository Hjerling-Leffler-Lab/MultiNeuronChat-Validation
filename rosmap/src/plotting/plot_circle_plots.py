import os

from multineuronchat import MultiNeuronChatObject
from multineuronchat.visualize import plot_p_value_differential_communication_circle_plot, plot_wasserstein_ranked_differential_communication_circle_plot, plot_aula_medica_plot

import matplotlib.pyplot as plt

from utils import cm, cell_type_to_short

import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot P-value differential communication circle plot.')

    parser.add_argument(
        '--path_to_mnc_object',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--statistical_test',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--width_in_cm_circle_plot',
        type=float,
        required=False,
        default=7,
    )
    parser.add_argument(
        '--height_in_cm_circle_plot',
        type=float,
        required=False,
        default=7,
    )

    parser.add_argument(
        '--width_in_cm_cell_type_legend',
        type=float,
        required=False,
        default=7,
    )
    parser.add_argument(
        '--height_in_cm_cell_type_legend',
        type=float,
        required=False,
        default=7,
    )

    parser.add_argument(
        '--width_in_cm_arrow_legend',
        type=float,
        required=False,
        default=7,
    )
    parser.add_argument(
        '--height_in_cm_arrow_legend',
        type=float,
        required=False,
        default=7,
    )

    parser.add_argument(
        '--path_to_circle_plot',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_cell_type_legend',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_arrow_legend',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    path_to_mnc_object: str = args.path_to_mnc_object
    statistical_test: str = args.statistical_test

    width_in_cm_circle_plot: float = args.width_in_cm_circle_plot
    height_in_cm_circle_plot: float = args.height_in_cm_circle_plot

    width_in_cm_cell_type_legend: float = args.width_in_cm_cell_type_legend
    height_in_cm_cell_type_legend: float = args.height_in_cm_cell_type_legend

    width_in_cm_arrow_legend: float = args.width_in_cm_arrow_legend
    height_in_cm_arrow_legend: float = args.height_in_cm_arrow_legend

    path_to_circle_plot: str = args.path_to_circle_plot
    path_to_cell_type_legend: str = args.path_to_cell_type_legend
    path_to_arrow_legend: str = args.path_to_arrow_legend

    # Create missing output paths
    all_output_paths: list[str] = [os.path.dirname(path) for path in [path_to_circle_plot, path_to_cell_type_legend, path_to_arrow_legend]]
    for output_path in all_output_paths:
        os.makedirs(output_path, exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)

    fig, axs = plt.subplots(1, 1, figsize=(width_in_cm_circle_plot * cm, height_in_cm_circle_plot * cm))
    legend_fig, legend_axs = plt.subplots(1, 1, figsize=(width_in_cm_cell_type_legend * cm, height_in_cm_cell_type_legend * cm))
    arrow_legend_fig, arrow_legend_axs = plt.subplots(1, 1, figsize=(width_in_cm_arrow_legend * cm, height_in_cm_arrow_legend * cm))

    plot_p_value_differential_communication_circle_plot(
        axs=axs,
        mnc_object=mnc_object,
        significance_test_to_use=statistical_test,
        cell_type_labels_short=cell_type_to_short,
        significance_threshold=0.05,
        use_adj_p_values=True,
        radius_of_nodes=0.05,
        show_labels=True,
        legend_axs=legend_axs,
        legend_marker_size=4,
        legend_font_size=6,
        legend_marker_edge_width=0.2,
        legend_n_cols=1,
        label_font_size=7,
        title_font_size=12,
        arrow_thickness_factor=0.5,
        arrow_head_length=2,
        arrow_head_width=1,
        arrow_legend_axs=arrow_legend_axs,
        arrow_legend_font_size=6,
        arrow_legend_n_cols=1,
        arrow_normalization_factor=None,
    )

    fig.tight_layout()
    fig.savefig(path_to_circle_plot)

    legend_fig.tight_layout()
    legend_fig.savefig(path_to_cell_type_legend)

    arrow_legend_fig.tight_layout()
    arrow_legend_fig.savefig(path_to_arrow_legend)

    plt.close(fig)
    plt.close(legend_fig)
    plt.close(arrow_legend_fig)

if __name__ == '__main__':
    main()