import os

import pickle

import numpy as np
import xarray as xr

from multineuronchat import MultiNeuronChatObject
from multineuronchat.visualize import plot_aula_medica_plot

import matplotlib.pyplot as plt

from utils import cm, cell_type_to_short

import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot aula medica')

    parser.add_argument(
        '--path_to_mnc_object',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_wasserstein_object',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--statistical_test',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_aula_medica_plot',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_aula_medica_wasserstein_label',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_aula_medica_p_value_label',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    path_to_mnc_object: str = args.path_to_mnc_object
    path_to_wasserstein_object: str = args.path_to_wasserstein_object

    statistical_test: str = args.statistical_test

    path_to_aula_medica_plot: str = args.path_to_aula_medica_plot
    path_to_aula_medica_wasserstein_label: str = args.path_to_aula_medica_wasserstein_label
    path_to_aula_medica_p_value_label: str = args.path_to_aula_medica_p_value_label

    # Create paths if it doesn't exist
    all_paths: list[str] = [path_to_aula_medica_plot, path_to_aula_medica_wasserstein_label, path_to_aula_medica_p_value_label]
    for path in all_paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)

    wasserstein_dict: dict[str, dict[str, float]] = pickle.load(open(path_to_wasserstein_object, 'rb'))
    # Get the distances for the wasserstein distance
    wasserstein_distances: xr.DataArray = wasserstein_dict['wasserstein_distances']
    wasserstein_mask = wasserstein_dict['wasserstein_mask']

    scaling_for_clean_fill: float = 1

    valid_interactions = np.logical_not(wasserstein_mask.sum(dim='source').sum(dim='receiver') == 0)
    valid_interactions = valid_interactions.coords['interaction'][valid_interactions].values.tolist()

    top_cell_types = [
        ('Astrocyte', mnc_object.source_cell_types),
        ('Excitatory Neurons', mnc_object.source_cell_types),
        ('Inhibitory Neurons', mnc_object.source_cell_types),
    ]
    bottom_cell_types = [
        ('Microglia', mnc_object.source_cell_types),
        ('OPCs', mnc_object.source_cell_types),
        ('Oligodendrocytes', mnc_object.source_cell_types),
    ]

    fig, axs = plt.subplots(2, 1, figsize=(scaling_for_clean_fill * 18 * cm, scaling_for_clean_fill * 12.3 * cm))
    wasserstein_legend_fig, wasserstein_legend_axs = plt.subplots(1, 1, figsize=(3 * cm, 5 * cm), layout='constrained')
    p_value_legend_fig, p_value_legend_axs = plt.subplots(1, 1, figsize=(3 * cm, 5 * cm), layout='constrained')

    wasserstein_legend_axs.set_visible(False)
    small_wasserstein_axis = wasserstein_legend_fig.add_axes([0.3, 0.1, 0.1, 0.8])

    p_value_legend_axs.set_visible(False)
    small_p_value_axis = p_value_legend_fig.add_axes([0.3, 0.1, 0.1, 0.8])

    plot_aula_medica_plot(
        axs=axs[0],
        mnc_object=mnc_object,
        wasserstein_distance_matrix=wasserstein_distances,
        statistical_test=statistical_test,
        use_adjusted_p_values=True,
        cell_types=top_cell_types,
        cell_type_labels_short=cell_type_to_short,
        ligand_target_interactions=valid_interactions,
        # triangle_line_width: float = 0.5,
        # source_cell_type_split_line_width: float = 1.0,
        scale=1,
        tick_font_size=7 * scaling_for_clean_fill,
        marker_size=10 * scaling_for_clean_fill,
        wasserstein_color_map='green',
        p_value_color_map='red',
        # not_tested_color: str = '#CCCCCC',
        # wasserstein_legend_axs: plt.Axes | None = None,
        # p_value_legend_axs: plt.Axes | None = None,
        # wasserstein_rounding: int | None = None,
        # p_value_rounding: int | None = None,
        annotate_source_cell_types = True,
        annotate_receiver_cell_types=False,
        annotate_interactions = True,
        annotate_significance = True,
        significance_threshold = 0.05,
        # max_log_p_value: float | None = None,
        # max_wasserstein_distance: float | None = None,
    )

    plot_aula_medica_plot(
        axs=axs[1],
        mnc_object=mnc_object,
        wasserstein_distance_matrix=wasserstein_distances,
        statistical_test=statistical_test,
        use_adjusted_p_values=True,
        cell_types=bottom_cell_types,
        cell_type_labels_short=cell_type_to_short,
        ligand_target_interactions=valid_interactions,
        # triangle_line_width: float = 0.5,
        # source_cell_type_split_line_width: float = 1.0,
        scale=1,
        tick_font_size=7 * scaling_for_clean_fill,
        marker_size=10 * scaling_for_clean_fill,
        wasserstein_color_map='green',
        p_value_color_map='red',
        # not_tested_color: str = '#CCCCCC',
        wasserstein_legend_axs=small_wasserstein_axis,
        p_value_legend_axs=small_p_value_axis,
        wasserstein_rounding=2,
        p_value_rounding=2,
        annotate_source_cell_types = True,
        annotate_receiver_cell_types = True,
        annotate_interactions = True,
        annotate_significance = True,
        significance_threshold = 0.05,
        # max_log_p_value: float | None = None,
        # max_wasserstein_distance: float | None = None,
    )

    fig.tight_layout()
    fig.savefig(
        path_to_aula_medica_plot
    )
    wasserstein_legend_fig.savefig(
        path_to_aula_medica_wasserstein_label
    )
    p_value_legend_fig.savefig(
        path_to_aula_medica_p_value_label
    )

    plt.show()


if __name__ == '__main__':
    main()