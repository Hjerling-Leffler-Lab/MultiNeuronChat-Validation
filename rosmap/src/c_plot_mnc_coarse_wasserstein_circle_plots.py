import os

import pickle

import xarray as xr

from multineuronchat.visualize import plot_wasserstein_ranked_differential_communication_circle_plot

import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Plot p-value circle plot for MNC object.')

parser.add_argument(
    '--path_to_wasserstein_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_circle_plot_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_legend_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_arrow_legend_png',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_circle_plot_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_legend_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_arrow_legend_pdf',
    type=str,
    required=True,
)
parser.add_argument(
    '--width_in_cm_main',
    type=float,
    default=10.0,
)
parser.add_argument(
    '--height_in_cm_main',
    type=float,
    default=10.0,
)
parser.add_argument(
    '--width_in_cm_legend_cell_type',
    type=float,
    default=3.0,
)
parser.add_argument(
    '--height_in_cm_legend_cell_type',
    type=float,
    default=1.5,
)
parser.add_argument(
    '--width_in_cm_legend_arrow',
    type=float,
    default=3.0,
)
parser.add_argument(
    '--height_in_cm_legend_arrow',
    type=float,
    default=1.5,
)
args = parser.parse_args()

path_to_wasserstein_object: str = args.path_to_wasserstein_object

path_to_wasserstein_circle_plot_png: str = args.path_to_wasserstein_circle_plot_png
path_to_wasserstein_legend_png: str = args.path_to_wasserstein_legend_png
path_to_wasserstein_arrow_legend_png: str = args.path_to_wasserstein_arrow_legend_png
path_to_wasserstein_circle_plot_pdf: str = args.path_to_wasserstein_circle_plot_pdf
path_to_wasserstein_legend_pdf: str = args.path_to_wasserstein_legend_pdf
path_to_wasserstein_arrow_legend_pdf: str = args.path_to_wasserstein_arrow_legend_pdf

statistical_test: str = args.statistical_test

width_in_cm_main: float = args.width_in_cm_main
height_in_cm_main: float = args.height_in_cm_main
width_in_cm_legend_cell_type: float = args.width_in_cm_legend_cell_type
height_in_cm_legend_cell_type: float = args.height_in_cm_legend_cell_type
width_in_cm_legend_arrow: float = args.width_in_cm_legend_arrow
height_in_cm_legend_arrow: float = args.height_in_cm_legend_arrow

cm: float = 1/2.54  # cm to inches conversion factor

def main():
    cell_type_label_short: dict[str, str] = {
        'Astrocyte': 'Astro',
        'Excitatory Neurons': 'Exc',
        'Inhibitory Neurons': 'Inh',
        'Microglia': 'Micro',
        'OPCs': 'OPCs',
        'Oligodendrocytes': 'Oligo',
        'Vascular Niche': 'Vasc',
    }

    # For each output figure create the directory if it does not exist
    output_figures = [
        path_to_wasserstein_circle_plot_png,
        path_to_wasserstein_legend_png,
        path_to_wasserstein_arrow_legend_png,
        path_to_wasserstein_circle_plot_pdf,
        path_to_wasserstein_legend_pdf,
        path_to_wasserstein_arrow_legend_pdf
    ]
    for output_figure in output_figures:
        output_dir = os.path.dirname(output_figure)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    wasserstein_object: dict[str, xr.DataArray] = pickle.load(open(path_to_wasserstein_object, 'rb'))

    wasserstein_mask = wasserstein_object['wasserstein_mask']
    wasserstein_distances = wasserstein_object['wasserstein_distances']

    fig, axs = plt.subplots(1, 1, figsize=(width_in_cm_main * cm, height_in_cm_main * cm))
    legend_fig, legend_axs = plt.subplots(1, 1, figsize=(width_in_cm_legend_cell_type * cm, height_in_cm_legend_cell_type * cm))
    arrow_legend_fig, arrow_legend_axs = plt.subplots(1, 1, figsize=(width_in_cm_legend_arrow * cm, height_in_cm_legend_arrow * cm))

    plot_wasserstein_ranked_differential_communication_circle_plot(
        axs=axs,
        wasserstein_distances=wasserstein_distances,
        top_n_edges=18,
        cell_type_labels_short=cell_type_label_short,
        radius_of_nodes=0.05,
        show_labels=True,
        legend_axs=legend_axs,
        legend_marker_size=4,
        legend_font_size=8,
        legend_marker_edge_width=0.2,
        legend_n_cols=1,
        label_font_size=8,
        # title='Perturbed Cell-Cell Interactions \n(Wasserstein Distance, 99th percentile)',
        title_font_size=12,
        arrow_thickness_factor=0.5,
        arrow_head_length=2,
        arrow_head_width=1,
        arrow_legend_axs=arrow_legend_axs,
        arrow_legend_font_size=6,
        arrow_legend_n_cols=3,
        arrow_normalization_factor=None,
    )

    fig.tight_layout()
    legend_fig.tight_layout()
    arrow_legend_fig.tight_layout()

    fig.savefig(path_to_wasserstein_circle_plot_png, dpi=300)
    legend_fig.savefig(path_to_wasserstein_legend_png, dpi=300)
    arrow_legend_fig.savefig(path_to_wasserstein_arrow_legend_png, dpi=300)

    fig.savefig(path_to_wasserstein_circle_plot_pdf, dpi=300)
    legend_fig.savefig(path_to_wasserstein_legend_pdf, dpi=300)
    arrow_legend_fig.savefig(path_to_wasserstein_arrow_legend_pdf, dpi=300)

    plt.close(fig)
    plt.close(legend_fig)
    plt.close(arrow_legend_fig)



if __name__ == '__main__':
    main()
