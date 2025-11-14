import os

import numpy as np
import xarray as xr

import pickle

from multineuronchat import MultiNeuronChatObject

import matplotlib.pyplot as plt

from utils import cm, extract_expected_perturbations_df

import argparse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_expected_perturbations',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--mnc_file_paths',
        type=str,
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--wasserstein_paths',
        type=str,
        nargs='+',
        required=True,
    )

    parser.add_argument(
        '--path_to_ks_figure',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_anderson_figures',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_cvm_figure',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_mwu_figure',
        type=str,
        required=True,
    )

    parser.add_argument(
        '--mean_type',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--case',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--proportion',
        type=float,
        required=True,
    )

    parser.add_argument(
        '--width_in_cm',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--height_in_cm',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--font_size',
        type=float,
        required=True,
    )
    args = parser.parse_args()

    path_to_expected_perturbations: str = args.path_to_expected_perturbations

    mnc_file_paths: list[str] = args.mnc_file_paths
    wasserstein_paths: list[str] = args.wasserstein_paths

    # Get a random mnc object just to define the dimensions
    path_to_arbitrary_mnc_object: str = mnc_file_paths[0]

    mean_type: str = args.mean_type
    case: str = args.case
    proportion: float = args.proportion

    path_to_ks_figure: str = args.path_to_ks_figure
    path_to_anderson_figures: str = args.path_to_anderson_figures
    path_to_cvm_figure: str = args.path_to_cvm_figure
    path_to_mwu_figure: str = args.path_to_mwu_figure

    significance_test_to_path: dict[str, str] = {
        'KS': path_to_ks_figure,
        'Anderson': path_to_anderson_figures,
        'CVM': path_to_cvm_figure,
        'MannWhitneyU': path_to_mwu_figure,
    }

    # Create paths if they don't exist
    for path in [path_to_ks_figure, path_to_anderson_figures, path_to_cvm_figure, path_to_mwu_figure]:
        os.makedirs(os.path.dirname(path), exist_ok=True)

    width_in_cm: float = args.width_in_cm
    height_in_cm: float = args.height_in_cm
    font_size: float = args.font_size

    plot_neg_log_p_values: bool = True
    plot_log_wasserstein: bool = False

    colors = ['#ca0020', '#f4a582', '#92c5de', '#0571b0']

    fig_size: tuple[float, float] = (width_in_cm * cm, height_in_cm * cm)

    # Load MultiNeuronChat object to get the cell types and ligand-target interactions
    # This is an arbitrary MultiNeuronChat object, as we only need the cell types and ligand-target interactions
    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(
        path_to_file=path_to_arbitrary_mnc_object
    )

    cell_types = mnc_object.p_values['CVM'].coords['source'].to_numpy().tolist()
    lt_interactions = mnc_object.p_values['CVM'].coords['interaction'].to_numpy().tolist()

    expected_perturbations = extract_expected_perturbations_df(
        path_to_expected_perturbations=path_to_expected_perturbations,
        cell_types=cell_types,
        lt_interactions=lt_interactions,
    )

    true_perturbations_indices = expected_perturbations > 0
    true_not_perturbations_indices = expected_perturbations == 0

    for i, (mnc_file, wasserstein_file) in enumerate(zip(mnc_file_paths, wasserstein_paths)):
        print(f'Processing {mnc_file} and {wasserstein_file}')

        mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(mnc_file)
        wasserstein_distances_dict: dict[str, xr.DataArray] = pickle.load(open(wasserstein_file, 'rb'))

        wasserstein_distances: xr.DataArray = wasserstein_distances_dict['wasserstein_distances']

        for significance_test in mnc_object.p_values.keys():
            p_values: xr.DataArray = mnc_object.p_values[significance_test]

            # match the p-values to the wasserstein distances
            wasserstein_values: np.ndarray = wasserstein_distances.values
            p_values_values: np.ndarray = p_values.values

            # Get all indices that are not NaN in both arrays
            valid_indices: np.ndarray = np.logical_and(~np.isnan(wasserstein_values),
                                                       ~np.isnan(p_values_values))

            true_perturbations_indices = valid_indices & true_perturbations_indices
            true_not_perturbations_indices = valid_indices & true_not_perturbations_indices

            true_perturbed_wasserstein_values = wasserstein_values[true_perturbations_indices].flatten()
            true_not_perturbed_wasserstein_values = wasserstein_values[
                true_not_perturbations_indices].flatten()

            true_perturbed_p_values_values = p_values_values[true_perturbations_indices].flatten()
            true_not_perturbed_p_values_values = p_values_values[true_not_perturbations_indices].flatten()

            if plot_neg_log_p_values:
                true_perturbed_p_values_values = -np.log10(true_perturbed_p_values_values)
                true_not_perturbed_p_values_values = -np.log10(true_not_perturbed_p_values_values)

            if plot_log_wasserstein:
                true_perturbed_wasserstein_values = np.log10(true_perturbed_wasserstein_values)
                true_not_perturbed_wasserstein_values = np.log10(true_not_perturbed_wasserstein_values)

            fig = plt.figure(figsize=fig_size)

            gs = fig.add_gridspec(2, 2, width_ratios=(4, 1),
                                  height_ratios=(1, 4),
                                  left=0.15, right=0.97, bottom=0.1, top=0.9,
                                  wspace=0.00, hspace=0.00)

            axs = fig.add_subplot(gs[1, 0])
            axs_hist_x = fig.add_subplot(gs[0, 0], sharex=axs)
            axs_hist_y = fig.add_subplot(gs[1, 1], sharey=axs)

            # fig, axs = plt.subplots(1, 1, figsize=fig_size)

            axs.scatter(
                true_not_perturbed_wasserstein_values,
                true_not_perturbed_p_values_values,
                alpha=0.5,
                s=10,
                label='Not Perturbed Pathways',
                color=colors[1]
            )
            axs.scatter(
                true_perturbed_wasserstein_values,
                true_perturbed_p_values_values,
                alpha=0.5,
                s=10,
                label='Perturbed Pathways',
                color=colors[0]
            )

            axs_hist_x.hist(
                true_not_perturbed_wasserstein_values,
                bins=20,
                orientation='vertical',
                color=colors[1],
                alpha=0.5,
                density=True
            )
            axs_hist_x.hist(
                true_perturbed_wasserstein_values,
                bins=20,
                orientation='vertical',
                color=colors[0],
                alpha=0.5,
                density=True
            )

            axs_hist_y.hist(
                true_not_perturbed_p_values_values,
                bins=20,
                orientation='horizontal',
                color=colors[1],
                alpha=0.5,
                density=True
            )
            axs_hist_y.hist(
                true_perturbed_p_values_values,
                bins=20,
                orientation='horizontal',
                color=colors[0],
                alpha=0.5,
                density=True
            )

            axs_hist_x.set_axis_off()
            axs_hist_y.set_axis_off()

            if plot_neg_log_p_values:
                axs.set_ylabel('-log10(p-value)', fontsize=font_size)
            else:
                axs.set_ylabel('p-value', fontsize=font_size)

            if plot_log_wasserstein:
                axs.set_xlabel('log10(Wasserstein Distance)', fontsize=font_size)
            else:
                axs.set_xlabel('Wasserstein Distance', fontsize=font_size)

            axs.tick_params(axis='both', which='major', labelsize=font_size)

            # Compute correlation and add text to top left corner
            wasserstein_values_for_correlation = wasserstein_values[valid_indices].flatten()
            p_values_values_for_correlation = p_values_values[valid_indices].flatten()

            if plot_neg_log_p_values:
                p_values_values_for_correlation = -np.log10(p_values_values_for_correlation)

            if plot_log_wasserstein:
                wasserstein_values_for_correlation = np.log10(wasserstein_values_for_correlation)

            correlation = np.corrcoef(wasserstein_values_for_correlation, p_values_values_for_correlation)[
                0, 1]
            axs.text(0.05, 0.9, f'R={correlation:.2f}', transform=axs.transAxes, fontsize=font_size)

            fig.savefig(significance_test_to_path[significance_test])
            plt.close()


if __name__ == '__main__':
    main()
