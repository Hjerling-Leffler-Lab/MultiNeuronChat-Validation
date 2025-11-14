import numpy as np
import pandas as pd
import xarray as xr

from sklearn.metrics import precision_recall_curve, auc

from multineuronchat import MultiNeuronChatObject

from utils import get_expected_perturbations_array

import pickle

import argparse


def compute_pr_curve_error_bands_2(
        curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    According to https://bbabenko.github.io/prs/
    :param curves:
    :return:
    """
    sampled_thresholds = np.linspace(0.0, 1.0, 10000)
    sampled_precisions = []
    sampled_recalls = []
    # assume curves is a list of (precision, recall, threshold)
    # tuples where each of those three is a numpy array
    for precision, recall, threshold in curves:
        sampled_precisions.append(
            np.interp(sampled_thresholds, threshold, precision[:-1]))
        sampled_recalls.append(
            np.interp(sampled_thresholds, threshold, recall[:-1]))

    # compute the mean and standard deviation of the interpolated precisions and recalls
    sampled_precisions = np.array(sampled_precisions)
    sampled_recalls = np.array(sampled_recalls)
    mean_precision = np.mean(sampled_precisions, axis=0)
    mean_recall = np.mean(sampled_recalls, axis=0)

    std_precision = np.std(sampled_precisions, axis=0)
    std_recall = np.std(sampled_recalls, axis=0)

    return mean_precision, mean_recall, std_precision, std_recall


def compute_pr_curve_error_bands(
        curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
        num_samples: int = 10000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean precision-recall curve and error bands by sampling thresholds based on the empirical distribution.
    :param curves: List of tuples containing precision, recall, and threshold arrays.
    :param num_samples: Number of thresholds to sample.
    :return: Tuple containing mean precision, mean recall, standard deviation of precision, and standard deviation of recall.
    """
    # Collect all thresholds from all curves
    all_thresholds = np.concatenate([threshold for _, _, threshold in curves])
    # Remove duplicates and sort thresholds
    all_thresholds = np.unique(all_thresholds)

    # Compute the empirical cumulative distribution function (CDF) of the thresholds
    sorted_thresholds = np.sort(all_thresholds)
    cdf = np.arange(1, len(sorted_thresholds) + 1) / len(sorted_thresholds)

    # Generate uniformly spaced probabilities
    uniform_probs = np.linspace(0, 1, num_samples)
    # Interpolate to find thresholds corresponding to these probabilities
    sampled_thresholds = np.interp(uniform_probs, cdf, sorted_thresholds)

    sampled_precisions = []
    sampled_recalls = []
    for precision, recall, threshold in curves:
        # Ensure threshold is sorted for interpolation
        sorted_indices = np.argsort(threshold)
        threshold_sorted = threshold[sorted_indices]
        precision_sorted = precision[sorted_indices]
        recall_sorted = recall[sorted_indices]

        # Interpolate precision and recall at the sampled thresholds
        interp_precision = np.interp(sampled_thresholds, threshold_sorted, precision_sorted)
        interp_recall = np.interp(sampled_thresholds, threshold_sorted, recall_sorted)
        sampled_precisions.append(interp_precision)
        sampled_recalls.append(interp_recall)

    # Convert lists to numpy arrays
    sampled_precisions = np.array(sampled_precisions)
    sampled_recalls = np.array(sampled_recalls)

    # Compute mean and standard deviation
    mean_precision = np.mean(sampled_precisions, axis=0)
    mean_recall = np.mean(sampled_recalls, axis=0)

    std_precision = np.std(sampled_precisions, axis=0)
    std_recall = np.std(sampled_recalls, axis=0)

    return mean_precision, mean_recall, std_precision, std_recall


def convert_p_values_to_input_for_pr_curve(
        p_values: xr.DataArray,
        cell_types: list[str],
        lt_names: list[str]
) -> np.ndarray:
    p_values_sorted = p_values.loc[cell_types, cell_types, lt_names]
    p_values_sorted = np.abs(1 - p_values_sorted)
    p_values_sorted = p_values_sorted.values.flatten()

    return p_values_sorted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--paths_to_mnc_files',
        nargs='+',
        required=True,
        type=str
    )
    parser.add_argument(
        '--path_to_expected_perturbations',
        required=True,
        type=str
    )

    parser.add_argument(
        '--path_to_curves_output',
        required=True,
        type=str
    )
    parser.add_argument(
        '--path_to_pr_aucs_output',
        required=True,
        type=str
    )
    parser.add_argument(
        '--path_to_error_bands_dict',
        required=True,
        type=str
    )

    parser.add_argument(
        '--statistical_tests',
        nargs='+',
        type=str,
        default=['KS', 'Anderson', 'CVM', 'MannWhitneyU']
    )
    args = parser.parse_args()

    paths_to_mnc_files: list[str] = args.paths_to_mnc_files
    path_to_expected_perturbations: str = args.path_to_expected_perturbations
    path_to_curves_output: str = args.path_to_curves_output
    path_to_pr_aucs_output: str = args.path_to_pr_aucs_output
    path_to_error_bands_dict: str = args.path_to_error_bands_dict
    statistical_tests: list[str] = args.statistical_tests

    mnc_reference_obj: MultiNeuronChatObject = MultiNeuronChatObject.load(paths_to_mnc_files[0])
    cell_types: list[str] = mnc_reference_obj.source_cell_types
    lt_names: list[str] = mnc_reference_obj.interaction_names

    expected_perturbations: pd.DataFrame = pd.read_csv(path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]
    reference_matrix = get_expected_perturbations_array(
        expected_perturbations=expected_perturbations,
        mnc_object=mnc_reference_obj
    )
    expected_perturbations_flat = reference_matrix.values.flatten()

    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        statistical_test: []
        for statistical_test in statistical_tests
    }
    pr_aucs: dict[str, list[float] | np.array] = {
        statistical_test: []
        for statistical_test in statistical_tests
    }

    for mnc_path in paths_to_mnc_files:
        mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(mnc_path)

        for statistical_test in statistical_tests:
            p_values = convert_p_values_to_input_for_pr_curve(
                p_values=mnc_object.p_values[statistical_test],
                cell_types=cell_types,
                lt_names=lt_names,
            )
            # If MultiNeuronChat has not tested a hypothesis, the p-value is set to NaN. To work with these here, we
            # set the inverted p-values (1 would correspond to p-value 0 and 0 would correspond to p-value 1) to 0, simulating
            # a negative prediction.
            p_values[np.isnan(p_values)] = 0

            curve = precision_recall_curve(
                expected_perturbations_flat,
                p_values
            )

            pr_auc = auc(curve[1], curve[0])

            curves[statistical_test].append(curve)
            pr_aucs[statistical_test].append(pr_auc)

    error_bands_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        label: compute_pr_curve_error_bands(curves[label])
        for label in curves.keys()
    }

    with open(path_to_curves_output, 'wb') as f:
        pickle.dump(curves, f)
    with open(path_to_pr_aucs_output, 'wb') as f:
        pickle.dump(pr_aucs, f)
    with open(path_to_error_bands_dict, 'wb') as f:
        pickle.dump(error_bands_dict, f)

if __name__ == '__main__':
    main()