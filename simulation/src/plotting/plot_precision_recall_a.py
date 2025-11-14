import copy

import pickle

import numpy as np
import xarray as xr
import pandas as pd

from multineuronchat import MultiNeuronChatObject

from utils import get_expected_perturbations_array

import argparse


def compute_confusion_matrix(
        reference_matrix: xr.DataArray,
        test_matrix: xr.DataArray
) -> xr.DataArray:
    confusion_matrix: xr.DataArray = xr.DataArray(
        np.zeros((2, 2)),
        dims=('actual', 'predicted'),
        coords={
            'actual': ['positive', 'negative'],
            'predicted': ['positive', 'negative']
        }
    )

    confusion_matrix.loc[{'actual': 'positive', 'predicted': 'positive'}] = np.sum(
        np.logical_and(reference_matrix == 1, test_matrix == 1)
    )
    confusion_matrix.loc[{'actual': 'positive', 'predicted': 'negative'}] = np.sum(
        np.logical_and(reference_matrix == 1, test_matrix == 0)
    )
    confusion_matrix.loc[{'actual': 'negative', 'predicted': 'positive'}] = np.sum(
        np.logical_and(reference_matrix == 0, test_matrix == 1)
    )
    confusion_matrix.loc[{'actual': 'negative', 'predicted': 'negative'}] = np.sum(
        np.logical_and(reference_matrix == 0, test_matrix == 0)
    )

    return confusion_matrix


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paths_to_mnc_object',
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--paths_to_wasserstein_object',
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--path_to_expected_perturbations',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_summary_df',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--statistical_test',
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

    args = parser.parse_args()

    paths_to_mnc_object: str = args.paths_to_mnc_object
    paths_to_wasserstein_object: str = args.paths_to_wasserstein_object
    path_to_expected_perturbations: str = args.path_to_expected_perturbations
    path_to_summary_df: str = args.path_to_summary_df

    statistical_test: str = args.statistical_test

    mean_type = args.mean_type
    case = args.case
    proportion = args.proportion

    expected_perturbations: pd.DataFrame = pd.read_csv(path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]
    reference_mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(paths_to_mnc_object[0])
    reference_matrix: xr.DataArray = get_expected_perturbations_array(
        expected_perturbations=expected_perturbations,
        mnc_object=reference_mnc_object
    )
    del reference_mnc_object


    precision_recall_dict: dict[str, list[str | float]] = {
        'Mean Type': [],
        'Case': [],
        'Proportion': [],
        'Statistical Test': [],
        'Precision': [],
        'Recall': []
    }

    for mnc_path, wasserstein_path in zip(paths_to_mnc_object, paths_to_wasserstein_object):
        mnc_object = MultiNeuronChatObject.load(mnc_path)
        wasserstein_object: dict[str, xr.DataArray] = pickle.load(
            open(
                wasserstein_path,
                'rb'
            )
        )

        wasserstein_distances = wasserstein_object['wasserstein_distances']

        wasserstein_mask = wasserstein_distances > np.nanpercentile(wasserstein_distances, 90)

        mnc_object_corrected = copy.deepcopy(mnc_object)

        # Compute the significance again for the wasserstein mask set at the top 10 percent of cells
        mnc_object_corrected.compute_significance(
            statistical_test=statistical_test,
            mask=wasserstein_mask,
        )

        mnc_object_corrected.correct_p_values(
            statistical_test=statistical_test,
            method='by'
        )

        confusion_matrix_test: xr.DataArray = compute_confusion_matrix(
            reference_matrix=reference_matrix,
            test_matrix=mnc_object.p_values_adj[statistical_test] < 0.05
        )

        confusion_matrix_wasserstein_plus_test: xr.DataArray = compute_confusion_matrix(
            reference_matrix=reference_matrix,
            test_matrix=mnc_object_corrected.p_values_adj[statistical_test] < 0.05
        )

        precision = confusion_matrix_test.loc[{'actual': 'positive', 'predicted': 'positive'}] / np.sum(confusion_matrix_test.loc[{'predicted': 'positive'}])
        recall = confusion_matrix_test.loc[{'actual': 'positive', 'predicted': 'positive'}] / np.sum(confusion_matrix_test.loc[{'actual': 'positive'}])

        precision_with_wasserstein = confusion_matrix_wasserstein_plus_test.loc[{'actual': 'positive','predicted': 'positive'}] / np.sum(confusion_matrix_wasserstein_plus_test.loc[{'predicted': 'positive'}])
        recall_with_wasserstein = confusion_matrix_wasserstein_plus_test.loc[{'actual': 'positive','predicted': 'positive'}] / np.sum(confusion_matrix_wasserstein_plus_test.loc[{'actual': 'positive'}])

        # Store results in dictionary
        precision_recall_dict['Mean Type'].append(mean_type)
        precision_recall_dict['Case'].append(case)
        precision_recall_dict['Proportion'].append(proportion)
        precision_recall_dict['Statistical Test'].append(f'{statistical_test}_adj')
        precision_recall_dict['Precision'].append(float(precision.values))
        precision_recall_dict['Recall'].append(float(recall.values))

        precision_recall_dict['Mean Type'].append(mean_type)
        precision_recall_dict['Case'].append(case)
        precision_recall_dict['Proportion'].append(proportion)
        precision_recall_dict['Statistical Test'].append(f'Wasserstein_{statistical_test}_adj')
        precision_recall_dict['Precision'].append(float(precision_with_wasserstein.values))
        precision_recall_dict['Recall'].append(float(recall_with_wasserstein.values))

    precision_recall_df: pd.DataFrame = pd.DataFrame(precision_recall_dict)
    precision_recall_df.to_pickle(path_to_summary_df)

if __name__ == '__main__':
    main()