import os

import time

import pickle

import xarray as xr

from multineuronchat import MultiNeuronChatObject
from multineuronchat.masks import compute_wasserstein_mask

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_subject_wise_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_cell_type_no_mask',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_cell_type',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_grouping_by_no_mask',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_grouping_by',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_cell_type',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_grouping_by',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_timings',
    type=str,
    required=True,
)
parser.add_argument(
    '--chunk_size_to_use',
    type=int,
    default=4096,
)
parser.add_argument(
    '--n_processes_to_use',
    type=int,
    default=1
)
parser.add_argument(
    '--min_n_cells_threshold_to_use',
    type=int,
    default=10,
)
parser.add_argument(
    '--significance_tests',
    type=str,
    nargs='+',
    default=['KS', 'Anderson', 'CVM', 'MannWhitneyU'],
    help='List of significance tests to use. Options: KS, Anderson, CVM, MannWhitneyU'
)
args = parser.parse_args()

# Paths to the input files and output folder
path_to_subject_wise_normalized_loom: str = args.path_to_subject_wise_normalized_loom

path_to_mnc_cell_type_no_mask: str = args.path_to_mnc_cell_type_no_mask
path_to_mnc_cell_type: str = args.path_to_mnc_cell_type
path_to_mnc_grouping_by_no_mask: str = args.path_to_mnc_grouping_by_no_mask
path_to_mnc_grouping_by: str = args.path_to_mnc_grouping_by
path_to_wasserstein_cell_type: str = args.path_to_wasserstein_cell_type
path_to_wasserstein_grouping_by: str = args.path_to_wasserstein_grouping_by
path_to_timings: str = args.path_to_timings

chunk_size_to_use: int = args.chunk_size_to_use
n_processes_to_use: int = args.n_processes_to_use
min_n_cells_threshold_to_use: int = args.min_n_cells_threshold_to_use
significance_tests: list[str] = args.significance_tests



def main():
    if not os.path.exists(path_to_subject_wise_normalized_loom):
        raise FileNotFoundError(f'File not found: {path_to_subject_wise_normalized_loom}')

    # Create output folder if it does not exist
    all_output_paths = [
        path_to_mnc_cell_type_no_mask,
        path_to_mnc_cell_type,
        path_to_mnc_grouping_by_no_mask,
        path_to_mnc_grouping_by,
        path_to_wasserstein_cell_type,
        path_to_wasserstein_grouping_by,
        path_to_timings
    ]
    for output_path in all_output_paths:
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    mnc_object_cell_type: MultiNeuronChatObject = MultiNeuronChatObject(
        condition_label_column='cogdx',
        condition_names=('NCI', 'AD'),
        subject_label_column='individualID',
        cell_type_label_column='cell.type',
        db='human_extended',
    )

    ### START TRACKING TIME FOR cell_type NO MASK ###
    start_time_total_cell_type_no_mask = time.time()
    mnc_object_cell_type.compute_communication_scores(
        path_to_data_loom=path_to_subject_wise_normalized_loom,
        path_to_subject_wise_max_normalized=path_to_subject_wise_normalized_loom,
        chunk_size=chunk_size_to_use,
        n_processes=n_processes_to_use,
        min_n_cells_threshold=min_n_cells_threshold_to_use,
        mean_type='tri_mean',
        verbose=True
    )

    end_time_computation_of_communication_scores_cell_type_no_mask = time.time()

    for significance_test in significance_tests:
        mnc_object_cell_type.compute_significance(
            statistical_test=significance_test
        )

    end_time_significance_cell_type_no_mask = time.time()

    for significance_test in significance_tests:
        mnc_object_cell_type.correct_p_values(
            statistical_test=significance_test,
            method='by'
        )

    end_time_correction_cell_type_no_mask = time.time()

    mnc_object_cell_type.save(
        path_to_mnc_cell_type_no_mask
    )
    start_time_total_cell_type_mask = time.time()
    mask_xr_cell_type, wasserstein_distances_xr_cell_type = compute_wasserstein_mask(
        mnc_object=mnc_object_cell_type,
        top_percentile=99,
        p=10,
        normalized=False,
        exclude_zero_distributions=True,
        return_wasserstein_distances=True
    )
    end_time_mask_cell_type = time.time()

    for significance_test in significance_tests:
        mnc_object_cell_type.compute_significance(
            statistical_test=significance_test,
            mask=mask_xr_cell_type
        )

    end_time_significance_cell_type_mask = time.time()

    for significance_test in significance_tests:
        mnc_object_cell_type.correct_p_values(
            statistical_test=significance_test,
            method='by'
        )

    end_time_correction_cell_type_mask = time.time()
    mnc_object_cell_type.save(
        path_to_mnc_cell_type
    )

    mnc_object_grouping_by: MultiNeuronChatObject = MultiNeuronChatObject(
        condition_label_column='cogdx',
        condition_names=('NCI', 'AD'),
        subject_label_column='individualID',
        cell_type_label_column='grouping.by',
        db='human_extended',
    )
    start_time_total_grouping_by = time.time()
    mnc_object_grouping_by.compute_communication_scores(
        path_to_data_loom=path_to_subject_wise_normalized_loom,
        path_to_subject_wise_max_normalized=path_to_subject_wise_normalized_loom,
        chunk_size=chunk_size_to_use,
        n_processes=n_processes_to_use,
        min_n_cells_threshold=min_n_cells_threshold_to_use,
        mean_type='tri_mean',
        verbose=True
    )
    end_time_computation_of_communication_scores_grouping_by = time.time()

    for significance_test in significance_tests:
        mnc_object_grouping_by.compute_significance(
            statistical_test=significance_test
        )
    end_time_significance_grouping_by = time.time()

    for significance_test in significance_tests:
        mnc_object_grouping_by.correct_p_values(
            statistical_test=significance_test,
            method='by'
        )

    end_time_correction_grouping_by = time.time()

    mnc_object_grouping_by.save(
        path_to_mnc_grouping_by_no_mask
    )

    start_time_total_grouping_by_mask = time.time()
    mask_xr_grouping_by, wasserstein_distances_xr_grouping_by = compute_wasserstein_mask(
        mnc_object=mnc_object_grouping_by,
        top_percentile=98,
        p=10,
        normalized=False,
        exclude_zero_distributions=True,
        return_wasserstein_distances=True
    )
    end_time_mask_grouping_by = time.time()

    for significance_test in significance_tests:
        mnc_object_grouping_by.compute_significance(
            statistical_test=significance_test,
            mask=mask_xr_grouping_by
        )

    end_time_significance_grouping_by_mask = time.time()

    for significance_test in significance_tests:
        mnc_object_grouping_by.correct_p_values(
            statistical_test=significance_test,
            method='by'
        )

    end_time_correction_grouping_by_mask = time.time()
    mnc_object_grouping_by.save(
        path_to_mnc_grouping_by
    )

    # Save the wasserstein distance in pickle format. Join them together in a dictionary
    wasserstein_dict_cell_type: dict[str, xr.DataArray] = {
        'wasserstein_mask': mask_xr_cell_type,
        'wasserstein_distances': wasserstein_distances_xr_cell_type
    }
    wasserstein_dict_grouping_by: dict[str, xr.DataArray] = {
        'wasserstein_mask': mask_xr_grouping_by,
        'wasserstein_distances': wasserstein_distances_xr_grouping_by
    }

    with open(path_to_wasserstein_cell_type, 'wb') as f:
        pickle.dump(wasserstein_dict_cell_type, f)

    with open(path_to_wasserstein_grouping_by, 'wb') as f:
        pickle.dump(wasserstein_dict_grouping_by, f)


    # Save the timings in a dictionary and pickle it
    time_dict: dict[str, dict[str, tuple[float, float]]] = {
        'cell_type': {
            'total_no_mask': (start_time_total_cell_type_no_mask, end_time_correction_cell_type_no_mask),
            'computation_of_communication_scores': (start_time_total_cell_type_no_mask, end_time_computation_of_communication_scores_cell_type_no_mask),
            'significance_no_mask': (end_time_computation_of_communication_scores_cell_type_no_mask, end_time_significance_cell_type_no_mask),
            'correction_no_mask': (end_time_significance_cell_type_no_mask, end_time_correction_cell_type_no_mask),
            'mask': (start_time_total_cell_type_mask, end_time_mask_cell_type),
            'significance_mask': (end_time_mask_cell_type, end_time_significance_cell_type_mask),
            'correction_mask': (end_time_significance_cell_type_mask, end_time_correction_cell_type_mask),
        },
        'grouping_by': {
            'total_no_mask': (start_time_total_grouping_by, end_time_correction_grouping_by),
            'computation_of_communication_scores': (start_time_total_grouping_by, end_time_computation_of_communication_scores_grouping_by),
            'significance_no_mask': (end_time_computation_of_communication_scores_grouping_by, end_time_significance_grouping_by),
            'correction_no_mask': (end_time_significance_grouping_by, end_time_correction_grouping_by),
            'mask': (start_time_total_grouping_by_mask, end_time_mask_grouping_by),
            'significance_mask': (end_time_mask_grouping_by, end_time_significance_grouping_by_mask),
            'correction_mask': (end_time_significance_grouping_by_mask, end_time_correction_grouping_by_mask),
        }
    }
    with open(path_to_timings, 'wb') as f:
        pickle.dump(time_dict, f)

if __name__ == '__main__':
    main()
