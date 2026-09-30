import os

import time

import pickle

from multineuronchat import MultiNeuronChatObject

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_subject_wise_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_cell_type',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_grouping_by',
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
    default=['KS'],
    help='List of significance tests to use. Options: KS, CVM, MannWhitneyU'
)
args = parser.parse_args()

# Paths to the input files and output folder
path_to_subject_wise_normalized_loom: str = args.path_to_subject_wise_normalized_loom

path_to_mnc_cell_type: str = args.path_to_mnc_cell_type
path_to_mnc_grouping_by: str = args.path_to_mnc_grouping_by
path_to_timings: str = args.path_to_timings

chunk_size_to_use: int = args.chunk_size_to_use
n_processes_to_use: int = args.n_processes_to_use
min_n_cells_threshold_to_use: int = args.min_n_cells_threshold_to_use
significance_tests: list[str] = args.significance_tests


def run_mnc(cell_type_label_column: str, path_to_output: str) -> dict[str, tuple[float, float]]:
    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject(
        condition_label_column='cogdx',
        condition_names=('NCI', 'AD'),
        subject_label_column='individualID',
        cell_type_label_column=cell_type_label_column,
        db='human_extended',
    )

    start_time_total = time.time()
    mnc_object.compute_communication_scores(
        path_to_data_loom=path_to_subject_wise_normalized_loom,
        path_to_subject_wise_max_normalized=path_to_subject_wise_normalized_loom,
        chunk_size=chunk_size_to_use,
        n_processes=n_processes_to_use,
        min_n_cells_threshold=min_n_cells_threshold_to_use,
        mean_type='tri_mean',
        verbose=True
    )
    end_time_computation_of_communication_scores = time.time()

    for significance_test in significance_tests:
        mnc_object.compute_significance(
            statistical_test=significance_test
        )
    end_time_significance = time.time()

    for significance_test in significance_tests:
        mnc_object.correct_p_values(
            statistical_test=significance_test,
            method='by'
        )
    end_time_correction = time.time()

    mnc_object.save(
        path_to_output
    )

    return {
        'total': (start_time_total, end_time_correction),
        'computation_of_communication_scores': (start_time_total, end_time_computation_of_communication_scores),
        'significance': (end_time_computation_of_communication_scores, end_time_significance),
        'correction': (end_time_significance, end_time_correction),
    }


def main():
    if not os.path.exists(path_to_subject_wise_normalized_loom):
        raise FileNotFoundError(f'File not found: {path_to_subject_wise_normalized_loom}')

    # Create output folder if it does not exist
    all_output_paths = [
        path_to_mnc_cell_type,
        path_to_mnc_grouping_by,
        path_to_timings
    ]
    for output_path in all_output_paths:
        output_folder = os.path.dirname(output_path)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    # Save the timings in a dictionary and pickle it
    time_dict: dict[str, dict[str, tuple[float, float]]] = {
        'cell_type': run_mnc('cell.type', path_to_mnc_cell_type),
        'grouping_by': run_mnc('grouping.by', path_to_mnc_grouping_by),
    }
    with open(path_to_timings, 'wb') as f:
        pickle.dump(time_dict, f)

if __name__ == '__main__':
    main()
