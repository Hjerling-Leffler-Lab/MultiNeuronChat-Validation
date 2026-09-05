import os

import multiprocessing

import pickle

import warnings

import xarray as xr

from multineuronchat import MultiNeuronChatObject
from multineuronchat.masks import (
    compute_wasserstein_mask,
    compute_variance_mask,
    compute_abundance_mask,
)

import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_subject_wise_max_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_mnc_output_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_wasserstein_dict_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_timing_dict_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--mean_type',
    type=str,
    required=True,
)
parser.add_argument(
    '--trim_mean_fraction',
    type=float,
    required=True,
)
parser.add_argument(
    '--n_processes',
    type=int,
    required=True,
)
args = parser.parse_args()

path_to_subject_wise_max_normalized_loom: str = args.path_to_subject_wise_max_normalized_loom
path_to_mnc_output_file: str = args.path_to_mnc_output_file
path_to_wasserstein_dict_file: str = args.path_to_wasserstein_dict_file
path_to_timing_dict_file: str = args.path_to_timing_dict_file
mean_type: str = args.mean_type
trim_mean_fraction: float = args.trim_mean_fraction
n_processes: int = max(1, min(args.n_processes, multiprocessing.cpu_count()))


def main():
    path_to_mnc_output_dir: str = os.path.dirname(path_to_mnc_output_file)
    path_to_wasserstein_output_dir: str = os.path.dirname(path_to_wasserstein_dict_file)
    path_to_timing_output_dir: str = os.path.dirname(path_to_timing_dict_file)

    os.makedirs(path_to_mnc_output_dir, exist_ok=True)
    os.makedirs(path_to_wasserstein_output_dir, exist_ok=True)
    os.makedirs(path_to_timing_output_dir, exist_ok=True)

    timings: dict[str, tuple[float, float]] = {}

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject(
        condition_label_column='Disease',
        condition_names=('CTRL', 'CASE'),
        subject_label_column='Donor',
        cell_type_label_column='cluster_name_15CTs',
        db='human_extended',
    )

    print('Computing communication scores')
    start_time_total_fine_no_mask = time.time()

    mnc_object.compute_communication_scores(
        path_to_data_loom=path_to_subject_wise_max_normalized_loom,
        path_to_subject_wise_max_normalized=path_to_subject_wise_max_normalized_loom,
        n_processes=n_processes,
        min_n_cells_threshold=10,
        mean_type=mean_type,
        trim_mean_fraction=trim_mean_fraction,
        verbose=True,
    )

    end_time_computation_of_communication_scores_fine_no_mask = time.time()
    timings['computation_of_communication_scores'] = (
        start_time_total_fine_no_mask,
        end_time_computation_of_communication_scores_fine_no_mask,
    )

    print('Computing prefilter statistics')
    start_time_wasserstein_mask = time.time()

    # Legacy Wasserstein prefilter, kept only for the revision comparison. The distance is not
    # independent of the significance-test statistics under the null (Bourgon, Gentleman & Huber 2010),
    # so it is deprecated as a prefilter in the package. We keep it here deliberately to show the
    # comparison against the label-blind replacements below, hence the DeprecationWarning is silenced.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        mask_xr, wasserstein_distances_xr = compute_wasserstein_mask(
            mnc_object,
            top_percentile=99,
            p=10,
            normalized=False,
            exclude_zero_distributions=True,
            return_wasserstein_distances=True,
        )

    end_time_wasserstein_mask = time.time()
    timings['wasserstein_mask'] = (
        start_time_wasserstein_mask,
        end_time_wasserstein_mask,
    )

    # Label-blind replacements for the Wasserstein prefilter (independent of the test statistics under
    # the null). We save the raw per-triple statistics rather than a mask at a fixed retention, so the
    # downstream benchmark can apply both the top-10% and the >0 retention cutoffs without re-running
    # this step. compute_variance_mask uses the squared coefficient of variation ('cv2') of the pooled
    # communication scores; compute_abundance_mask uses the bottleneck (min) of the pooled mean ligand
    # and target abundances. Each is timed separately so the runtime comparison can report them per
    # filter rather than as a combined figure.
    start_time_variance_mask = time.time()

    _, variances_xr = compute_variance_mask(
        mnc_object,
        min_value=0.0,
        statistic='cv2',
        return_variances=True,
    )

    end_time_variance_mask = time.time()
    timings['variance_mask'] = (
        start_time_variance_mask,
        end_time_variance_mask,
    )

    start_time_abundance_mask = time.time()

    _, abundances_xr = compute_abundance_mask(
        mnc_object,
        min_value=0.0,
        return_abundances=True,
    )

    end_time_abundance_mask = time.time()
    timings['abundance_mask'] = (
        start_time_abundance_mask,
        end_time_abundance_mask,
    )

    wasserstein_dict: dict[str, xr.DataArray] = {
        'mask': mask_xr,
        'wasserstein_distances': wasserstein_distances_xr,
        'variances': variances_xr,
        'abundances': abundances_xr,
    }

    print('Computing significance')
    for statistical_test in ['KS', 'Anderson', 'CVM', 'MannWhitneyU']:
        start_time_statistical_test = time.time()

        mnc_object.compute_significance(
            statistical_test=statistical_test,
            random_state=42,
        )

        end_time_statistical_test = time.time()
        timings[f'significance_{statistical_test}'] = (
            start_time_statistical_test,
            end_time_statistical_test,
        )

    print('Correcting p-values')
    for statistical_test in ['KS', 'Anderson', 'CVM', 'MannWhitneyU']:
        start_time_correction = time.time()
        mnc_object.correct_p_values(statistical_test=statistical_test)
        end_time_correction = time.time()

        timings[f'correction_{statistical_test}'] = (
            start_time_correction,
            end_time_correction,
        )

    # Save the MultiNeuronChat object
    mnc_object.save(path_to_mnc_output_file)

    # Save wasserstein distances
    with open(path_to_wasserstein_dict_file, 'wb') as f:
        pickle.dump(wasserstein_dict, f)

    # Save timings
    with open(path_to_timing_dict_file, 'wb') as f:
        pickle.dump(timings, f)

if __name__ == "__main__":
    main()
