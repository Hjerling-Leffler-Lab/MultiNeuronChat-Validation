import os

import argparse

import pickle

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_timing_results',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_timing_result_summary_csv',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_timing_result_summary_xlsx',
    type=str,
    required=True,
)
args = parser.parse_args()

path_to_timing_results: str = args.path_to_timing_results
path_to_timing_result_summary_csv: str = args.path_to_timing_result_summary_csv
path_to_timing_result_summary_xlsx: str = args.path_to_timing_result_summary_xlsx

def convert_mean_to_str_version(mean: str) -> tuple[str, str]:
    """
    Convert the mean string into "mean" and "Trim Mean Value" components.

    Trim Mean Value is '0' for mean and tri-mean

    :param mean: str: Mean string
    :return: tuple[str, str]: Mean type and Trim Mean Value
    """
    if mean == 'mean_0':
        return 'Mean', '0'
    elif mean == 'tri_mean_0':
        return 'Triangular Mean', '0'
    elif mean.startswith('trim_mean_'):
        trim_value: str = mean.split('_')[-1]
        return 'Trimmed Mean', trim_value
    else:
        raise ValueError(f"Unknown mean type: {mean}")


def convert_cases_to_log2_fc_str(case: str) -> str:
    """
    Convert case string into log2 FC string.

    :param case: str: Case string
    :return: str: Log2 FC string
    """
    if case.startswith('CASE_'):
        log2_fc: str = case.split('_')[-1]
        return log2_fc
    else:
        raise ValueError(f"Unknown case type: {case}")

def split_timing_tuple(timing_tuple: tuple[float, float]) -> tuple[float, float, float]:
    """
    Split a timing tuple into start, end, and difference.

    :param timing_tuple: tuple[float, float]: Timing tuple
    :return: tuple[float, float, float]: Start, end, and difference
    """
    start: float = timing_tuple[0]
    end: float = timing_tuple[1]
    difference: float = end - start
    return start, end, difference


def main():
    means: list[str] = [
        'mean_0',
        'tri_mean_0',
        'trim_mean_0.1',
        'trim_mean_0.05'
    ]
    cases: list[str] = [
        'CASE_0.3',
        'CASE_0.5',
        'CASE_1',
        'CASE_1.5',
    ]
    proportion_affected_donors: list[str] = [
        '0.1',
        '0.3',
        '0.5',
        '0.8',
        '1.0'
    ]

    timing_df: pd.DataFrame = pd.DataFrame(columns=[
        'Mean Type',
        'Trim Mean Value',
        'Log2 FC',
        'Proportion of Affected Donors',
        'Repeat Index',
        'computation_of_communication_scores',
        'computation_of_communication_scores_start',
        'computation_of_communication_scores_end',
        'computation_of_communication_scores_difference',
        'wasserstein_mask',
        'wasserstein_mask_start',
        'wasserstein_mask_end',
        'wasserstein_mask_difference',
        'significance_KS',
        'significance_KS_start',
        'significance_KS_end',
        'significance_KS_difference',
        'significance_Anderson',
        'significance_Anderson_start',
        'significance_Anderson_end',
        'significance_Anderson_difference',
        'significance_CVM',
        'significance_CVM_start',
        'significance_CVM_end',
        'significance_CVM_difference',
        'significance_MannWhitneyU',
        'significance_MannWhitneyU_start',
        'significance_MannWhitneyU_end',
        'significance_MannWhitneyU_difference',
        'correction_KS',
        'correction_KS_start',
        'correction_KS_end',
        'correction_KS_difference',
        'correction_Anderson',
        'correction_Anderson_start',
        'correction_Anderson_end',
        'correction_Anderson_difference',
        'correction_CVM',
        'correction_CVM_start',
        'correction_CVM_end',
        'correction_CVM_difference',
        'correction_MannWhitneyU',
        'correction_MannWhitneyU_start',
        'correction_MannWhitneyU_end',
        'correction_MannWhitneyU_difference',
    ])

    for mean in means:
        mean_type, trim_mean_value = convert_mean_to_str_version(mean)

        for case in cases:

            log2_fc: str = convert_cases_to_log2_fc_str(case)

            for proportion_affected_donor in proportion_affected_donors:
                path_to_specific_timings: str = os.path.join(path_to_timing_results, mean, case, proportion_affected_donor)

                # Get all files in this directory
                timing_files: list[str] = [
                    os.path.join(path_to_specific_timings, x)
                    for x in os.listdir(path_to_specific_timings)
                    if x.endswith('.pkl')
                ]

                for i, timing_file in enumerate(timing_files):
                    with open(timing_file, 'rb') as f:
                        # Timing data is a dict with keys as timing types and values as tuples of floats for start and end times
                        timing_data: dict[str, float] = pickle.load(f)

                         # Extract the relevant timing information

                        timing_info: dict[str, str | float | tuple[float, float]] = {
                            'Mean Type': mean_type,
                            'Trim Mean Value': trim_mean_value,
                            'Log2 FC': log2_fc,
                            'Proportion of Affected Donors': proportion_affected_donor,
                            'Repeat Index': i,
                        }

                        for key in timing_data.keys():
                            timing_tuple: tuple[float, float] = timing_data.get(key, (np.nan, np.nan))
                            start, end, difference = split_timing_tuple(timing_tuple)
                            timing_info[key] = timing_tuple
                            timing_info[f'{key}_start'] = start
                            timing_info[f'{key}_end'] = end
                            timing_info[f'{key}_difference'] = difference


                        # Append the timing information to the DataFrame
                        timing_df = pd.concat([timing_df, pd.DataFrame([timing_info])], ignore_index=True)

    # Save df to CSV and Excel
    timing_df.to_csv(path_to_timing_result_summary_csv, index=False)
    print(f"Timing summary saved to {path_to_timing_result_summary_csv}")
    timing_df.to_excel(path_to_timing_result_summary_xlsx, index=False)
    print(f"Timing summary saved to {path_to_timing_result_summary_xlsx}")

if __name__ == "__main__":
    main()