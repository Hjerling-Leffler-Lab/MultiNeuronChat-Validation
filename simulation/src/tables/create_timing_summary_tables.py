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

    # Timed steps of step 7, in table order. Keys not listed here (the prefilter masks or a legacy Anderson test in
    # older timing files) are left out.
    timing_keys: list[str] = [
        'computation_of_communication_scores',
        'significance_KS',
        'significance_CVM',
        'significance_MannWhitneyU',
        'correction_KS',
        'correction_CVM',
        'correction_MannWhitneyU',
    ]

    rows: list[dict[str, str | int | float]] = []

    for mean in means:
        mean_type, trim_mean_value = convert_mean_to_str_version(mean)

        for case in cases:

            log2_fc: str = convert_cases_to_log2_fc_str(case)

            for proportion_affected_donor in proportion_affected_donors:
                path_to_specific_timings: str = os.path.join(path_to_timing_results, mean, case, proportion_affected_donor)

                # Timing files are named {case}_{proportion}_{dataset_index}.pkl
                timing_files: list[str] = sorted(
                    (x for x in os.listdir(path_to_specific_timings) if x.endswith('.pkl')),
                    key=lambda x: int(x[:-len('.pkl')].split('_')[-1]),
                )

                for timing_file in timing_files:
                    with open(os.path.join(path_to_specific_timings, timing_file), 'rb') as f:
                        # Timing data is a dict with keys as timing types and values as tuples of floats for start and end times
                        timing_data: dict[str, tuple[float, float]] = pickle.load(f)

                    timing_info: dict[str, str | int | float] = {
                        'Mean Type': mean_type,
                        'Trim Mean Value': trim_mean_value,
                        'Log2 FC': log2_fc,
                        'Proportion of Affected Donors': proportion_affected_donor,
                        'Repeat Index': int(timing_file[:-len('.pkl')].split('_')[-1]),
                    }

                    for key in timing_keys:
                        start, end, difference = split_timing_tuple(timing_data.get(key, (np.nan, np.nan)))
                        timing_info[f'{key}_start'] = start
                        timing_info[f'{key}_end'] = end
                        timing_info[f'{key}_difference'] = difference

                    rows.append(timing_info)

    timing_df: pd.DataFrame = pd.DataFrame(rows)

    # Save df to CSV and Excel
    timing_df.to_csv(path_to_timing_result_summary_csv, index=False)
    print(f"Timing summary saved to {path_to_timing_result_summary_csv}")
    timing_df.to_excel(path_to_timing_result_summary_xlsx, index=False)
    print(f"Timing summary saved to {path_to_timing_result_summary_xlsx}")

if __name__ == "__main__":
    main()