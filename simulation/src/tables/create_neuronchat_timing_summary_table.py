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
    '--path_to_timing_comparison_results',
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
path_to_timing_comparison_results: str = args.path_to_timing_comparison_results

path_to_timing_result_summary_csv: str = args.path_to_timing_result_summary_csv
path_to_timing_result_summary_xlsx: str = args.path_to_timing_result_summary_xlsx


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
        'Log2 FC',
        'Proportion of Affected Donors',
        'Repeat Index',

        'Start Reading Loom Time',
        'End Reading Loom Time',
        'Difference Reading Loom Time',

        'Start NeuronChat Time CTRL M0',
        'End NeuronChat Time CTRL M0',
        'Difference NeuronChat Time CTRL M0',
        'Start NeuronChat Time CASE M0',
        'End NeuronChat Time CASE M0',
        'Difference NeuronChat Time CASE M0',

        'Start NeuronChat Time CTRL M100',
        'End NeuronChat Time CTRL M100',
        'Difference NeuronChat Time CTRL M100',
        'Start NeuronChat Time CASE M100',
        'End NeuronChat Time CASE M100',
        'Difference NeuronChat Time CASE M100',

        'Start Comparison Time M0',
        'End Comparison Time M0',
        'Difference Comparison Time M0',

        'Start Comparison Time M100',
        'End Comparison Time M100',
        'Difference Comparison Time M100',
    ])


    for case in cases:
        for proportion in proportion_affected_donors:
            for i in range(10):
                path_to_timing_results_csv: str = os.path.join(path_to_timing_results, f'{case}_{proportion}_{i}.csv')
                path_to_timing_comparison_results_csv: str = os.path.join(path_to_timing_comparison_results, f'{case}_{proportion}_{i}_comparison.csv')

                # Load the timing results
                # Columns are: start.time.reading_loom	end.time.reading_loom	diff.time.reading_loom	start.time.m0	end.time.m0.ctrl	end.time.m0	diff.time.m0.ctrl	diff.time.m0.case	diff.time.m0	start.time.m100	end.time.m100.ctrl	end.time.m100	diff.time.m100.ctrl	diff.time.m100.case	diff.time.m100
                timing_results_df: pd.DataFrame = pd.read_csv(path_to_timing_results_csv)

                # Convert all the diff.time.* columns from minutes to seconds
                for col in timing_results_df.columns:
                    if col.startswith('diff.time.'):
                        timing_results_df[col] = timing_results_df[col] * 60

                # Load the timing comparison results
                # Comlumns are: case	proportion	dataset_id	start_time_m0	end_time_m0	start_time_m100	end_time_m100	duration_m0_secs	duration_m100_secs
                timing_comparison_results_df: pd.DataFrame = pd.read_csv(path_to_timing_comparison_results_csv)

                log2_fc: str = convert_cases_to_log2_fc_str(case)
                row_dict: dict = {
                    'Log2 FC': log2_fc,
                    'Proportion of Affected Donors': proportion,
                    'Repeat Index': i,

                    'Start Reading Loom Time': timing_results_df['start.time.reading_loom'].values[0],
                    'End Reading Loom Time': timing_results_df['end.time.reading_loom'].values[0],
                    'Difference Reading Loom Time': timing_results_df['diff.time.reading_loom'].values[0],

                    'Start NeuronChat Time CTRL M0': timing_results_df['start.time.m0'].values[0],
                    'End NeuronChat Time CTRL M0': timing_results_df['end.time.m0.ctrl'].values[0],
                    'Difference NeuronChat Time CTRL M0': timing_results_df['diff.time.m0.ctrl'].values[0],
                    'Start NeuronChat Time CASE M0': timing_results_df['end.time.m0.ctrl'].values[0],
                    'End NeuronChat Time CASE M0': timing_results_df['end.time.m0'].values[0],
                    'Difference NeuronChat Time CASE M0': timing_results_df['diff.time.m0.case'].values[0],

                    'Start NeuronChat Time CTRL M100': timing_results_df['start.time.m100'].values[0],
                    'End NeuronChat Time CTRL M100': timing_results_df['end.time.m100.ctrl'].values[0],
                    'Difference NeuronChat Time CTRL M100': timing_results_df['diff.time.m100.ctrl'].values[0],
                    'Start NeuronChat Time CASE M100': timing_results_df['end.time.m100.ctrl'].values[0],
                    'End NeuronChat Time CASE M100': timing_results_df['end.time.m100'].values[0],
                    'Difference NeuronChat Time CASE M100': timing_results_df['diff.time.m100.case'].values[0],

                    'Start Comparison Time M0': timing_comparison_results_df['start_time_m0'].values[0],
                    'End Comparison Time M0': timing_comparison_results_df['end_time_m0'].values[0],
                    'Difference Comparison Time M0': timing_comparison_results_df['duration_m0_secs'].values[0],

                    'Start Comparison Time M100': timing_comparison_results_df['start_time_m100'].values[0],
                    'End Comparison Time M100': timing_comparison_results_df['end_time_m100'].values[0],
                    'Difference Comparison Time M100': timing_comparison_results_df['duration_m100_secs'].values[0],
                }

                timing_df = pd.concat([timing_df, pd.DataFrame([row_dict])], ignore_index=True)

    # Create output folder if does not exist
    os.makedirs(os.path.dirname(path_to_timing_result_summary_csv), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_timing_result_summary_xlsx), exist_ok=True)
    # Save df to CSV and Excel
    timing_df.to_csv(path_to_timing_result_summary_csv, index=False)
    timing_df.to_excel(path_to_timing_result_summary_xlsx, index=False)

if __name__ == "__main__":
    main()