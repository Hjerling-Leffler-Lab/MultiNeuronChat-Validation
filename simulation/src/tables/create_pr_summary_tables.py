import os

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_summary_files',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_pr_summary_table_excel',
    type=str,
    required=True,
)
args = parser.parse_args()

path_to_summary_files: str = args.path_to_summary_files
path_to_pr_summary_table_excel: str = args.path_to_pr_summary_table_excel


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

def main():
    means: list[str] = [
        'tri_mean_0',
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
    statistical_tests: list[str] = ['KS', 'Anderson', 'CVM', 'MannWhitneyU']

    summary_table: pd.DataFrame = pd.DataFrame(columns=[
        'Mean Type',
        'Trim Mean Value',
        'Log2 FC',
        'Proportion Affected Donors',
        'Statistical Test',
        'With Wasserstein',
        'Repeat Number',
        'Precision',
        'Recall'
    ])

    for mean in means:
        mean_str, trim_value = convert_mean_to_str_version(mean)

        for case in cases:
            log2_fc_str = convert_cases_to_log2_fc_str(case)

            for statistical_test in statistical_tests:
                for proportion in proportion_affected_donors:
                    path_to_file: str = os.path.join(path_to_summary_files, f'{mean}_{statistical_test}_{case}_{proportion}.pkl')

                    if not os.path.exists(path_to_file):
                        raise FileNotFoundError(f"File not found: {path_to_file}")

                    df: pd.DataFrame = pd.read_pickle(path_to_file)

                    wasserstein_counter: int = 0
                    non_wasserstein_counter: int = 0
                    for row_i, row in df.iterrows():
                        # Determine if it's with or without wasserstein distance
                        if 'Wasserstein' in row['Statistical Test']:
                            summary_table = pd.concat([
                                summary_table,
                                pd.DataFrame([{
                                    'Mean Type': mean_str,
                                    'Trim Mean Value': trim_value,
                                    'Log2 FC': log2_fc_str,
                                    'Proportion Affected Donors': proportion,
                                    'Statistical Test': statistical_test,
                                    'With Wasserstein': True,
                                    'Repeat Number': wasserstein_counter,
                                    'Precision': row['Precision'],
                                    'Recall': row['Recall']
                                }])
                            ], ignore_index=True)

                            wasserstein_counter += 1
                        else:
                            summary_table = pd.concat([
                                summary_table,
                                pd.DataFrame([{
                                    'Mean Type': mean_str,
                                    'Trim Mean Value': trim_value,
                                    'Log2 FC': log2_fc_str,
                                    'Proportion Affected Donors': proportion,
                                    'Statistical Test': statistical_test,
                                    'With Wasserstein': False,
                                    'Repeat Number': non_wasserstein_counter,
                                    'Precision': row['Precision'],
                                    'Recall': row['Recall']
                                }])
                            ], ignore_index=True)

                            non_wasserstein_counter += 1

    summary_table.to_excel(path_to_pr_summary_table_excel, index=False)

if __name__ == "__main__":
    main()