import os

import pickle

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_summary_pkls',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_pr_auc_summary_table_excel',
    type=str,
    required=True,
)
args = parser.parse_args()

path_to_summary_pkls: str = args.path_to_summary_pkls
path_to_pr_auc_summary_table_excel: str = args.path_to_pr_auc_summary_table_excel

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
    statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']

    pr_auc_summary_table: pd.DataFrame = pd.DataFrame(columns=[
        'Mean Type',
        'Trim Mean Value',
        'Log2 FC',
        'Proportion of Affected Donors',
    ] + [
        f'PR AUC – {statistical_test} – Repeat {repeat}'
        for statistical_test in statistical_tests
        for repeat in range(10)
    ])

    for mean in means:
        mean_str, trim_value = convert_mean_to_str_version(mean)

        for case in cases:
                log2_fc_str = convert_cases_to_log2_fc_str(case)

                for proportion in proportion_affected_donors:
                    path_to_pkl: str = os.path.join(path_to_summary_pkls, f'{mean}_{case}_{proportion}.pkl')

                    if not os.path.exists(path_to_pkl):
                        raise FileNotFoundError(f"File not found: {path_to_pkl}")

                    with open(path_to_pkl, 'rb') as f:
                        summary_data: dict[str, list[float]] = pickle.load(f)

                    row_data: dict[str, str | float] = {
                        'Mean Type': mean_str,
                        'Trim Mean Value': trim_value,
                        'Log2 FC': log2_fc_str,
                        'Proportion of Affected Donors': proportion,
                    }

                    for statistical_test in statistical_tests:
                        pr_aucs: list[float] = summary_data.get(statistical_test, [])


                        for repeat_index, pr_auc in enumerate(pr_aucs):
                            # Get the mean type

                            column_name: str = f'PR AUC – {statistical_test} – Repeat {repeat_index}'
                            row_data[column_name] = pr_auc

                    pr_auc_summary_table = pd.concat([pr_auc_summary_table, pd.DataFrame([row_data])], ignore_index=True)

    pr_auc_summary_table.to_excel(path_to_pr_auc_summary_table_excel, index=False)


if __name__ == "__main__":
    main()
