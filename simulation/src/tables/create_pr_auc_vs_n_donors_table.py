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
parser.add_argument(
    '--mean_types',
    nargs='+',
    default=['mean_0', 'tri_mean_0'],
)
parser.add_argument(
    '--cases',
    nargs='+',
    default=['CASE_0.5', 'CASE_1'],
)
parser.add_argument(
    '--n_donors',
    nargs='+',
    type=int,
    default=[2, 3, 5, 8, 10, 20],
)
args = parser.parse_args()

path_to_summary_pkls: str = args.path_to_summary_pkls
path_to_pr_auc_summary_table_excel: str = args.path_to_pr_auc_summary_table_excel
mean_types: list[str] = args.mean_types
cases: list[str] = args.cases
n_donors: list[int] = args.n_donors


def convert_mean_to_str_version(mean: str) -> tuple[str, str]:
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
    if case.startswith('CASE_'):
        return case.split('_')[-1]
    else:
        raise ValueError(f"Unknown case type: {case}")


def main():
    statistical_tests: list[str] = ['KS', 'Anderson', 'CVM', 'MannWhitneyU']

    pr_auc_summary_table: pd.DataFrame = pd.DataFrame(columns=[
        'Mean Type',
        'Trim Mean Value',
        'Log2 FC',
        'Donors per Sex per Group',
        'Donors per Group',
        'Total Study Donors',
    ] + [
        f'PR AUC – {statistical_test} – Repeat {repeat}'
        for statistical_test in statistical_tests
        for repeat in range(10)
    ])

    for mean in mean_types:
        mean_str, trim_value = convert_mean_to_str_version(mean)

        for case in cases:
            log2_fc_str = convert_cases_to_log2_fc_str(case)

            for n in n_donors:
                path_to_pkl: str = os.path.join(path_to_summary_pkls, f'{mean}_{case}_{n}.pkl')

                if not os.path.exists(path_to_pkl):
                    raise FileNotFoundError(f"File not found: {path_to_pkl}")

                with open(path_to_pkl, 'rb') as f:
                    summary_data: dict[str, list[float]] = pickle.load(f)

                row_data: dict[str, str | float] = {
                    'Mean Type': mean_str,
                    'Trim Mean Value': trim_value,
                    'Log2 FC': log2_fc_str,
                    'Donors per Sex per Group': n,
                    'Donors per Group': n * 2,
                    'Total Study Donors': n * 4,
                }

                for statistical_test in statistical_tests:
                    pr_aucs: list[float] = summary_data.get(statistical_test, [])

                    for repeat_index, pr_auc in enumerate(pr_aucs):
                        column_name: str = f'PR AUC – {statistical_test} – Repeat {repeat_index}'
                        row_data[column_name] = pr_auc

                pr_auc_summary_table = pd.concat([pr_auc_summary_table, pd.DataFrame([row_data])], ignore_index=True)

    pr_auc_summary_table.to_excel(path_to_pr_auc_summary_table_excel, index=False)


if __name__ == "__main__":
    main()
