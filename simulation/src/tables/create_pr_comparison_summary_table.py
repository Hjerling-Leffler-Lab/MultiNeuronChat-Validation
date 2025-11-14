import os

import pandas as pd

import argparse

parser = argparse.ArgumentParser()

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


def main():
    log2_fcs: list[str] = ['0.3', '0.5', '1', '1.5']
    proportion_of_affected_donors: list[str] = ['0.1', '0.3', '0.5', '0.8', '1.0']

    # Add the columns KS Repeat 1 - 10 and Wasserstein KS Repeat 1 - 10 for Precision and Recall
    summary_table_columns = ['Proportion of Affected Donors', 'Log2 FC']
    for metric in ['Precision', 'Recall']:
        for metric_2 in ['Mean', 'Wasserstein + Mean', 'Percentage Point Improvement']:
            summary_table_columns.append(f'{metric} – {metric_2}')

    summary_table = pd.DataFrame(columns=summary_table_columns)

    for log2fc in log2_fcs:
        for proportion in proportion_of_affected_donors:
            file_name: str = f'tri_mean_0_KS_CASE_{log2fc}_{proportion}.pkl'
            file_path: str = os.path.join(path_to_summary_files, file_name)

            row_dict = {
                'Proportion of Affected Donors': proportion,
                'Log2 FC': log2fc,
            }

            df = pd.read_pickle(file_path)

            # Get all KS_adj and all Wasserstein_KS_adj
            df_ks: pd.DataFrame = df[df['Statistical Test'] == 'KS_adj']
            df_wass_ks: pd.DataFrame = df[df['Statistical Test'] == 'Wasserstein_KS_adj']

            mean_precision_ks = df_ks['Precision'].mean()
            mean_precision_wass_ks = df_wass_ks['Precision'].mean()

            precision_improvement_percentage_points = (mean_precision_wass_ks - mean_precision_ks) * 100

            mean_recall_ks = df_ks['Recall'].mean()
            mean_recall_wass_ks = df_wass_ks['Recall'].mean()

            recall_improvement_percentage_points = (mean_recall_wass_ks - mean_recall_ks) * 100

            row_dict['Precision – Mean'] = mean_precision_ks
            row_dict['Precision – Wasserstein + Mean'] = mean_precision_wass_ks
            row_dict['Recall – Mean'] = mean_recall_ks
            row_dict['Recall – Wasserstein + Mean'] = mean_recall_wass_ks

            row_dict['Precision – Percentage Point Improvement'] = precision_improvement_percentage_points
            row_dict['Recall – Percentage Point Improvement'] = recall_improvement_percentage_points

            summary_table = pd.concat([
                summary_table,
                pd.DataFrame([row_dict])
            ], ignore_index=True)


    summary_table.to_excel(path_to_pr_summary_table_excel, index=False)

if __name__ == '__main__':
    main()
