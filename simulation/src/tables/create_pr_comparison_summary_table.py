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


# Statistical test this comparison table is built for (the ROSMAP/paper default).
STATISTICAL_TEST: str = 'KS'

# Prefilters compared against the unfiltered baseline, as (label prefix, display name), in table order.
FILTERS: list[tuple[str, str]] = [
    ('Wasserstein', 'Wasserstein (top 10%, legacy)'),
    ('Variance-top10', 'Variance (top 10%)'),
    ('Variance-min0', 'Variance (> 0)'),
    ('Abundance-top10', 'Abundance (top 10%)'),
    ('Abundance-min0', 'Abundance (> 0)'),
]


def main():
    log2_fcs: list[str] = ['0.3', '0.5', '1', '1.5']
    proportion_of_affected_donors: list[str] = ['0.1', '0.3', '0.5', '0.8', '1.0']

    # One block per metric: the unfiltered baseline, then each filter's value and its percentage-point
    # improvement over the baseline.
    summary_table_columns: list[str] = ['Proportion of Affected Donors', 'Log2 FC']
    for metric in ['Precision', 'Recall']:
        summary_table_columns.append(f'{metric} – No filter')
        for _, display_name in FILTERS:
            summary_table_columns.append(f'{metric} – {display_name}')
            summary_table_columns.append(f'{metric} – {display_name} (Δpp)')

    summary_table = pd.DataFrame(columns=summary_table_columns)

    for log2fc in log2_fcs:
        for proportion in proportion_of_affected_donors:
            file_name: str = f'tri_mean_0_{STATISTICAL_TEST}_CASE_{log2fc}_{proportion}.pkl'
            file_path: str = os.path.join(path_to_summary_files, file_name)

            row_dict = {
                'Proportion of Affected Donors': proportion,
                'Log2 FC': log2fc,
            }

            df = pd.read_pickle(file_path)

            for metric in ['Precision', 'Recall']:
                baseline_label: str = f'{STATISTICAL_TEST}_adj'
                baseline_mean: float = df[df['Statistical Test'] == baseline_label][metric].mean()
                row_dict[f'{metric} – No filter'] = baseline_mean

                for filter_key, display_name in FILTERS:
                    filter_label: str = f'{filter_key}_{STATISTICAL_TEST}_adj'
                    filter_mean: float = df[df['Statistical Test'] == filter_label][metric].mean()

                    row_dict[f'{metric} – {display_name}'] = filter_mean
                    row_dict[f'{metric} – {display_name} (Δpp)'] = (filter_mean - baseline_mean) * 100

            summary_table = pd.concat([
                summary_table,
                pd.DataFrame([row_dict])
            ], ignore_index=True)


    summary_table.to_excel(path_to_pr_summary_table_excel, index=False)

if __name__ == '__main__':
    main()
