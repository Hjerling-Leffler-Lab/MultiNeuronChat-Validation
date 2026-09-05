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


# Prefilter label prefixes used in the 'Statistical Test' column of the summary dataframes, mapped to
# human-readable names. Mirrors utils.filter_display_names (the tables package cannot import the plotting
# utils, so the mapping is duplicated here). 'None' is the unfiltered baseline, whose label has no prefix.
FILTER_DISPLAY_NAMES: dict[str, str] = {
    'None': 'No filter',
    'Wasserstein': 'Wasserstein (top 10%, legacy)',
    'Variance-top10': 'Variance (top 10%)',
    'Variance-min0': 'Variance (> 0)',
    'Abundance-top10': 'Abundance (top 10%)',
    'Abundance-min0': 'Abundance (> 0)',
}


def parse_filter_key(label: str, statistical_test: str) -> str:
    """
    Extract the prefilter key from a 'Statistical Test' label, given the (known) statistical test name.

    :param label: e.g. 'KS_adj' or 'Variance-top10_KS_adj'
    :param statistical_test: the test name the file is for, e.g. 'KS'
    :return: the filter key, e.g. 'None' or 'Variance-top10'
    """
    core: str = label[:-len('_adj')] if label.endswith('_adj') else label
    if core == statistical_test:
        return 'None'
    suffix: str = f'_{statistical_test}'
    if core.endswith(suffix):
        return core[:-len(suffix)]
    return 'None'


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
    statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']

    summary_table: pd.DataFrame = pd.DataFrame(columns=[
        'Mean Type',
        'Trim Mean Value',
        'Log2 FC',
        'Proportion Affected Donors',
        'Statistical Test',
        'Filter',
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

                    # Per-filter repeat counter: each dataset contributes one row per prefilter.
                    filter_counters: dict[str, int] = {}
                    for row_i, row in df.iterrows():
                        filter_key: str = parse_filter_key(row['Statistical Test'], statistical_test)
                        repeat: int = filter_counters.get(filter_key, 0)

                        summary_table = pd.concat([
                            summary_table,
                            pd.DataFrame([{
                                'Mean Type': mean_str,
                                'Trim Mean Value': trim_value,
                                'Log2 FC': log2_fc_str,
                                'Proportion Affected Donors': proportion,
                                'Statistical Test': statistical_test,
                                'Filter': FILTER_DISPLAY_NAMES.get(filter_key, filter_key),
                                'Repeat Number': repeat,
                                'Precision': row['Precision'],
                                'Recall': row['Recall']
                            }])
                        ], ignore_index=True)

                        filter_counters[filter_key] = repeat + 1

    summary_table.to_excel(path_to_pr_summary_table_excel, index=False)

if __name__ == "__main__":
    main()