import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--paths_to_summary_dfs',
    nargs='+',
    required=True,
    help='Summaries of plotting/plot_n_sweep_precision_recall_a.py',
)
parser.add_argument(
    '--paths_to_full_cohort_summary_dfs',
    nargs='+',
    required=True,
    help='Summaries of plotting/plot_precision_recall_a.py at proportion 1.0, the largest cohort of the sweep',
)
parser.add_argument(
    '--path_to_table_excel',
    type=str,
    required=True,
)
args = parser.parse_args()


def main():
    n_sweep_df: pd.DataFrame = pd.concat(
        [pd.read_pickle(path) for path in args.paths_to_summary_dfs], ignore_index=True,
    )
    # The sweep sub-samples donors from the full simulated cohort, so the full cohort with all case donors affected
    # (proportion 1.0) is its largest cohort.
    full_cohort_df: pd.DataFrame = pd.concat(
        [pd.read_pickle(path) for path in args.paths_to_full_cohort_summary_dfs], ignore_index=True,
    )
    full_cohort_df = full_cohort_df[full_cohort_df['Proportion'] == 1.0].drop(columns=['Proportion'])

    df: pd.DataFrame = pd.concat([n_sweep_df, full_cohort_df], ignore_index=True)
    df['Log2 FC'] = df['Case'].str.split('_').str[-1]

    group_columns: list[str] = ['Mean Type', 'Log2 FC', 'N Donors Per Group', 'Test']
    grouped = df.groupby(group_columns, sort=True)

    summary: pd.DataFrame = grouped.agg(**{
        'Precision (mean)': ('Precision', 'mean'),
        'Precision (SD)': ('Precision', 'std'),
        'Precision (median)': ('Precision', 'median'),
        'Datasets with calls': ('Precision', 'count'),
        'Recall (mean)': ('Recall', 'mean'),
        'Recall (SD)': ('Recall', 'std'),
        'Recall (median)': ('Recall', 'median'),
        'Significant calls (mean)': ('N Significant', 'mean'),
        'Tested hypotheses m (mean)': ('N Hypotheses', 'mean'),
        'True tetrads tested (mean)': ('N Ground Truth Tested', 'mean'),
        'Minimal attainable p-value': ('P Floor', 'first'),
        'Triples needed at floor for any call (mean)': ('N Needed At Floor', 'mean'),
        'Triples at or below floor (mean)': ('N At Floor', 'mean'),
    }).reset_index()
    summary = summary.rename(columns={'N Donors Per Group': 'Donors per Group'})

    per_dataset: pd.DataFrame = df[group_columns + [
        'Dataset Index', 'Precision', 'Recall', 'N Significant', 'N True Positives', 'N Hypotheses',
        'N Ground Truth Tested', 'Min Raw P', 'P Floor', 'N At Floor', 'N Needed At Floor',
    ]].sort_values(group_columns + ['Dataset Index'])

    with pd.ExcelWriter(args.path_to_table_excel) as writer:
        summary.to_excel(writer, sheet_name='Summary', index=False)
        per_dataset.to_excel(writer, sheet_name='Per dataset', index=False)


if __name__ == '__main__':
    main()
