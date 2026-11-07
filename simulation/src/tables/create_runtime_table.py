import os

import argparse

import pandas as pd

parser = argparse.ArgumentParser(
    description='Combine the MultiNeuronChat and NeuronChat runtimes into one workbook (Supplementary Table T3).'
)
parser.add_argument(
    '--path_to_mnc_runtime_csv',
    type=str,
    required=True,
    help='Output of create_timing_summary_tables.py',
)
parser.add_argument(
    '--path_to_neuronchat_runtime_csv',
    type=str,
    required=True,
    help='Output of create_neuronchat_timing_summary_table.py',
)
parser.add_argument(
    '--path_to_runtime_table_excel',
    type=str,
    required=True,
)
args = parser.parse_args()


def main():
    # Read as strings where the source tables store labels (e.g. 'Log2 FC' '1.0'), so the workbook shows them unchanged.
    label_columns: dict[str, type] = {'Trim Mean Value': str, 'Log2 FC': str, 'Proportion of Affected Donors': str}
    mnc_runtime: pd.DataFrame = pd.read_csv(args.path_to_mnc_runtime_csv, dtype=label_columns)
    neuronchat_runtime: pd.DataFrame = pd.read_csv(args.path_to_neuronchat_runtime_csv, dtype=label_columns)

    os.makedirs(os.path.dirname(args.path_to_runtime_table_excel), exist_ok=True)
    with pd.ExcelWriter(args.path_to_runtime_table_excel) as writer:
        mnc_runtime.to_excel(writer, sheet_name='MultiNeuronChat', index=False)
        neuronchat_runtime.to_excel(writer, sheet_name='NeuronChat', index=False)


if __name__ == '__main__':
    main()
