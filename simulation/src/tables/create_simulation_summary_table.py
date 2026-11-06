import os
from copy import copy

import argparse

import openpyxl
from openpyxl.worksheet.formula import ArrayFormula
from openpyxl.worksheet.worksheet import Worksheet

parser = argparse.ArgumentParser(
    description='Combine the in-silico result tables into one workbook (Supplementary Table T4).'
)
parser.add_argument('--path_to_pr_auc_summary_table_excel', type=str, required=True)
parser.add_argument('--path_to_averaging_mode_comparison_table_excel', type=str, required=True)
parser.add_argument('--path_to_statistical_test_ranking_table_excel', type=str, required=True)
parser.add_argument('--path_to_pr_auc_vs_n_donors_table_excel', type=str, required=True)
parser.add_argument('--path_to_precision_recall_vs_n_donors_table_excel', type=str, required=True)
parser.add_argument('--path_to_simulation_summary_table_excel', type=str, required=True)
args = parser.parse_args()

PROPORTION_SHEETS: list[str] = ['8_OutOf_80', '24_OutOf_80', '40_OutOf_80', '64_OutOf_80', '80_OutOf_80']


def copy_sheet(source: Worksheet, target: Worksheet, sheet_renames: dict[str, str]):
    """
    Copy values, formulas, styles, merged cells and dimensions of one worksheet into another. Formulas that refer to
    other sheets of the source workbook are rewritten to the new sheet names.
    """
    def rename_references(formula: str) -> str:
        for old_name, new_name in sheet_renames.items():
            formula = formula.replace(f"'{old_name}'!", f"'{new_name}'!")
        return formula

    for row in source.iter_rows():
        for cell in row:
            value = cell.value
            if isinstance(value, ArrayFormula):
                value = ArrayFormula(ref=value.ref, text=rename_references(value.text))
            elif isinstance(value, str) and value.startswith('='):
                value = rename_references(value)

            target_cell = target.cell(row=cell.row, column=cell.column, value=value)
            if cell.has_style:
                target_cell.font = copy(cell.font)
                target_cell.fill = copy(cell.fill)
                target_cell.border = copy(cell.border)
                target_cell.alignment = copy(cell.alignment)
                target_cell.number_format = cell.number_format
                target_cell.protection = copy(cell.protection)

    for merged_range in source.merged_cells.ranges:
        target.merge_cells(str(merged_range))
    for key, dimension in source.column_dimensions.items():
        target.column_dimensions[key].width = dimension.width
    for key, dimension in source.row_dimensions.items():
        target.row_dimensions[key].height = dimension.height
    target.freeze_panes = source.freeze_panes


def main():
    # (source workbook, source sheet, sheet name in T4), in the order of the T4 workbook
    sheets: list[tuple[str, str, str]] = [
        (args.path_to_pr_auc_summary_table_excel, 'Sheet1', 'PR AUC'),
        (args.path_to_averaging_mode_comparison_table_excel, 'Median PR AUC', 'Averaging – Median PR AUC'),
        (args.path_to_averaging_mode_comparison_table_excel, 'Pairwise comparisons', 'Averaging – Pairwise'),
    ] + [
        (args.path_to_statistical_test_ranking_table_excel, sheet, f"Test Ranking – {sheet.split('_')[0]} of 80")
        for sheet in PROPORTION_SHEETS
    ] + [
        (args.path_to_statistical_test_ranking_table_excel, 'Pairwise comparisons', 'Test Ranking – Pairwise'),
        (args.path_to_pr_auc_vs_n_donors_table_excel, 'Sheet1', 'Donor Sweep – PR AUC'),
        (args.path_to_precision_recall_vs_n_donors_table_excel, 'Summary', 'Donor Sweep – Precision&Recall'),
        (args.path_to_precision_recall_vs_n_donors_table_excel, 'Per dataset', 'Donor Sweep – Per Dataset'),
    ]

    # Cross-sheet references only occur within the ranking workbook (the mean rank summary averages its sheets).
    sheet_renames: dict[str, str] = {
        source_sheet: target_sheet
        for path, source_sheet, target_sheet in sheets
        if path == args.path_to_statistical_test_ranking_table_excel
    }

    summary_workbook = openpyxl.Workbook()
    summary_workbook.remove(summary_workbook.active)

    source_workbooks: dict[str, openpyxl.Workbook] = {}
    for path, source_sheet, target_sheet in sheets:
        if path not in source_workbooks:
            source_workbooks[path] = openpyxl.load_workbook(path)
        copy_sheet(
            source=source_workbooks[path][source_sheet],
            target=summary_workbook.create_sheet(target_sheet),
            sheet_renames=sheet_renames,
        )

    os.makedirs(os.path.dirname(args.path_to_simulation_summary_table_excel), exist_ok=True)
    summary_workbook.save(args.path_to_simulation_summary_table_excel)


if __name__ == '__main__':
    main()
