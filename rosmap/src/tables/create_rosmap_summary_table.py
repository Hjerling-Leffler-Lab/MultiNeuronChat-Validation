"""Excel summary of the ROSMAP MNC results: every tested tetrad with its p-values and per-condition score summaries.

All values are aggregated over the donors of a condition (no donor IDs, no per-donor scores), so the workbook can be
exported from Bianca. Tetrads are tested and selected exactly as in the figures (BH over all tested tetrads, recomputed
from the raw p; see plotting.utils.get_adjusted_p_values).

Sheets:
    README               test, correction, threshold, cohort and column descriptions
    All tetrads          one row per tested tetrad, sorted by BH-adjusted p
    Significant tetrads  the tetrads with BH-adjusted p < threshold
    Ligand target tests  NCI vs AD tests of the ligand and target abundance behind the significant tetrads
                         (copied from e_plot_mnc_coarse_ligand_target_decomposition.py)
"""
import os
import sys
import argparse
from datetime import date

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from plotting.utils import get_adjusted_p_values

parser = argparse.ArgumentParser(description='Excel summary of all tested ROSMAP tetrads.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_ligand_target_tests_csv',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_table_excel',
    type=str,
    required=True,
)
parser.add_argument(
    '--statistical_test',
    type=str,
    required=True,
)
parser.add_argument(
    '--significance_threshold',
    type=float,
    default=0.05,
)
parser.add_argument(
    '--cell_type_annotation',
    type=str,
    default='grouping.by',
    help='Cell-type annotation of the MNC run, only written to the README sheet.'
)
args = parser.parse_args()

statistical_test_names: dict[str, str] = {
    'KS': 'two-sample Kolmogorov-Smirnov',
    'CVM': 'two-sample Cramér-von Mises',
    'MannWhitneyU': 'two-sided Mann-Whitney U',
}


def score_summary(mnc_object: MultiNeuronChatObject, condition: str, tetrads: pd.DataFrame) -> pd.DataFrame:
    """Number of donors with a score, median, mean and % of donors with a score of exactly 0, per tetrad."""
    scores = mnc_object.communication_score_dict[condition]
    rows = []
    for tetrad in tetrads.itertuples():
        values = scores.loc[:, tetrad.source, tetrad.receiver, tetrad.interaction].to_numpy()
        values = values[~np.isnan(values)]
        rows.append({
            f'N donors {condition}': len(values),
            f'Median score {condition}': np.median(values) if len(values) else np.nan,
            f'Mean score {condition}': np.mean(values) if len(values) else np.nan,
            f'% zero scores {condition}': 100 * np.mean(values == 0) if len(values) else np.nan,
        })
    return pd.DataFrame(rows, index=tetrads.index)


def readme(mnc_object: MultiNeuronChatObject, n_tested: int, n_significant: int) -> pd.DataFrame:
    n_donors = {condition: mnc_object.communication_score_dict[condition].sizes['subject'] for condition in mnc_object.condition_names}
    entries = [
        ('Data', 'ROSMAP snRNA-seq, dorsolateral prefrontal cortex'),
        ('Conditions', ' vs '.join(f'{condition} (n = {n} donors)' for condition, n in n_donors.items())),
        ('Cell-type annotation', args.cell_type_annotation),
        ('Communication score', 'tri-mean per donor'),
        ('Statistical test', f'{args.statistical_test} ({statistical_test_names.get(args.statistical_test, args.statistical_test)}), per tetrad, NCI vs AD'),
        ('Multiple-testing correction', f'Benjamini-Hochberg over all {n_tested} tested tetrads'),
        ('Significance threshold', f'BH-adjusted p < {args.significance_threshold} ({n_significant} tetrads)'),
        ('Prefilter', 'none'),
        ('Generated', date.today().isoformat()),
        ('', ''),
        ('Column', 'Description'),
        ('Source / Receiver', 'sending / receiving cell type'),
        ('Interaction', 'ligand_target pair as named in the MNC interaction database (human_extended)'),
        ('Raw p', f'{args.statistical_test} p-value of NCI vs AD scores'),
        ('BH-adjusted p', 'Benjamini-Hochberg adjusted p-value'),
        ('N donors <condition>', 'donors with a communication score for this tetrad'),
        ('Median / Mean score <condition>', 'median / mean communication score over these donors'),
        ('% zero scores <condition>', 'percentage of these donors with a score of exactly 0'),
    ]
    return pd.DataFrame(entries, columns=['Item', 'Value'])


def main():
    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(args.path_to_mnc_object)

    p_values = mnc_object.p_values[args.statistical_test].to_dataframe('Raw p')
    p_values_adj = get_adjusted_p_values(mnc_object, args.statistical_test).to_dataframe('BH-adjusted p')
    tetrads = p_values.join(p_values_adj).dropna().sort_values(['BH-adjusted p', 'Raw p']).reset_index()
    tetrads = tetrads[['source', 'receiver', 'interaction', 'Raw p', 'BH-adjusted p']]
    tetrads['Significant'] = tetrads['BH-adjusted p'] < args.significance_threshold

    summaries = [score_summary(mnc_object, condition, tetrads) for condition in mnc_object.condition_names]
    tetrads = pd.concat([tetrads, *summaries], axis=1)
    tetrads = tetrads.rename(columns={'source': 'Source', 'receiver': 'Receiver', 'interaction': 'Interaction'})

    significant = tetrads[tetrads['Significant']]
    print(f'{len(significant)} of {len(tetrads)} tested tetrads with {args.statistical_test} BH-adjusted p < {args.significance_threshold}')

    ligand_target_tests = pd.read_csv(args.path_to_ligand_target_tests_csv).rename(columns={
        'side': 'Side', 'cell_type': 'Cell type', 'interaction': 'Interaction', 'statistical_test': 'Test',
        'p': 'Raw p', 'p_adj': 'BH-adjusted p (over these tests)',
    })

    os.makedirs(os.path.dirname(args.path_to_table_excel), exist_ok=True)
    with pd.ExcelWriter(args.path_to_table_excel) as writer:
        readme(mnc_object, len(tetrads), len(significant)).to_excel(writer, sheet_name='README', index=False)
        tetrads.to_excel(writer, sheet_name='All tetrads', index=False)
        significant.to_excel(writer, sheet_name='Significant tetrads', index=False)
        ligand_target_tests.to_excel(writer, sheet_name='Ligand target tests', index=False)


if __name__ == '__main__':
    main()
