import os

import numpy as np
import pandas as pd

from multineuronchat import MultiNeuronChatObject

import argparse

parser = argparse.ArgumentParser(description='Export the tetrad test results and the per-donor values of the significant tetrads as source data.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_source_data_xlsx',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_donor_id_mapping_csv',
    type=str,
    required=True,
    help='Mapping of the pseudonymised donor labels to ROSMAP IDs. Controlled-access information, do not publish.'
)
parser.add_argument(
    '--statistical_tests',
    type=str,
    nargs='+',
    required=True,
)
parser.add_argument(
    '--significance_thresholds',
    type=float,
    nargs='+',
    required=True,
    help='BY-adjusted p-value threshold per statistical test, in the same order as --statistical_tests.'
)
parser.add_argument(
    '--random_seed',
    type=int,
    default=0,
    help='Seed for shuffling donors before labelling, so the label order does not follow the ROSMAP IDs.'
)
args = parser.parse_args()

path_to_mnc_object: str = args.path_to_mnc_object
path_to_source_data_xlsx: str = args.path_to_source_data_xlsx
path_to_donor_id_mapping_csv: str = args.path_to_donor_id_mapping_csv

statistical_tests: list[str] = args.statistical_tests
significance_thresholds: list[float] = args.significance_thresholds
random_seed: int = args.random_seed

if len(statistical_tests) != len(significance_thresholds):
    raise ValueError('Provide one significance threshold per statistical test.')
threshold_per_test: dict[str, float] = dict(zip(statistical_tests, significance_thresholds))

tetrad_columns: list[str] = ['source', 'receiver', 'interaction']


def get_donor_mapping(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    """Assign each donor a label Donor_001, Donor_002, ... in a seeded random order."""
    donors = pd.DataFrame([
        {'rosmap_id': subject, 'condition': condition}
        for condition in mnc_object.condition_names
        for subject in mnc_object.communication_score_dict[condition].subject.values
    ])
    donors = donors.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    n_digits = max(3, len(str(len(donors))))
    donors.insert(0, 'donor', [f'Donor_{i + 1:0{n_digits}d}' for i in range(len(donors))])
    return donors


def get_tetrad_results(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    """One row per tested tetrad: median score and number of donors per condition, statistic, p-value and adjusted p-value per test."""
    columns = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition]
        columns.append(scores.median('subject', skipna=True).to_dataframe(f'median communication score {condition}'))
        columns.append(scores.notnull().sum('subject').to_dataframe(f'n donors {condition}'))
    for test in statistical_tests:
        columns.append(mnc_object.statistics[test]['statistics'].to_dataframe(f'{test} statistic'))
        columns.append(mnc_object.p_values[test].to_dataframe(f'{test} p-value'))
        columns.append(mnc_object.p_values_adj[test].to_dataframe(f'{test} adj. p-value (BY)'))
    results = pd.concat([column.iloc[:, -1] for column in columns], axis=1)

    p_value_columns = [f'{test} p-value' for test in statistical_tests]
    results = results.dropna(subset=p_value_columns, how='all')
    for test in statistical_tests:
        results[f'{test} significant'] = results[f'{test} adj. p-value (BY)'] < threshold_per_test[test]
    return results.reset_index().sort_values(tetrad_columns).reset_index(drop=True)


def get_per_donor_values(mnc_object: MultiNeuronChatObject, tetrad_results: pd.DataFrame, donor_mapping: pd.DataFrame) -> pd.DataFrame:
    """One row per donor and tetrad significant in at least one test, with the ligand and target abundance and communication score."""
    significant_columns = [f'{test} significant' for test in statistical_tests]
    significant = tetrad_results[tetrad_results[significant_columns].any(axis=1)]
    rosmap_id_to_donor = dict(zip(donor_mapping['rosmap_id'], donor_mapping['donor']))

    rows = []
    for tetrad in significant.to_dict('records'):
        significant_in = ', '.join(test for test in statistical_tests if tetrad[f'{test} significant'])
        for condition in mnc_object.condition_names:
            ligand = mnc_object.ligand_abundance_dict[condition].loc[:, tetrad['source'], tetrad['interaction']]
            target = mnc_object.target_abundance_dict[condition].loc[:, tetrad['receiver'], tetrad['interaction']]
            score = mnc_object.communication_score_dict[condition].loc[:, tetrad['source'], tetrad['receiver'], tetrad['interaction']]
            rows.append(pd.DataFrame({
                'source': tetrad['source'],
                'receiver': tetrad['receiver'],
                'interaction': tetrad['interaction'],
                'significant in': significant_in,
                'donor': [rosmap_id_to_donor[subject] for subject in score.subject.values],
                'condition': condition,
                'ligand abundance (source)': ligand.to_numpy(),
                'target abundance (receiver)': target.to_numpy(),
                'communication score': score.to_numpy(),
            }))
    if not rows:
        return pd.DataFrame(columns=tetrad_columns + ['significant in', 'donor', 'condition', 'ligand abundance (source)', 'target abundance (receiver)', 'communication score'])

    per_donor = pd.concat(rows, ignore_index=True)
    # Donors without enough cells of the source or receiver type have no score for this tetrad
    per_donor = per_donor.dropna(subset=['communication score'])
    return per_donor.sort_values(tetrad_columns + ['condition', 'donor']).reset_index(drop=True)


def get_readme(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    thresholds = ', '.join(f'{test}: adj. p < {threshold}' for test, threshold in threshold_per_test.items())
    rows = [
        ('Tetrad results', '', 'One row per tested tetrad (source cell type, receiver cell type, ligand-target interaction).'),
        ('Tetrad results', 'median communication score <condition>', 'Median communication score across the donors of the condition.'),
        ('Tetrad results', 'n donors <condition>', 'Number of donors of the condition with a communication score for the tetrad.'),
        ('Tetrad results', '<test> statistic', 'Two-sample test statistic comparing the donor-level communication scores of the two conditions.'),
        ('Tetrad results', '<test> p-value', 'Unadjusted p-value.'),
        ('Tetrad results', '<test> adj. p-value (BY)', 'Benjamini-Yekutieli adjusted p-value across all tested tetrads.'),
        ('Tetrad results', '<test> significant', f'Adjusted p-value below the threshold ({thresholds}).'),
        ('Per-donor values', '', 'One row per donor and tetrad that is significant in at least one test.'),
        ('Per-donor values', 'significant in', 'Tests in which the tetrad is significant.'),
        ('Per-donor values', 'donor', 'Pseudonymised donor label.'),
        ('Per-donor values', 'condition', f'{mnc_object.condition_names[0]} or {mnc_object.condition_names[1]}.'),
        ('Per-donor values', 'ligand abundance (source)', 'Ligand abundance of the interaction in the source cell type of the donor.'),
        ('Per-donor values', 'target abundance (receiver)', 'Target abundance of the interaction in the receiver cell type of the donor.'),
        ('Per-donor values', 'communication score', 'Communication score of the tetrad in the donor.'),
    ]
    return pd.DataFrame(rows, columns=['sheet', 'column', 'description'])


def main():
    for output_path in [path_to_source_data_xlsx, path_to_donor_id_mapping_csv]:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(path_to_mnc_object)

    donor_mapping: pd.DataFrame = get_donor_mapping(mnc_object)
    tetrad_results: pd.DataFrame = get_tetrad_results(mnc_object)
    per_donor_values: pd.DataFrame = get_per_donor_values(mnc_object, tetrad_results, donor_mapping)

    with pd.ExcelWriter(path_to_source_data_xlsx) as writer:
        get_readme(mnc_object).to_excel(writer, sheet_name='README', index=False)
        tetrad_results.to_excel(writer, sheet_name='Tetrad results', index=False)
        per_donor_values.to_excel(writer, sheet_name='Per-donor values', index=False)

    donor_mapping.sort_values('donor').to_csv(path_to_donor_id_mapping_csv, index=False)


if __name__ == '__main__':
    main()
