import os

import numpy as np
import pandas as pd

import loompy

import scipy.io

import gzip

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_cell_annotations',
    type=str,
)
parser.add_argument(
    '--path_to_clinical_data',
    type=str,
)
parser.add_argument(
    '--path_to_matrices',
    type=str,
)
parser.add_argument(
    '--path_to_output_loom',
    type=str,
)
args = parser.parse_args()

# Paths to the input files and output loom file
path_to_cell_annotations: str = args.path_to_cell_annotations
path_to_clinical_data: str = args.path_to_clinical_data
path_to_matrices: str = args.path_to_matrices
path_to_output_loom: str = args.path_to_output_loom


def main():
    if not os.path.exists(path_to_cell_annotations):
        raise FileNotFoundError(f'File not found: {path_to_cell_annotations}')
    if not os.path.exists(path_to_clinical_data):
        raise FileNotFoundError(f'File not found: {path_to_clinical_data}')
    if not os.path.exists(path_to_matrices):
        raise FileNotFoundError(f'Directory not found: {path_to_matrices}')

    # Create folder for output loom file if it does not exist
    output_folder = os.path.dirname(path_to_output_loom)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    annotations: pd.DataFrame = pd.read_csv(path_to_cell_annotations)
    batches: pd.Series = annotations['batch'].unique()

    clinical_data: pd.DataFrame = pd.read_csv(path_to_clinical_data)

    # Exclude all MAP batches
    batches = batches[[not batch.startswith('MAP') for batch in batches]]

    test_features_dict: dict = {}

    with loompy.new(path_to_output_loom) as dsout:
        for i, batch in enumerate(batches):
            print(f'Processing batch {batch} ({i + 1}/{len(batches)})')

            path_to_features_gz: str = os.path.join(path_to_matrices, f'{batch}.features.tsv.gz')
            path_to_barcodes_gz: str = os.path.join(path_to_matrices, f'{batch}.barcodes.tsv.gz')
            path_to_matrix_gz: str = os.path.join(path_to_matrices, f'{batch}.matrix.mtx.gz')

            if not os.path.exists(path_to_features_gz):
                raise FileNotFoundError(f'File not found: {path_to_features_gz}')
            if not os.path.exists(path_to_barcodes_gz):
                raise FileNotFoundError(f'File not found: {path_to_barcodes_gz}')
            if not os.path.exists(path_to_matrix_gz):
                raise FileNotFoundError(f'File not found: {path_to_matrix_gz}')

            with gzip.open(path_to_features_gz, 'rt') as f:
                features: pd.DataFrame = pd.read_csv(f, sep='\t', header=None)
                features.columns = ['GeneID', 'GeneName', 'GeneType']

            with gzip.open(path_to_barcodes_gz, 'rt') as f:
                barcodes: pd.DataFrame = pd.read_csv(f, sep='\t', header=None)
                barcodes.columns = ['Barcode']

            # Create the cell annotations
            cell_ids = [f'{batch}_{x}' for x in barcodes['Barcode'].values]
            cell_annotations = annotations[annotations['cell'].isin(cell_ids)]

            cell_ids_present = cell_annotations['cell'].values

            cell_ids_present_idx = np.arange(len(cell_ids))
            cell_ids_filtered = list(filter(lambda x: x[0] in cell_ids_present, zip(cell_ids, cell_ids_present_idx)))
            cell_ids_order = [x[0] for x in cell_ids_filtered]
            cell_ids_order_idx = [x[1] for x in cell_ids_filtered]

            # Make sure that cell_ids and the cell_annotations have the same order
            cell_annotations = cell_annotations.set_index('cell').loc[cell_ids_order].reset_index()

            col_dict: dict[str, np.array] = {
                column: cell_annotations[column].values
                for column in cell_annotations.columns
            }
            # Remove columns: "Unnamed 0"
            col_dict.pop('Unnamed: 0')

            clinical_columns = [
                'projid', 'Study', 'msex', 'educ', 'race', 'spanish', 'apoe_genotype',
                'age_at_visit_max', 'age_first_ad_dx', 'age_death', 'cts_mmse30_first_ad_dx',
                'cts_mmse30_lv', 'pmi', 'braaksc', 'ceradsc', 'cogdx', 'dcfdx_lv'
            ]
            for column in clinical_columns:
                col_dict[column] = []

            # Add clinical data:
            for individual_id in col_dict['individualID']:
                clinical_data_row = clinical_data[clinical_data['individualID'] == individual_id]
                for column in clinical_columns:
                    col_dict[column].append(clinical_data_row[column].values[0])

            for column in clinical_columns:
                col_dict[column] = np.array(col_dict[column])

            # Check if the order of cell_ids is the same as the order of the cell annotations
            if np.array_equal(cell_ids, cell_annotations['cell'].values):
                print(f'Cell order is the same for batch {batch}')

            with gzip.open(path_to_matrix_gz, 'rt') as f:
                sparse_matrix: scipy.sparse.coo_matrix = scipy.io.mmread(f)

            dense_matrix = sparse_matrix.todense()[:, cell_ids_order_idx]

            if i == 0:
                features_dict: dict = {
                    'GeneID': features['GeneID'].values,
                    'Gene': features['GeneName'].values,
                    'GeneType': features['GeneType'].values
                }

                test_features_dict = features_dict

                dsout.add_columns(
                    layers=dense_matrix,
                    row_attrs=features_dict,
                    col_attrs=col_dict
                )
            else:
                # Check if the features are the same
                if not np.array_equal(test_features_dict['GeneID'], features['GeneID'].values):
                    raise ValueError(f'Features are not the same for batch {batch}')

                dsout.add_columns(
                    layers=dense_matrix,
                    col_attrs=col_dict
                )


if __name__ == '__main__':
    main()
