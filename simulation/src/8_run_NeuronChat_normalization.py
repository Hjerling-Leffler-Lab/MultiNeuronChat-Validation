import os

import numpy as np

from tqdm import tqdm

import loompy
from multineuronchat.MultiNeuronChat import InteractionDB
from multineuronchat.normalize import cell_wise_log_normalization
from multineuronchat.utils import filter_genes

import argparse

parser = argparse.ArgumentParser(description='Normalize NeuronChat datasets.')
parser.add_argument(
    '--input_loom',
    type=str,
    help="Input loom file path.",
)
parser.add_argument(
    '--cell_wise_log_normalized_loom',
    type=str,
    help="Cell-wise log normalized loom file path.",
)
parser.add_argument(
    '--gene_filtered_loom',
    type=str,
    help="Gene filtered loom file path.",
)
parser.add_argument(
    '--output_loom',
    type=str,
    help="Output loom file path.",
)
parser.add_argument(
    '--chunk_size',
    type=int,
    default=2048,
    help="Chunk size for processing the loom file.",
)

args = parser.parse_args()
input_loom = args.input_loom
cell_wise_log_normalized_loom = args.cell_wise_log_normalized_loom
gene_filtered_loom = args.gene_filtered_loom
output_loom = args.output_loom
chunk_size = args.chunk_size

def max_normalization(
        path_to_input_loom: str,
        path_to_output_loom: str,
        chunk_size: int = -1,
        verbose: bool = False,
):
    with loompy.connect(path_to_input_loom, mode='r') as input_loom:
        max_value: float = -np.inf

        n_rows, n_cols = input_loom.shape

        # if the chunk_size is set to zero or a negative number, we load the complete matrix into memory
        if chunk_size <= 0:
            chunk_size = n_cols

        # Search for maximum value
        for i in tqdm(range(0, n_cols, chunk_size), disable=(not verbose), desc='Finding maximum value'):
            end_i: int = min(i + chunk_size, n_cols)

            data_view = input_loom.view[:, i:end_i]

            max_value = max(np.max(data_view[:,:]), max_value)

        print(f'Maximum value found: {max_value}')

        # Max normalize the dataset
        with loompy.new(path_to_output_loom) as output_loom:
            for i in tqdm(range(0, n_cols, chunk_size), disable=(not verbose), desc='Max normalizing'):
                end_i: int = min(i + chunk_size, n_cols)

                data_view = input_loom.view[:, i:end_i]

                if i == 0:
                    output_loom.add_columns(
                        data_view[:, :] / max_value,
                        row_attrs=data_view.row_attrs,
                        col_attrs=data_view.col_attrs,
                    )
                else:
                    output_loom.add_columns(
                        data_view[:, :] / max_value,
                        col_attrs=data_view.col_attrs,
                    )

def main():
    human_interaction_db: InteractionDB = InteractionDB(db='human_extended')
    gene_set: set[str] = human_interaction_db.get_set_of_genes()

    # Perform cell-wise log normalization on the loom file
    # This file is temporary and will be removed after the max normalization that is needed for MultiNeuronChat
    cell_wise_log_normalization(
        path_to_loom=input_loom,
        path_to_normalized_loom=cell_wise_log_normalized_loom,
        chunk_size=chunk_size,
        verbose=True
    )

    # First perform the gene filtering
    filter_genes(
        path_to_loom=cell_wise_log_normalized_loom,
        set_of_genes=gene_set,
        path_to_filtered_loom=gene_filtered_loom,
        chunk_size=chunk_size,
        verbose=True
    )

    # Max normalize the dataset to the correct folder
    max_normalization(
        path_to_input_loom=gene_filtered_loom,
        path_to_output_loom=output_loom,
        chunk_size=chunk_size,
        verbose=True
    )

    # Remove temporary files
    os.remove(cell_wise_log_normalized_loom)
    os.remove(gene_filtered_loom)

if __name__ == '__main__':
    main()