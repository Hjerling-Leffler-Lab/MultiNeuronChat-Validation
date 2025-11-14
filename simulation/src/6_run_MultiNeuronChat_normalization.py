import os
import sys

import tempfile

from multineuronchat import InteractionDB
from multineuronchat.normalize import cell_wise_log_normalization
from multineuronchat.utils import gene_filter_and_subject_wise_normalize_dataset

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path_to_dataset_to_normalize',
    type=str,
    required=True,
    help='Path to the combined datasets folder'
)
parser.add_argument(
    '--path_to_cell_wise_log_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_subject_wise_max_normalized_loom',
    type=str,
    required=True,
    help='Path to the folder where the normalized output should be saved'
)
parser.add_argument(
    '--chunk_size',
    type=int,
    default=2 ** 14,
)

args = parser.parse_args()

path_to_dataset_to_normalize: str = args.path_to_dataset_to_normalize

path_to_cell_wise_log_normalized_loom: str = args.path_to_cell_wise_log_normalized_loom
path_to_subject_wise_max_normalized_loom: str = args.path_to_subject_wise_max_normalized_loom

chunk_size: int = args.chunk_size

def main():
    human_interaction_db: InteractionDB = InteractionDB(db='human_extended')
    gene_set: set[str] = human_interaction_db.get_set_of_genes()

    path_to_cell_wise_log_normalized_folder: str = os.path.dirname(path_to_cell_wise_log_normalized_loom)
    path_to_normalized_output_folder: str = os.path.dirname(path_to_subject_wise_max_normalized_loom)

    os.makedirs(path_to_cell_wise_log_normalized_folder, exist_ok=True)
    os.makedirs(path_to_normalized_output_folder, exist_ok=True)

    cell_wise_log_normalization(
        path_to_loom=path_to_dataset_to_normalize,
        path_to_normalized_loom=path_to_cell_wise_log_normalized_loom,
        chunk_size=chunk_size,
        verbose=True
    )

    # Perform gene filtering and subject-wise normalization on the cell-wise log normalized loom file
    gene_filter_and_subject_wise_normalize_dataset(
        path_to_loom=path_to_cell_wise_log_normalized_loom,
        gene_set=gene_set,
        subject_label_column='Donor',
        path_to_normalized_loom=path_to_subject_wise_max_normalized_loom,
        chunk_size=chunk_size,
        verbose=True
    )


if __name__ == '__main__':
    main()
