import os.path

from multineuronchat import InteractionDB
from multineuronchat.normalize import cell_wise_log_normalization
from multineuronchat.utils import gene_filter_and_subject_wise_normalize_dataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_collapsed_duplicate_genes_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_cell_wise_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_subject_wise_normalized_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--chunk_size_to_use',
    type=int,
    default=4096,
)
args = parser.parse_args()

# Paths to the input files and output loom file
path_to_collapsed_duplicate_genes_loom: str = args.path_to_collapsed_duplicate_genes_loom
path_to_cell_wise_normalized_loom: str = args.path_to_cell_wise_normalized_loom
path_to_subject_wise_normalized_loom: str = args.path_to_subject_wise_normalized_loom
tmp_path: str = os.environ['PDC_TMP']
chunk_size_to_use: int = args.chunk_size_to_use
verbose: bool = True


def main():
    if not os.path.exists(path_to_collapsed_duplicate_genes_loom):
        raise FileNotFoundError(f'File not found: {path_to_collapsed_duplicate_genes_loom}')

    # Make output directories if they do not exist
    os.makedirs(os.path.dirname(path_to_cell_wise_normalized_loom), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_subject_wise_normalized_loom), exist_ok=True)

    # First perform cell-wise log normalization
    cell_wise_log_normalization(
        path_to_loom=path_to_collapsed_duplicate_genes_loom,
        path_to_normalized_loom=path_to_cell_wise_normalized_loom,
        chunk_size=chunk_size_to_use,
        verbose=verbose,
    )

    # Then perform subject-wise max normalization
    db = InteractionDB('human_extended')
    gene_set = db.get_set_of_genes()

    gene_filter_and_subject_wise_normalize_dataset(
        path_to_loom=path_to_cell_wise_normalized_loom,
        gene_set=gene_set,
        subject_label_column='individualID',
        path_to_normalized_loom=path_to_subject_wise_normalized_loom,
        chunk_size=chunk_size_to_use,
        tmp_path=tmp_path,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
