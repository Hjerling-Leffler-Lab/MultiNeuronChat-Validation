import loompy

import numpy as np
from tqdm import tqdm

from multineuronchat.loompy_utils import create_empty_loom_file

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_joined_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_subsampled_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--chunk_size',
    type=int,
    default=4096,
)
args = parser.parse_args()

# Paths to the input loom file and output loom file
path_to_joined_loom: str = args.path_to_joined_loom
path_to_subsampled_loom: str = args.path_to_subsampled_loom

chunk_size: int = args.chunk_size

def main():

    nci_id: int = 1
    ad_id: int = 4
    diagnosis_column_name: str = 'cogdx'

    num_to_label: dict = {
        nci_id: 'NCI',
        ad_id: 'AD'
    }

    with loompy.connect(path_to_joined_loom, 'r') as src:
        n_rows, n_cols = src.shape

        # Create a list of indices that should be kept
        indices_to_keep: np.ndarray = np.where(
            (src.ca[diagnosis_column_name] == nci_id) | (src.ca[diagnosis_column_name] == ad_id)
        )[0]

        n_cols_to_keep: int = len(indices_to_keep)

        row_attrs = {k: src.ra[k] for k in src.ra.keys()}
        # get the column attributes that should be kept
        col_attrs = {k: src.ca[k][indices_to_keep] for k in src.col_attrs.keys()}
        # Add the diagnosis column with the labels
        col_attrs[diagnosis_column_name] = np.array([num_to_label[x] for x in src.ca[diagnosis_column_name][indices_to_keep]])

        # Create a new loom file with the same number of rows and the filtered columns
        create_empty_loom_file(
            path_to_loom=path_to_subsampled_loom,
            shape=(n_rows, n_cols_to_keep),
            row_attrs=row_attrs,
            col_attrs=col_attrs,
            dtype_to_use=src[:, :1].dtype,  # Use the dtype of the first column to ensure consistency
        )

        with loompy.connect(path_to_subsampled_loom, 'r+') as dst:
            for i in tqdm(range(0, len(indices_to_keep), chunk_size), desc='Processing chunks'):
                end_i: int = min(i + chunk_size, len(indices_to_keep))

                dst[:, i:end_i] = src[:, indices_to_keep[i:end_i]]


if __name__ == '__main__':
    main()
