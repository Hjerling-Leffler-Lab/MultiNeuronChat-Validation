import os

import numpy as np

import loompy

from multineuronchat.loompy_utils import create_empty_loom_file

from typing import Dict

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_joined_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_collapsed_duplicate_genes_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--chunk_size',
    type=int,
    default=2048,
)

args = parser.parse_args()

# Paths to the input loom file and output loom file
path_to_joined_loom: str = args.path_to_joined_loom
path_to_collapsed_duplicate_genes_loom: str = args.path_to_collapsed_duplicate_genes_loom

chunk_size_to_use: int = args.chunk_size


def join_duplicate_genes(
    path_in: str,
    path_out: str,
    chunk_size: int = 2048,
    dtype=np.float32,
    gene_key: str = "Gene",
    carry_other_row_attrs: bool = True,   # take first occurrence for other row attrs
):
    """
    Collapse duplicate genes by summing their rows.
    - Rows are grouped by ds.ra['Gene'].
    - For each column-chunk, we sum duplicates with np.add.reduceat and write to dst[:, i:end].
    - Row order in output follows np.unique(sorted_genes).

    Notes:
      * dtype is enforced (float32 recommended).
      * If carry_other_row_attrs=True, other row attrs are taken from the first row
        of each gene group (deterministic).
    """
    with loompy.connect(path_in, "r") as src:
        n_rows, n_cols = src.shape
        genes = src.ra[gene_key].astype(object)

        # ----- group rows once -----
        order = np.argsort(genes, kind="mergesort")
        genes_sorted = genes[order]
        uniq, idx_start, counts = np.unique(
            genes_sorted, return_index=True, return_counts=True
        )
        n_rows_out = uniq.shape[0]

        # ----- build output row attrs -----
        new_ra: Dict[str, np.ndarray] = {gene_key: uniq}
        if carry_other_row_attrs:
            for k, v in src.ra.items():
                if k == gene_key:
                    continue
                # take first occurrence per gene group, preserving stable order
                vv = v[order]                       # reorder like genes_sorted
                new_ra[k] = vv[idx_start]           # pick first row of each group

        # ----- preallocate output loom and write attrs once -----
        create_empty_loom_file(
            path_out,
            shape=(n_rows_out, n_cols),
            dtype_to_use=dtype,
            row_attrs=new_ra,
            col_attrs=src.ca
        )

        # ----- stream columns and write collapsed blocks -----
        with loompy.connect(path_out, "r+") as dst:
            for i in range(0, n_cols, chunk_size):
                end = min(i + chunk_size, n_cols)

                # read this chunk in grouped row order
                X = src[:, i:end].astype(dtype, copy=False)  # (n_rows, chunk)
                X = X[order, :]

                # collapse duplicates in one vectorized op
                # X has shape (n_rows, chunk); reduceat sums runs starting at idx_start
                collapsed = np.add.reduceat(X, idx_start, axis=0)  # (n_rows_out, chunk)

                # write directly into destination slice
                dst[:, i:end] = collapsed


def main():
    os.makedirs(os.path.dirname(path_to_joined_loom), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_collapsed_duplicate_genes_loom), exist_ok=True)

    join_duplicate_genes(
        path_in=path_to_joined_loom,
        path_out=path_to_collapsed_duplicate_genes_loom,
        chunk_size=chunk_size_to_use,
        dtype=np.float32,
        gene_key='Gene',
        carry_other_row_attrs=True,  # take first occurrence for other row attrs
    )


if __name__ == '__main__':
    main()
