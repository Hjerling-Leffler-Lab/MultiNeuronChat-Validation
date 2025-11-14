import os

import numpy as np
import pandas as pd

import loompy

from multineuronchat.loompy_utils import create_empty_loom_file

from collections.abc import Sequence


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_case',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_non_perturbed_case_folder',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_ctrl_folder',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_patient_meta_output_file',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_output_loom',
    type=str,
    required=True,
)
parser.add_argument(
    '--proportion_of_affected_donors',
    type=float,
    required=True,
)
parser.add_argument(
    '--dataset_index',
    type=int,
    required=True,
)
parser.add_argument(
    '--n_samples_per_group_and_sex',
    type=int,
    required=True,
)
args = parser.parse_args()

path_to_case: str = args.path_to_case
path_to_non_perturbed_CASE_folder: str = args.path_to_non_perturbed_case_folder
path_to_ctrl_folder: str = args.path_to_ctrl_folder

path_to_output_loom: str = args.path_to_output_loom
path_to_patient_meta_output_file: str = args.path_to_patient_meta_output_file

proportion_of_affected_donors: float = args.proportion_of_affected_donors
dataset_index: int = args.dataset_index
n_samples_per_group_and_sex: int = args.n_samples_per_group_and_sex

np.random.seed(int(np.floor((42 * dataset_index + n_samples_per_group_and_sex)*proportion_of_affected_donors)))


def main():
    if not os.path.exists(path_to_ctrl_folder):
        raise FileNotFoundError(f'Path to CTRL folder does not exist: {path_to_ctrl_folder}')

    if not os.path.exists(path_to_non_perturbed_CASE_folder):
        raise FileNotFoundError(f'Path to non-perturbed CASE folder does not exist: {path_to_non_perturbed_CASE_folder}')

    if not os.path.exists(path_to_case):
        raise FileNotFoundError(f'Path to CASE folder does not exist: {path_to_case}')

    path_to_female_CTRL: str = os.path.join(path_to_ctrl_folder, 'f')
    path_to_male_CTRL: str = os.path.join(path_to_ctrl_folder, 'm')

    path_to_female_unperturbed_CASE: str = os.path.join(path_to_non_perturbed_CASE_folder, 'f')
    path_to_male_unperturbed_CASE: str = os.path.join(path_to_non_perturbed_CASE_folder, 'm')

    path_to_female_CASE: str = os.path.join(path_to_case, 'f')
    path_to_male_CASE: str = os.path.join(path_to_case, 'm')

    female_CTRL_samples: Sequence[str] = [f for f in os.listdir(path_to_female_CTRL) if f.endswith('.loom')]
    male_CTRL_samples: Sequence[str] = [f for f in os.listdir(path_to_male_CTRL) if f.endswith('.loom')]

    female_unperturbed_CASE_samples: Sequence[str] = [f for f in os.listdir(path_to_female_unperturbed_CASE) if f.endswith('.loom')]
    male_unperturbed_CASE_samples: Sequence[str] = [f for f in os.listdir(path_to_male_unperturbed_CASE) if f.endswith('.loom')]

    female_CASE_samples: Sequence[str] = [f for f in os.listdir(path_to_female_CASE) if f.endswith('.loom')]
    male_CASE_samples: Sequence[str] = [f for f in os.listdir(path_to_male_CASE) if f.endswith('.loom')]

    # Generate datasets
    proportion_of_unaffected_donors: float = 1.0 - proportion_of_affected_donors
    n_unaffected_donors: int = int(n_samples_per_group_and_sex * proportion_of_unaffected_donors)
    n_affected_donors: int = n_samples_per_group_and_sex - n_unaffected_donors

    # Sample CTRL group
    female_ctrl: list[str] = np.random.choice(female_CTRL_samples, n_samples_per_group_and_sex, replace=False).tolist()
    male_ctrl: list[str] = np.random.choice(male_CTRL_samples, n_samples_per_group_and_sex, replace=False).tolist()

    # Sample not perturbed CASE group
    female_unperturbed_case: list[str] = np.random.choice(female_unperturbed_CASE_samples, n_unaffected_donors, replace=False).tolist()
    male_unperturbed_case: list[str] = np.random.choice(male_unperturbed_CASE_samples, n_unaffected_donors, replace=False).tolist()

    # Sample perturbed CASE group
    female_case: list[str] = np.random.choice(female_CASE_samples, n_affected_donors, replace=False).tolist()
    male_case: list[str] = np.random.choice(male_CASE_samples, n_affected_donors, replace=False).tolist()

    df_dict: dict[str, list[str | bool]] = {
        'Sample': female_ctrl + male_ctrl + female_unperturbed_case + male_unperturbed_case + female_case + male_case,
        'Sex': (['F'] * len(female_ctrl)) + (['M'] * len(male_ctrl)) + (['F'] * len(female_unperturbed_case)) + (['M'] * len(male_unperturbed_case)) + (['F'] * len(female_case)) + (['M'] * len(male_case)),
        'Perturbed': (['CTRL'] * (len(female_ctrl) + len(male_ctrl))) + (['NP_CASE'] * (len(female_unperturbed_case) + len(male_unperturbed_case))) + (['CASE'] * (len(female_case) + len(male_case)))
    }
    df: pd.DataFrame = pd.DataFrame(df_dict)
    df.to_csv(path_to_patient_meta_output_file, index=False)

    all_paths: list[str] = (
            [os.path.join(path_to_female_CTRL, f) for f in female_ctrl] +
            [os.path.join(path_to_male_CTRL, f) for f in male_ctrl] +
            [os.path.join(path_to_female_unperturbed_CASE, f) for f in female_unperturbed_case] +
            [os.path.join(path_to_male_unperturbed_CASE, f) for f in male_unperturbed_case] +
            [os.path.join(path_to_female_CASE, f) for f in female_case] +
            [os.path.join(path_to_male_CASE, f) for f in male_case]
    )

    # Two runs:
    # 1. Get the shapes and combine all the row_attrs and col_attrs
    # 2. Insert the data correctly

    row_attrs: dict[str, list] = None
    col_attrs: dict[str, list] = {}

    n_genes: int = 0
    n_cells: int = 0

    # First run: get the shapes and combine all the row_attrs and col_attrs
    for path in all_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f'Path to loom file does not exist: {path}')

        with loompy.connect(path, mode='r') as src:
            n_rows, n_cols = src.shape

            if n_genes == 0: n_genes = n_rows
            n_cells += n_cols

            if row_attrs is None:
                row_attrs = {
                    key: src.ra[key] for key in src.ra.keys()
                }

            if not col_attrs:
                col_attrs = {
                    key: list(src.ca[key]) for key in src.ca.keys()
                }
            else:
                for key in src.ca.keys():
                    if key not in col_attrs:
                        raise ValueError(f'Column attribute {key} not found in all datasets')
                    col_attrs[key].extend(src.ca[key])

    # Create a new loom file with the combined row and column attributes
    create_empty_loom_file(
        path_to_loom=path_to_output_loom,
        shape=(n_genes, n_cells),
        row_attrs=row_attrs,
        col_attrs=col_attrs,
        dtype_to_use=np.float32
    )

    # Second run: insert the data correctly
    i = 0
    with loompy.connect(path_to_output_loom, mode='r+') as dst:
        for path in all_paths:
            with loompy.connect(path, mode='r') as src:
                n_cols = src.shape[1]

                # Assert that the order of genes is the same
                if not np.array_equal(src.ra['Gene'], dst.ra['Gene']):
                    raise ValueError(f'Genes in loom file {path} do not match')

                dst[:, i:i+n_cols] = src[:, :]
                i += n_cols


if __name__ == '__main__':
    main()
