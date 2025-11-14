import os

import loompy

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_outdated_loom_file',
    type=str,
    required=True,
    help='Path to the outdated loom file that needs to be updated.',
)
parser.add_argument(
    '--path_to_updated_loom_file',
    type=str,
    required=True,
    help='Path to save the updated loom file with the new version.',
)
args = parser.parse_args()

path_to_outdated_loom_file: str = args.path_to_outdated_loom_file
path_to_updated_loom_file: str = args.path_to_updated_loom_file

def main():
    with loompy.connect(path_to_outdated_loom_file, 'r') as src:
        loompy.create(
            path_to_updated_loom_file,
            src[:, :],
            row_attrs=src.ra,
            col_attrs=src.ca
        )



if __name__ == '__main__':
    main()
