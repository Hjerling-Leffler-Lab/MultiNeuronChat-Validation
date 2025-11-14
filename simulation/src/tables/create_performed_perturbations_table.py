import os

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_to_gene_perturbations_csv",
    type=str,
    help="Path to the CSV file containing expected perturbations",
)
parser.add_argument(
    '--path_to_gene_perturbations_excel',
    type=str,
    help="Path to save the performed perturbations table excel",
)


args = parser.parse_args()
path_to_gene_perturbations_csv = args.path_to_gene_perturbations_csv
path_to_gene_perturbations_excel = args.path_to_gene_perturbations_excel

def main():
    # Create output folder if does not exist
    os.makedirs(os.path.dirname(path_to_gene_perturbations_excel), exist_ok=True)

    performed_gene_perturbations_df = pd.read_csv(path_to_gene_perturbations_csv)

    # Drop the "Ligand_or_Target" column
    performed_gene_perturbations_df = performed_gene_perturbations_df.drop(columns=["Ligand_or_Target"])

    # Rename columns
    # "Cell_Type" = "Cell Type"
    # "Gene" = "Gene"
    # "Up_or_Down" = "Up Regulated Or Down Regulated"

    performed_gene_perturbations_df = performed_gene_perturbations_df.rename(columns={
        "Cell_Type": "Cell Type",
        "Gene": "Gene",
        "Up_or_Down": "Up Regulated Or Down Regulated"
    })

    # Change all TRUE/FALSE to Up/Down
    performed_gene_perturbations_df["Up Regulated Or Down Regulated"] = performed_gene_perturbations_df["Up Regulated Or Down Regulated"].replace({
        True: "Up",
        False: "Down"
    })

    # Save the DataFrame to excel
    performed_gene_perturbations_df.to_excel(
        path_to_gene_perturbations_excel,
        index=False
    )


if __name__ == '__main__':
    main()