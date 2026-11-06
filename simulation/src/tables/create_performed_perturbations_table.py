import os

import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path_to_gene_perturbations_csv",
    type=str,
    help="Path to the CSV file containing the performed gene perturbations",
)
parser.add_argument(
    "--path_to_expected_perturbations_csv",
    type=str,
    help="Path to the CSV file containing the expected perturbed pathways (the ground truth)",
)
parser.add_argument(
    '--path_to_gene_perturbations_excel',
    type=str,
    help="Path to save the performed perturbations table excel",
)


args = parser.parse_args()
path_to_gene_perturbations_csv = args.path_to_gene_perturbations_csv
path_to_expected_perturbations_csv = args.path_to_expected_perturbations_csv
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

    # The ground truth: every source-receiver-ligand-target tetrad that contains a perturbed gene. The "Origin" column
    # (original sampling vs. added at gene level) is internal bookkeeping and left out.
    expected_perturbations_df = pd.read_csv(path_to_expected_perturbations_csv)
    expected_perturbations_df = expected_perturbations_df[['Source', 'Receiver', 'Ligand', 'Target', 'Perturbed_Gene']]

    with pd.ExcelWriter(path_to_gene_perturbations_excel) as writer:
        performed_gene_perturbations_df.to_excel(writer, sheet_name='Gene Perturbations', index=False)
        expected_perturbations_df.to_excel(writer, sheet_name='Expected Perturbations', index=False)


if __name__ == '__main__':
    main()