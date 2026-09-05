import numpy as np
import pandas as pd

from multineuronchat import InteractionDB, InteractionDBRow

from typing import *

import argparse

parser = argparse.ArgumentParser()

# path_to_perturbations_to_perform: str = './output/simulation/perturbations_to_perform.csv'
# path_to_gene_perturbations_to_perform: str = './output/simulation/gene_perturbations_to_perform.csv'

parser.add_argument(
    '--path_to_perturbations_to_perform',
    type=str,
    required=True,
    help='Path to the CSV file containing perturbations to perform.'
)
parser.add_argument(
    '--path_to_gene_perturbations_to_perform',
    type=str,
    required=True,
    help='Path to save the CSV file containing gene perturbations to perform.'
)
args = parser.parse_args()

path_to_perturbations_to_perform: str = args.path_to_perturbations_to_perform
path_to_gene_perturbations_to_perform: str = args.path_to_gene_perturbations_to_perform

def extract_db_information(db: InteractionDB):
    unique_ligands: Set[str] = set()
    unique_targets: Set[str] = set()

    ligands_to_targets: Dict[str, List[str]] = {}
    targets_to_ligands: Dict[str, List[str]] = {}

    ligands_to_genes_and_groups: Dict[str, Tuple[List[str], List[int]]] = {}
    targets_to_genes_and_groups: Dict[str, Tuple[List[str], List[int]]] = {}

    for interaction_row in db:
        interaction_row: InteractionDBRow

        interaction_name: str = interaction_row.interaction_name
        ligand_name: str = interaction_name.split('_')[0]
        target_name: str = '_'.join(interaction_name.split('_')[1:])

        unique_ligands.add(ligand_name)
        unique_targets.add(target_name)

        if ligand_name not in ligands_to_targets:
            ligands_to_targets[ligand_name] = []

        if target_name not in targets_to_ligands:
            targets_to_ligands[target_name] = []

        ligands_to_targets[ligand_name].append(target_name)
        targets_to_ligands[target_name].append(ligand_name)

        if ligand_name not in ligands_to_genes_and_groups:
            ligands_to_genes_and_groups[ligand_name] = (interaction_row.ligand_contributor, interaction_row.ligand_contributor_group)

        if target_name not in targets_to_genes_and_groups:
            targets_to_genes_and_groups[target_name] = (interaction_row.target_subunit, interaction_row.target_subunit_group)

    return {
        'unique_ligands': unique_ligands,
        'unique_targets': unique_targets,
        'ligands_to_targets': ligands_to_targets,
        'targets_to_ligands': targets_to_ligands,
        'ligands_to_genes_and_groups': ligands_to_genes_and_groups,
        'targets_to_genes_and_groups': targets_to_genes_and_groups
    }


def main():
    np.random.seed(1_299_709)

    perturbations_to_perform: pd.DataFrame = pd.read_csv(path_to_perturbations_to_perform)

    db: InteractionDB = InteractionDB('human_extended')

    db_information: Dict[str, Any] = extract_db_information(db)
    unique_ligands: Set[str] = db_information['unique_ligands']
    unique_targets: Set[str] = db_information['unique_targets']
    ligands_to_targets: Dict[str, List[str]] = db_information['ligands_to_targets']
    targets_to_ligands: Dict[str, List[str]] = db_information['targets_to_ligands']
    ligands_to_genes_and_groups: Dict[str, Tuple[List[str], List[int]]] = db_information['ligands_to_genes_and_groups']
    targets_to_genes_and_groups: Dict[str, Tuple[List[str], List[int]]] = db_information['targets_to_genes_and_groups']

    cell_type_to_lt_and_genes: Dict[str, Dict[str, Set[str]]] = {}

    # Find unique genes to perturb
    # For this we step through the perturbations to perform and check whether we have any overlapping perturbations,
    # i.e. target perturbations, which share genes in the same cell-type
    for i, perturbation_to_perform in perturbations_to_perform.iterrows():
        cell_type: str = perturbation_to_perform['Cell_Type']
        ligand_or_target: str = perturbation_to_perform['Ligand_or_Target']

        if ligand_or_target in unique_ligands:
            genes_to_perturb: Set[str] = set(ligands_to_genes_and_groups[ligand_or_target][0])
        elif ligand_or_target in unique_targets:
            genes_to_perturb: Set[str] = set(targets_to_genes_and_groups[ligand_or_target][0])
        else:
            raise ValueError(f'Unknown ligand or target: {ligand_or_target}')

        if cell_type not in cell_type_to_lt_and_genes:
            cell_type_to_lt_and_genes[cell_type] = {}

        if ligand_or_target not in cell_type_to_lt_and_genes[cell_type]:
            cell_type_to_lt_and_genes[cell_type][ligand_or_target] = genes_to_perturb

    gene_perturbations_to_perform: List[Tuple[str, str, str, bool]] = []

    # ------------------------------------------------------------------------------------
    # REPRODUCIBILITY WARNING — this section is NOT reproducible across runs.
    #
    # The random gene draw below (np.random.choice(list(unique_genes_i))) samples from a
    # Python `set` that has been converted to a `list`. The iteration/list order of a set of
    # strings depends on the interpreter's string hash seed (PYTHONHASHSEED), which is
    # randomized per process by default. As a result, `list(unique_genes_i)` is ordered
    # differently between runs, so np.random.choice picks a different gene even though numpy
    # is seeded above (np.random.seed(1_299_709)). This also shifts the subsequent
    # np.random.choice([True, False]) draw, so the entire gene_perturbations_to_perform.csv
    # is non-deterministic.
    #
    # This bug was found AFTER the manuscript was submitted. To keep the published results
    # intact, the committed data/simulation/gene_perturbations_to_perform.csv is treated as
    # the canonical ground truth and must NOT be regenerated. Accordingly, this script has
    # been removed from the Snakemake pipeline (the corresponding rule
    # `extract_genes_to_perturb_from_perturbations_to_perform` is commented out in
    # simulation/workflow/Snakefile). The rest of the pipeline is deterministic given that
    # committed CSV. The code is kept here, unchanged, for provenance only.
    # ------------------------------------------------------------------------------------
    for cell_type in cell_type_to_lt_and_genes.keys():
        ligands_or_targets_to_perturb: List[str] = list(cell_type_to_lt_and_genes[cell_type].keys())

        for i in range(len(ligands_or_targets_to_perturb)):
            genes_i = cell_type_to_lt_and_genes[cell_type][ligands_or_targets_to_perturb[i]]
            has_overlapping_genes: bool = False

            unique_genes_i = genes_i

            for j in range(i+1, len(ligands_or_targets_to_perturb)):
                gene_j = cell_type_to_lt_and_genes[cell_type][ligands_or_targets_to_perturb[j]]

                # Check overlap
                if len(genes_i.intersection(gene_j)) > 0:
                    has_overlapping_genes = True

                    print(f'Overlap in cell-type {cell_type} between {ligands_or_targets_to_perturb[i]} and {ligands_or_targets_to_perturb[j]}')
                    print(f'Genes: {genes_i.intersection(gene_j)}')
                    print(f'Unique genes in {ligands_or_targets_to_perturb[i]}: {genes_i - gene_j}')
                    print(f'Unique genes in {ligands_or_targets_to_perturb[j]}: {gene_j - genes_i}')
                    print('')

                    unique_genes_i = genes_i - gene_j

            if len(unique_genes_i) == 0:
                raise ValueError('No unique genes to perturb')
            # NON-REPRODUCIBLE: list(<set>) order is hash-seed dependent — see warning above.
            gene_to_perturb = np.random.choice(list(unique_genes_i))
            up_down = np.random.choice([True, False])
            gene_perturbations_to_perform.append((cell_type, ligands_or_targets_to_perturb[i], gene_to_perturb, up_down))


    with open(path_to_gene_perturbations_to_perform, 'w') as f:
        f.write('Cell_Type,Ligand_or_Target,Gene,Up_or_Down\n')
        for gene_perturbation_to_perform in gene_perturbations_to_perform:
            f.write(','.join([str(x) for x in gene_perturbation_to_perform]) + '\n')

    print(f'Saved gene perturbations to perform to {path_to_gene_perturbations_to_perform}')


if __name__ == '__main__':
    main()
