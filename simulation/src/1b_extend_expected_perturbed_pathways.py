import pandas as pd

from typing import *

import argparse

# The expected perturbed pathways written by 1_extract_ligand_target_production.R are defined per sampled ligand or
# target (e.g. GABA-A_a4b3g2S_complex in Inhibitory SST neurons). The simulation, however, perturbs a single gene of
# that ligand/target (gene_perturbations_to_perform.csv, e.g. GABRB3). If that gene is also a contributor/subunit of
# other ligands or targets (GABRB3 is a subunit of eleven GABA-A complexes), those change as well, but were missing from
# the ground truth. This script derives the expected perturbed pathways at the gene level and extends the original file
# with them. The original rows are always kept.

parser = argparse.ArgumentParser()
parser.add_argument(
    '--path_to_gene_perturbations_to_perform',
    type=str,
    required=True,
    help='Path to the CSV file with the performed gene perturbations (Cell_Type, Ligand_or_Target, Gene, Up_or_Down).'
)
parser.add_argument(
    '--path_to_interaction_db',
    type=str,
    required=True,
    help='Path to the interaction database TSV file (interactionDB_human_extended.tsv).'
)
parser.add_argument(
    '--path_to_expected_perturbed_pathways',
    type=str,
    required=True,
    help='Path to the original expected perturbed pathways CSV file written by 1_extract_ligand_target_production.R.'
)
parser.add_argument(
    '--path_to_producible_ligands',
    type=str,
    required=True,
    help='Path to a CSV file with a boolean matrix (rows: ligands, columns: cell types) of which cell types can '
         'produce which ligand.'
)
parser.add_argument(
    '--path_to_producible_targets',
    type=str,
    required=True,
    help='Path to a CSV file with a boolean matrix (rows: targets, columns: cell types) of which cell types can '
         'produce which target.'
)
parser.add_argument(
    '--path_to_output_expected_perturbed_pathways',
    type=str,
    required=True,
    help='Path to save the extended expected perturbed pathways CSV file.'
)
args = parser.parse_args()

path_to_gene_perturbations_to_perform: str = args.path_to_gene_perturbations_to_perform
path_to_interaction_db: str = args.path_to_interaction_db
path_to_expected_perturbed_pathways: str = args.path_to_expected_perturbed_pathways
path_to_producible_ligands: str = args.path_to_producible_ligands
path_to_producible_targets: str = args.path_to_producible_targets
path_to_output_expected_perturbed_pathways: str = args.path_to_output_expected_perturbed_pathways

PATHWAY_COLUMNS: List[str] = ['Source', 'Receiver', 'Ligand', 'Target']


def extract_db_information(path_to_db: str) -> Dict[str, Any]:
    """
    Parse the interaction database into ligand/target names, their partners and their genes. Ligand and target names
    are split from the interaction_name exactly as in 1_extract_ligand_target_production.R: the ligand is the part
    before the first '_', the target is the remainder (complex targets contain further '_').
    """
    db: pd.DataFrame = pd.read_csv(path_to_db, sep='\t')

    ligands_to_targets: Dict[str, List[str]] = {}
    targets_to_ligands: Dict[str, List[str]] = {}
    ligands_to_genes: Dict[str, Set[str]] = {}
    targets_to_genes: Dict[str, Set[str]] = {}

    for _, row in db.iterrows():
        interaction_name_split: List[str] = row['interaction_name'].split('_')
        ligand: str = interaction_name_split[0]
        target: str = '_'.join(interaction_name_split[1:])

        ligands_to_targets.setdefault(ligand, []).append(target)
        targets_to_ligands.setdefault(target, []).append(ligand)

        # As in the R script, the genes of a ligand/target are taken from its first occurrence in the database
        ligands_to_genes.setdefault(ligand, set(str(row['lig_contributor']).split('-')))
        targets_to_genes.setdefault(target, set(str(row['target_subunit']).split('-')))

    return {
        'ligands_to_targets': ligands_to_targets,
        'targets_to_ligands': targets_to_ligands,
        'ligands_to_genes': ligands_to_genes,
        'targets_to_genes': targets_to_genes,
    }


def load_producibility_matrix(path: str) -> pd.DataFrame:
    matrix: pd.DataFrame = pd.read_csv(path, index_col=0)
    return matrix.astype(bool)


def producing_cell_types(producibility: pd.DataFrame, ligand_or_target: str) -> List[str]:
    if ligand_or_target not in producibility.index:
        return []
    row: pd.Series = producibility.loc[ligand_or_target]
    return row.index[row.values].tolist()


def is_producible(producibility: pd.DataFrame, ligand_or_target: str, cell_type: str) -> bool:
    return ligand_or_target in producibility.index and bool(producibility.loc[ligand_or_target, cell_type])


def derive_gene_level_pathways(
        gene_perturbations: pd.DataFrame,
        db_information: Dict[str, Any],
        producible_ligands: pd.DataFrame,
        producible_targets: pd.DataFrame
) -> pd.DataFrame:
    """
    For every perturbed (cell type, gene), collect all pathways whose communication score contains that gene in that
    cell type:
    - every ligand the gene contributes to, produced by the perturbed cell type as source, towards every receiver that
      produces one of the ligand's targets
    - every target the gene is a subunit of, produced by the perturbed cell type as receiver, from every source that
      produces one of the target's ligands
    """
    pathways: List[Dict[str, str]] = []

    for _, perturbation in gene_perturbations.iterrows():
        cell_type: str = perturbation['Cell_Type']
        gene: str = perturbation['Gene']

        # The gene as a ligand contributor, i.e. the perturbed cell type is the source
        for ligand, ligand_genes in db_information['ligands_to_genes'].items():
            if gene not in ligand_genes or not is_producible(producible_ligands, ligand, cell_type):
                continue

            for target in db_information['ligands_to_targets'][ligand]:
                for receiver_cell_type in producing_cell_types(producible_targets, target):
                    pathways.append({
                        'Source': cell_type,
                        'Receiver': receiver_cell_type,
                        'Ligand': ligand,
                        'Target': target,
                        'Perturbed_Gene': gene,
                    })

        # The gene as a target subunit, i.e. the perturbed cell type is the receiver
        for target, target_genes in db_information['targets_to_genes'].items():
            if gene not in target_genes or not is_producible(producible_targets, target, cell_type):
                continue

            for ligand in db_information['targets_to_ligands'][target]:
                for source_cell_type in producing_cell_types(producible_ligands, ligand):
                    pathways.append({
                        'Source': source_cell_type,
                        'Receiver': cell_type,
                        'Ligand': ligand,
                        'Target': target,
                        'Perturbed_Gene': gene,
                    })

    gene_level: pd.DataFrame = pd.DataFrame(pathways, columns=PATHWAY_COLUMNS + ['Perturbed_Gene'])

    # A pathway can contain more than one perturbed gene (e.g. GABRA1 and GABRB3 both perturbed in the receiver)
    gene_level = (
        gene_level
        .groupby(PATHWAY_COLUMNS, as_index=False, sort=False)['Perturbed_Gene']
        .agg(lambda genes: ';'.join(sorted(set(genes))))
    )

    return gene_level


def main():
    gene_perturbations: pd.DataFrame = pd.read_csv(path_to_gene_perturbations_to_perform)
    original: pd.DataFrame = pd.read_csv(path_to_expected_perturbed_pathways)

    db_information: Dict[str, Any] = extract_db_information(path_to_interaction_db)
    producible_ligands: pd.DataFrame = load_producibility_matrix(path_to_producible_ligands)
    producible_targets: pd.DataFrame = load_producibility_matrix(path_to_producible_targets)

    unknown_cell_types: Set[str] = (
        set(gene_perturbations['Cell_Type'])
        - set(producible_ligands.columns)
        - set(producible_targets.columns)
    )
    if len(unknown_cell_types) > 0:
        raise ValueError(f'Cell types missing from the producibility matrices: {unknown_cell_types}')

    gene_level: pd.DataFrame = derive_gene_level_pathways(
        gene_perturbations=gene_perturbations,
        db_information=db_information,
        producible_ligands=producible_ligands,
        producible_targets=producible_targets
    )

    # Extend the original ground truth: all original rows in their original order, followed by the gene-level rows
    # that are not part of it
    original_annotated: pd.DataFrame = original[PATHWAY_COLUMNS].merge(gene_level, on=PATHWAY_COLUMNS, how='left')
    original_annotated['Origin'] = 'original'

    original_keys: pd.MultiIndex = pd.MultiIndex.from_frame(original[PATHWAY_COLUMNS])
    added: pd.DataFrame = gene_level[~pd.MultiIndex.from_frame(gene_level[PATHWAY_COLUMNS]).isin(original_keys)].copy()
    added['Origin'] = 'gene_level'

    extended: pd.DataFrame = pd.concat([original_annotated, added], ignore_index=True)

    if extended.duplicated(PATHWAY_COLUMNS).any():
        raise ValueError('The extended expected perturbed pathways contain duplicated pathways')

    print(f'Original expected perturbed pathways: {len(original)}')
    print(f'Gene-level expected perturbed pathways: {len(gene_level)}')
    print(f'Original pathways not re-derived at the gene level: {int(original_annotated["Perturbed_Gene"].isna().sum())}')
    print(f'Added pathways: {len(added)}')
    print(f'Extended expected perturbed pathways: {len(extended)}')

    extended[PATHWAY_COLUMNS + ['Perturbed_Gene', 'Origin']].to_csv(
        path_to_output_expected_perturbed_pathways,
        index=False
    )


if __name__ == '__main__':
    main()
