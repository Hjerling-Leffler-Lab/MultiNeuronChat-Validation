import pandas as pd

from typing import *

import argparse

# The expected perturbed pathways written by 1_extract_ligand_target_production.R are defined per sampled ligand or
# target (e.g. GABA-A_a4b3g2S_complex in Inhibitory SST neurons). The simulation, however, perturbs a single gene of
# that ligand/target (gene_perturbations_to_perform.csv, e.g. GABRB3). If that gene is also a contributor/subunit of
# other ligands or targets (GABRB3 is a subunit of eleven GABA-A complexes), those change as well, but were missing from
# the ground truth. This script derives the expected perturbed pathways at the gene level from the recorded gene
# perturbations and the (seeded) producibility matrices, and writes a report comparing them to the original file.

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
    help='Path to save the gene-level expected perturbed pathways CSV file.'
)
parser.add_argument(
    '--path_to_output_comparison_report',
    type=str,
    required=True,
    help='Path to save a text report comparing the gene-level to the original expected perturbed pathways.'
)
args = parser.parse_args()

path_to_gene_perturbations_to_perform: str = args.path_to_gene_perturbations_to_perform
path_to_interaction_db: str = args.path_to_interaction_db
path_to_expected_perturbed_pathways: str = args.path_to_expected_perturbed_pathways
path_to_producible_ligands: str = args.path_to_producible_ligands
path_to_producible_targets: str = args.path_to_producible_targets
path_to_output_expected_perturbed_pathways: str = args.path_to_output_expected_perturbed_pathways
path_to_output_comparison_report: str = args.path_to_output_comparison_report

PATHWAY_COLUMNS: List[str] = ['Source', 'Receiver', 'Ligand', 'Target']

REASON_SAME_LIGAND_OR_TARGET: str = 'same_ligand_or_target_other_partner_cell_type'
REASON_SHARED_GENE: str = 'perturbed_gene_in_other_ligand_or_target'

REASON_DESCRIPTIONS: Dict[str, str] = {
    REASON_SHARED_GENE: 'The perturbed gene is also part of another ligand/target (e.g. a subunit shared between '
                        'receptor complexes), or acts on the other side of the pathway (gap junctions)',
    REASON_SAME_LIGAND_OR_TARGET: 'Same sampled ligand/target as in the original, but a partner cell type is '
                                  'producible in the seeded producibility matrices and was not in the original '
                                  '(unseeded) draw',
}


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


def sampled_role(
        cell_type: str,
        ligand_or_target: str,
        original: pd.DataFrame,
        db_information: Dict[str, Any]
) -> str:
    """
    Whether script 1 sampled the ligand/target as a ligand (the cell type is the source) or as a target (the cell type
    is the receiver). Read from the original expected pathways; if there are none (no producible partner), fall back to
    the rule of 2_extract_genes_to_perturb_from_perturbations_to_perform.py, which checks ligands first.
    """
    if ((original['Source'] == cell_type) & (original['Ligand'] == ligand_or_target)).any():
        return 'ligand'
    if ((original['Receiver'] == cell_type) & (original['Target'] == ligand_or_target)).any():
        return 'target'
    return 'ligand' if ligand_or_target in db_information['ligands_to_genes'] else 'target'


def derive_gene_level_pathways(
        gene_perturbations: pd.DataFrame,
        original: pd.DataFrame,
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
    Each pathway is annotated with the perturbed gene(s) and whether it stems from the sampled ligand/target itself or
    from another ligand/target sharing the perturbed gene.
    """
    pathways: List[Dict[str, str]] = []

    for _, perturbation in gene_perturbations.iterrows():
        cell_type: str = perturbation['Cell_Type']
        gene: str = perturbation['Gene']
        sampled: str = perturbation['Ligand_or_Target']
        role: str = sampled_role(cell_type, sampled, original, db_information)

        # The gene as a ligand contributor, i.e. the perturbed cell type is the source
        for ligand, ligand_genes in db_information['ligands_to_genes'].items():
            if gene not in ligand_genes or not is_producible(producible_ligands, ligand, cell_type):
                continue

            is_sampled: bool = ligand == sampled and role == 'ligand'
            for target in db_information['ligands_to_targets'][ligand]:
                for receiver_cell_type in producing_cell_types(producible_targets, target):
                    pathways.append({
                        'Source': cell_type,
                        'Receiver': receiver_cell_type,
                        'Ligand': ligand,
                        'Target': target,
                        'Perturbed_Gene': gene,
                        'Reason': REASON_SAME_LIGAND_OR_TARGET if is_sampled else REASON_SHARED_GENE,
                    })

        # The gene as a target subunit, i.e. the perturbed cell type is the receiver
        for target, target_genes in db_information['targets_to_genes'].items():
            if gene not in target_genes or not is_producible(producible_targets, target, cell_type):
                continue

            is_sampled: bool = target == sampled and role == 'target'
            for ligand in db_information['targets_to_ligands'][target]:
                for source_cell_type in producing_cell_types(producible_ligands, ligand):
                    pathways.append({
                        'Source': source_cell_type,
                        'Receiver': cell_type,
                        'Ligand': ligand,
                        'Target': target,
                        'Perturbed_Gene': gene,
                        'Reason': REASON_SAME_LIGAND_OR_TARGET if is_sampled else REASON_SHARED_GENE,
                    })

    gene_level: pd.DataFrame = pd.DataFrame(pathways, columns=PATHWAY_COLUMNS + ['Perturbed_Gene', 'Reason'])

    # A pathway can contain more than one perturbed gene (e.g. GABRA1 and GABRB3 both perturbed in the receiver). If
    # any of them stems from the sampled ligand/target, that reason takes precedence.
    gene_level = gene_level.groupby(PATHWAY_COLUMNS, as_index=False, sort=False).agg(
        Perturbed_Gene=('Perturbed_Gene', lambda genes: ';'.join(sorted(set(genes)))),
        Reason=('Reason', lambda reasons: REASON_SAME_LIGAND_OR_TARGET
                if REASON_SAME_LIGAND_OR_TARGET in set(reasons) else REASON_SHARED_GENE),
    )

    return gene_level


def pathway_keys(df: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_frame(df[PATHWAY_COLUMNS])


def not_rederived_reason(
        row: pd.Series,
        producible_ligands: pd.DataFrame,
        producible_targets: pd.DataFrame
) -> str:
    reasons: List[str] = []
    if not is_producible(producible_ligands, row['Ligand'], row['Source']):
        reasons.append(f'{row["Source"]} does not produce {row["Ligand"]}')
    if not is_producible(producible_targets, row['Target'], row['Receiver']):
        reasons.append(f'{row["Receiver"]} does not produce {row["Target"]}')
    if len(reasons) == 0:
        reasons.append('no perturbed gene in this pathway')
    return '; '.join(reasons) + ' in the seeded producibility matrices'


def format_pathway(row: pd.Series) -> str:
    return f'{row["Source"]} -> {row["Receiver"]} : {row["Ligand"]}_{row["Target"]}'


def write_comparison_report(
        path: str,
        original: pd.DataFrame,
        gene_level: pd.DataFrame,
        not_rederived: pd.DataFrame,
        added: pd.DataFrame,
        producible_ligands: pd.DataFrame,
        producible_targets: pd.DataFrame,
        gene_perturbations: pd.DataFrame
) -> None:
    n_kept: int = len(original) - len(not_rederived)
    n_perturbations_without_pathways: int = sum(
        not ((gene_level['Perturbed_Gene'].str.split(';').apply(lambda genes: row['Gene'] in genes))
             & ((gene_level['Source'] == row['Cell_Type']) | (gene_level['Receiver'] == row['Cell_Type']))).any()
        for _, row in gene_perturbations.iterrows()
    )

    lines: List[str] = [
        'Comparison of the gene-level to the original expected perturbed pathways',
        '=' * 74,
        '',
        f'Original expected perturbed pathways (script 1):         {len(original)}',
        f'Gene-level expected perturbed pathways (this script):    {len(gene_level)}',
        f'  of which also in the original:                         {n_kept}',
        f'  of which added:                                        {len(added)}',
        f'Original pathways not re-derived (removed):              {len(not_rederived)}',
        '',
        f'Gene perturbations:                                      {len(gene_perturbations)}',
        f'  without any expected perturbed pathway:                {n_perturbations_without_pathways}',
        '',
        'Added pathways by reason:',
    ]
    for reason, description in REASON_DESCRIPTIONS.items():
        lines.append(f'  {reason}: {int((added["Reason"] == reason).sum())}')
        lines.append(f'    ({description})')

    lines += ['', 'Added pathways by perturbed gene:']
    for gene, count in added['Perturbed_Gene'].value_counts().items():
        lines.append(f'  {gene}: {count}')

    lines += ['', '-' * 74, 'Original pathways not re-derived', '-' * 74]
    if len(not_rederived) == 0:
        lines.append('  none')
    for _, row in not_rederived.iterrows():
        lines.append(f'  {format_pathway(row)}')
        lines.append(f'    reason: {not_rederived_reason(row, producible_ligands, producible_targets)}')

    for reason in REASON_DESCRIPTIONS:
        lines += ['', '-' * 74, f'Added pathways: {reason}', '-' * 74]
        added_with_reason: pd.DataFrame = added[added['Reason'] == reason]
        if len(added_with_reason) == 0:
            lines.append('  none')
        for _, row in added_with_reason.iterrows():
            lines.append(f'  {format_pathway(row)}  [perturbed gene: {row["Perturbed_Gene"]}]')

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


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
        original=original,
        db_information=db_information,
        producible_ligands=producible_ligands,
        producible_targets=producible_targets
    )

    is_in_original: pd.Series = pd.Series(pathway_keys(gene_level).isin(pathway_keys(original)), index=gene_level.index)
    not_rederived: pd.DataFrame = original[~pathway_keys(original).isin(pathway_keys(gene_level))]
    added: pd.DataFrame = gene_level[~is_in_original]

    # Output: the pathways that are also in the original first, in their original order, followed by the added ones
    kept: pd.DataFrame = original[PATHWAY_COLUMNS].merge(gene_level, on=PATHWAY_COLUMNS, how='inner')
    kept['Origin'] = 'original'
    kept['Reason'] = ''
    added_output: pd.DataFrame = added.copy()
    added_output['Origin'] = 'added'

    output: pd.DataFrame = pd.concat([kept, added_output], ignore_index=True)
    if output.duplicated(PATHWAY_COLUMNS).any() or len(output) != len(gene_level):
        raise ValueError('The gene-level expected perturbed pathways are inconsistent')

    output[PATHWAY_COLUMNS + ['Perturbed_Gene', 'Origin', 'Reason']].to_csv(
        path_to_output_expected_perturbed_pathways,
        index=False
    )

    write_comparison_report(
        path=path_to_output_comparison_report,
        original=original,
        gene_level=gene_level,
        not_rederived=not_rederived,
        added=added,
        producible_ligands=producible_ligands,
        producible_targets=producible_targets,
        gene_perturbations=gene_perturbations
    )

    print(f'Original expected perturbed pathways: {len(original)}')
    print(f'Gene-level expected perturbed pathways: {len(gene_level)} ({len(gene_level) - len(added)} also in the '
          f'original, {len(added)} added)')
    print(f'Original pathways not re-derived: {len(not_rederived)}')


if __name__ == '__main__':
    main()
