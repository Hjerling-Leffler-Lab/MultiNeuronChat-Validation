# Re-creates producible_ligands_2 / producible_targets_2 of 1_extract_ligand_target_production.R (lines 211-298) and
# exports them as CSV matrices (rows: ligands or targets, columns: cell types).
#
# The criterion is identical to script 1: a ligand/target is producible in a cell type if, for every gene group, the
# summed 25/50/75% quantiles of the simulated raw counts of the group's genes are > 0.
#
# The only difference: script 1 calls SPARSim_simulation() without count_data_simulation_seed, so SPARSim's count
# sampling is seeded from std::random_device and set.seed() does not control it. Here both SPARSim seeds are set
# explicitly, which makes the matrices reproducible.

args <- commandArgs(trailingOnly = TRUE)

path_to_joined_simulation_parameters <- args[1]
path_to_human_extended_database <- args[2]
seed <- as.integer(args[3])
path_to_output_producible_ligands <- args[4]
path_to_output_producible_targets <- args[5]

# Ligand/target names and their genes and groups, parsed as in script 1
database <- read.csv(path_to_human_extended_database, stringsAsFactors = FALSE, sep = '\t')
interaction_names_split <- strsplit(database$interaction_name, '_')
ligand_names <- sapply(interaction_names_split, function(x) x[1])
target_names <- sapply(interaction_names_split, function(x) paste(x[2:length(x)], collapse = '_'))

genes_and_groups <- function(names, genes_column, groups_column) {
  result <- list()
  for (i in seq_along(names)) {
    # As in script 1, the genes of a ligand/target are taken from its first occurrence in the database
    if (!(names[i] %in% names(result))) {
      result[[names[i]]] <- list(
        genes = strsplit(database[[genes_column]][i], '-')[[1]],
        groups = strsplit(as.character(database[[groups_column]][i]), '-')[[1]]
      )
    }
  }
  return(result)
}

ligand_to_genes_and_groups <- genes_and_groups(ligand_names, 'lig_contributor', 'lig_contributor_group')
target_to_genes_and_groups <- genes_and_groups(target_names, 'target_subunit', 'target_subunit_group')

is_producible <- function(genes_and_groups, count_matrix) {
  producible <- TRUE
  for (group in unique(genes_and_groups$groups)) {
    genes_in_group <- genes_and_groups$genes[genes_and_groups$groups == group]
    genes_in_group <- genes_in_group[genes_in_group %in% rownames(count_matrix)]

    s <- 0
    for (gene in genes_in_group) {
      s <- s + sum(quantile(count_matrix[gene, ], c(0.25, 0.5, 0.75)))
    }

    producible <- producible & s > 0
  }
  return(producible)
}

simulation_params <- readRDS(path_to_joined_simulation_parameters)
cell_types <- names(simulation_params$f)

producible_ligands <- matrix(
  data = TRUE, nrow = length(ligand_to_genes_and_groups), ncol = length(cell_types),
  dimnames = list(names(ligand_to_genes_and_groups), cell_types)
)
producible_targets <- matrix(
  data = TRUE, nrow = length(target_to_genes_and_groups), ncol = length(cell_types),
  dimnames = list(names(target_to_genes_and_groups), cell_types)
)

for (i in seq_along(cell_types)) {
  cell_type <- cell_types[i]

  count_matrix <- SPARSim::SPARSim_simulation(
    simulation_params$f[[cell_type]],
    gene_expr_simulation_seed = seed + i,
    count_data_simulation_seed = seed + i
  )$count_matrix

  for (ligand in names(ligand_to_genes_and_groups)) {
    producible_ligands[ligand, cell_type] <- is_producible(ligand_to_genes_and_groups[[ligand]], count_matrix)
  }
  for (target in names(target_to_genes_and_groups)) {
    producible_targets[target, cell_type] <- is_producible(target_to_genes_and_groups[[target]], count_matrix)
  }

  rm(count_matrix)
  gc()
}

write.csv(producible_ligands, path_to_output_producible_ligands)
write.csv(producible_targets, path_to_output_producible_targets)
