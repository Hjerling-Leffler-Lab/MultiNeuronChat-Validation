args <- commandArgs(trailingOnly = TRUE)

path_to_joined_simulation_parameters <- args[1]
path_to_cell_count_parameters <- args[2]
path_to_human_extened_database <- args[3]
path_to_output_perturbations_to_perform <- args[4]
path_to_output_expected_perturbed_pathways <- args[5]

path_to_output_dir <- dirname(path_to_output_perturbations_to_perform)
if(!dir.exists(path_to_output_dir)) {
  dir.create(path_to_output_dir, recursive = TRUE, showWarnings = FALSE)
}

# Stored in  the env variable PDC_TMP
path_to_tmp <- '/tmp/'
path_to_temporary_intermediate_results <- file.path(path_to_tmp, 'R_intermediate_results')
if(!dir.exists(path_to_temporary_intermediate_results)) {
    dir.create(path_to_temporary_intermediate_results, recursive = TRUE, showWarnings = FALSE)
}

# Path to producible_ligands_2 and producible_targets_2
path_to_producible_ligands_2 <- file.path(path_to_temporary_intermediate_results, 'producible_ligands_2.rds')
path_to_producible_targets_2 <- file.path(path_to_temporary_intermediate_results, 'producible_targets_2.rds')

# path_to_joined_simulation_parameters <- './data/simulation/estimated_parameters_all_genes_combined.rds'
# path_to_cell_count_parameters <- './data/simulation/cell_types_gamma_parameters.json'
# path_to_human_extened_database <- './db/interactionDB_human_extended.csv'
# path_to_output_perturbations_to_perform <- './output/simulation/perturbations_to_perform.csv'
# path_to_output_expected_perturbed_pathways <- './output/simulation/expected_perturbed_pathways.csv'


load_human_extened_database <- function(path_to_database) {
  database <- read.csv(path_to_database, stringsAsFactors = FALSE, sep='\t')

  # Split some columns from strings to list of strings based on seperator -
  database$lig_contributor <- strsplit(database$lig_contributor, '-')
  database$lig_contributor_coeff <- strsplit(database$lig_contributor_coeff, '-')
  database$lig_contributor_group <- strsplit(database$lig_contributor_group, '-')

  database$target_subunit <- strsplit(database$target_subunit, '-')
  database$target_subunit_coeff <- strsplit(database$target_subunit_coeff, '-')
  database$target_subunit_group <- strsplit(database$target_subunit_group, '-')

  return(database)
}

extract_unique_ligands_and_targets <- function(database) {
  # The unique ligands and targets can be extracted from the interaction_name, which is a string
  # of two values: ligand_target. Attention, as the complex targets can consist of multiple subunits, their names
  # contain additional "_" which should be ignored.
  interaction_names_split <- strsplit(database$interaction_name, '_')

  ligand_names <- sapply(interaction_names_split, function(x) x[1])
  target_names <- sapply(interaction_names_split, function(x) paste(x[2:length(x)], collapse='_'))

  unique_ligands <- unique(ligand_names)
  unique_targets <- unique(target_names)

  # create a dictionary from ligand strings to a list of targets
  ligands_to_targets <- list()
  for(ligand in unique_ligands) {
    if(!(ligand %in% names(ligands_to_targets))) {
      ligands_to_targets[[ligand]] <- list()
    }

    ligands_to_targets[[ligand]] <- c(ligands_to_targets[[ligand]], target_names[ligand_names == ligand])
  }

  targets_to_ligands <- list()
  for(target in unique_targets) {
    if(!(target %in% names(targets_to_ligands))) {
      targets_to_ligands[[target]] <- list()
    }

    targets_to_ligands[[target]] <- c(targets_to_ligands[[target]], ligand_names[target_names == target])
  }

  # Create a map from ligands and targets to their genes and their groups
  ligand_to_genes_and_groups <- list()
  target_to_genes_and_groups <- list()
  for(interaction_name in database$interaction_name) {
    interaction_row <- database[database$interaction_name == interaction_name,]

    ligand_name <- strsplit(interaction_name, '_')[[1]][1]
    target_name <- paste(tail(strsplit(interaction_name, '_')[[1]], -1), collapse='_')

    if(!(ligand_name %in% names(ligand_to_genes_and_groups))) {
      ligand_to_genes_and_groups[[ligand_name]] <- list(
        genes=interaction_row$lig_contributor[[1]],
        groups=interaction_row$lig_contributor_group[[1]]
      )
    }

    if(!(target_name %in% names(target_to_genes_and_groups))) {
      target_to_genes_and_groups[[target_name]] <- list(
        genes=interaction_row$target_subunit[[1]],
        groups=interaction_row$target_subunit_group[[1]]
      )
    }
  }

  return(list(
    unique_ligands = unique_ligands,
    unique_targets = unique_targets,
    ligands_to_targets = ligands_to_targets,
    targets_to_ligands = targets_to_ligands,
    ligand_to_genes_and_groups = ligand_to_genes_and_groups,
    target_to_genes_and_groups = target_to_genes_and_groups
  ))
}

human_db <- load_human_extened_database(path_to_human_extened_database)

unique_ligands_and_targets <- extract_unique_ligands_and_targets(human_db)
unique_ligands <- unique_ligands_and_targets$unique_ligands
unique_targets <- unique_ligands_and_targets$unique_targets
ligands_to_targets <- unique_ligands_and_targets$ligands_to_targets
targets_to_ligands <- unique_ligands_and_targets$targets_to_ligands
ligand_to_genes_and_groups <- unique_ligands_and_targets$ligand_to_genes_and_groups
target_to_genes_and_groups <- unique_ligands_and_targets$target_to_genes_and_groups

# Perform sanity check to see if ligands_to_targets and targets_to_ligands are consistent
for(ligand in unique_ligands) {
  targets <- ligands_to_targets[[ligand]]
  for(target in targets) {
    if(!(ligand %in% targets_to_ligands[[target]])) {
      print(paste('Inconsistency found for ligand', ligand, 'and target', target))
    }
  }

  if(length(ligands_to_targets[[ligand]]) != length(unique(ligands_to_targets[[ligand]]))) {
    print(paste('Duplicate targets found for ligand: ',ligand))
  }
}
for(target in unique_targets) {
  ligands <- targets_to_ligands[[target]]
  for(ligand in ligands) {
    if(!(target %in% ligands_to_targets[[ligand]])) {
      print(paste('Inconsistency found for ligand', ligand, 'and target', target))
    }
  }

  if(length(targets_to_ligands[[target]]) != length(unique(targets_to_ligands[[target]]))) {
    print(paste('Duplicate targets found for target: ', target))
  }
}

# Load simulation model
simulation_params <- readRDS(path_to_joined_simulation_parameters)

cell_types <- names(simulation_params$f)

n_cell_types <- length(cell_types)
n_ligands <- length(unique_ligands)
n_targets <- length(unique_targets)

# Check which cell types can produce which ligands and which targets
producible_ligands <- matrix(data=TRUE, nrow=n_ligands, ncol=n_cell_types)
producible_targets <- matrix(data=TRUE, nrow=n_targets, ncol=n_cell_types)

rownames(producible_ligands) <- unique_ligands
colnames(producible_ligands) <- cell_types

rownames(producible_targets) <- unique_targets
colnames(producible_targets) <- cell_types

# Calculate which ligands can/cannot be produced per cell-type
for(ligand in unique_ligands) {
  unique_ligand_groups <- unique(ligand_to_genes_and_groups[[ligand]]$groups)

  for(group in unique_ligand_groups) {
    ligand_group_genes_mask <- ligand_to_genes_and_groups[[ligand]]$groups == group
    ligand_group_genes <- ligand_to_genes_and_groups[[ligand]]$genes[ligand_group_genes_mask]

    present_genes_mask <- ligand_group_genes %in% names(simulation_params$f[[1]][[1]]$intensity)
    ligand_group_genes <- ligand_group_genes[present_genes_mask]

    for(cell_type in cell_types) {
      # Get intensity parameter of simulation
      ligand_group_intensities <- simulation_params$f[[cell_type]][[1]]$intensity[ligand_group_genes]

      if(all(ligand_group_intensities < 1)) {
        producible_ligands[ligand, cell_type] <- FALSE
      }
    }
  }
}

# Calculate which targets can/cannot be produced
for(target in unique_targets) {
  unique_target_groups <- unique(target_to_genes_and_groups[[target]]$groups)

  for(group in unique_target_groups) {
    target_group_genes_mask <- target_to_genes_and_groups[[target]]$groups == group
    target_group_genes <- target_to_genes_and_groups[[target]]$genes[target_group_genes_mask]

    present_genes_mask <- target_group_genes %in% names(simulation_params$f[[1]][[1]]$intensity)
    target_group_genes <- target_group_genes[present_genes_mask]

    for(cell_type in cell_types) {
      target_group_intensities <- simulation_params$f[[cell_type]][[1]]$intensity[target_group_genes]

      if(all(target_group_intensities < 1)) {
        producible_targets[target, cell_type] <- FALSE
      }
    }
  }
}


# Sanity check to see if filter criterion is well choosen
if(!file.exists(path_to_producible_ligands_2)) {
  set.seed(42)

  check_if_producible_ligands_and_targets_makes_sense <- function(sp, pl, pt, ltgg, ttgg) {
    cell_types <- names(sp$f)

    pl_2 <- matrix(data=TRUE, nrow=nrow(pl), ncol=ncol(pl))
    pt_2 <- matrix(data=TRUE, nrow=nrow(pt), ncol=ncol(pt))

    rownames(pl_2) <- rownames(pl)
    colnames(pl_2) <- colnames(pl)

    rownames(pt_2) <- rownames(pt)
    colnames(pt_2) <- colnames(pt)

    for(cell_type in cell_types) {
      simulation_results <- SPARSim::SPARSim_simulation(sp$f[[cell_type]])
      count_matrix <- simulation_results$count_matrix

      # Step through all ligands and check if the production of the ligand would be possible, given the tri mean (i.e. it should not be zero)
      for(ligand in names(ltgg)) {
        genes_to_produce_ligand <- ltgg[[ligand]]$genes
        groups_to_produce_ligand <- ltgg[[ligand]]$groups

        unique_groups <- unique(groups_to_produce_ligand)

        for(group in unique_groups) {
          genes_in_group <- genes_to_produce_ligand[groups_to_produce_ligand == group]

          s <- 0
          for(gene in genes_in_group) {
            if(!(gene %in% rownames(count_matrix))) {
              next
            }
            s <- s + sum(quantile(count_matrix[gene, ], c(0.25, 0.5, 0.75)))
          }

          pl_2[ligand, cell_type] <- pl_2[ligand, cell_type] & s > 0
        }
      }

      # Step through all targets and check if the production of the ligand would be possible, given the tri mean (i.e. it should not be zero)
      for(target in names(ttgg)) {
        genes_to_produce_target <- ttgg[[target]]$genes
        groups_to_produce_target <- ttgg[[target]]$groups

        unique_groups <- unique(groups_to_produce_target)

        for(group in unique_groups) {
          genes_in_group <- genes_to_produce_target[groups_to_produce_target == group]

          s <- 0
          for(gene in genes_in_group) {
            if(!(gene %in% rownames(count_matrix))) {
              next
            }
            s <- s + sum(quantile(count_matrix[gene, ], c(0.25, 0.5, 0.75)))
          }

          pt_2[target, cell_type] <- pt_2[target, cell_type] & s > 0
        }
      }
    }

    return(list(
      producible_ligands=pl_2,
      producible_targets=pt_2
    ))
  }

  res <- check_if_producible_ligands_and_targets_makes_sense(
    sp = simulation_params,
    pl=producible_ligands,
    pt=producible_targets,
    ltgg=ligand_to_genes_and_groups,
    ttgg=target_to_genes_and_groups
  )

  producible_ligands_2 <- res$producible_ligands
  producible_targets_2 <- res$producible_targets

  saveRDS(producible_ligands_2, path_to_producible_ligands_2)
  saveRDS(producible_targets_2, path_to_producible_targets_2)
} else{
  producible_ligands_2 <- readRDS(path_to_producible_ligands_2)
  producible_targets_2 <- readRDS(path_to_producible_targets_2)
}

#### Selecting pathways to perturb ####
# We want to randomly draw pathways to perturb from the set of already produced
# ligands and targets. A pathway is defined as
# 1. A ligand that is perturbed in a single source cell-type, with all it's associated target receptors and receiver cell-types
# 2. A target that is perturbed in a single receiver cell-type, with all it's associated ligands and source cell-types
# For example: If we perturbed the VIP ligand in the Inhibitory VIP Neurons, we expect all pathways to VIPR1 and VIPR2 to be perturbed in
# all cell-types that actually produce this target.

# First extract all source cell-type and ligand combinations + all receiver cell-type and target combinations
available_source_cell_type_and_ligand_combinations <- list()
available_receiver_cell_type_and_target_combinations <- list()

# TODO: here we use currently the "simulated" estimates if a ligand/target can be produced.
# This is because I'm not yet 100% convinced about the "theoretical" model requirments for filtering
for(ligand in unique_ligands) {
  for(cell_type in cell_types) {
    if(producible_ligands_2[ligand, cell_type]) {
      available_source_cell_type_and_ligand_combinations[[length(available_source_cell_type_and_ligand_combinations)+1]] <- c(ligand, cell_type)
    }
  }
}

for(target in unique_targets) {
  for(cell_type in cell_types) {
    producible_targets_2[target, cell_type]
    if(producible_targets_2[target, cell_type]) {
      available_receiver_cell_type_and_target_combinations[[length(available_receiver_cell_type_and_target_combinations)+1]] <- c(target, cell_type)
    }
  }
}

# The 100 000th prime number
set.seed(1299709)

tmp_available_sl <- available_source_cell_type_and_ligand_combinations
tmp_available_rt <- available_receiver_cell_type_and_target_combinations

perturbations_to_perform <- list()
expected_perturbed_pathways <- list()

total_number_of_affected_pathways <- 0
n_sampled_ligands <- 0
n_sampled_targets <- 0
while(total_number_of_affected_pathways < 300 && (length(tmp_available_sl) + length(tmp_available_rt)) > 0) {
  l_sl <- length(tmp_available_sl)
  l_rt <- length(tmp_available_rt)

  random_idx <- sample.int(l_sl+l_rt, 1)

  # Sampled a source-ligand pair
  if(random_idx <= l_sl) {
    n_sampled_ligands <- n_sampled_ligands + 1

    ligand_source_pair <- tmp_available_sl[[random_idx]]
    ligand <- ligand_source_pair[1]
    source_cell_type <- ligand_source_pair[2]

    perturbations_to_perform[[length(perturbations_to_perform)+1]] <- ligand_source_pair

    # Each ligand can have multiple targets that are associated with it
    targets <- ligands_to_targets[[ligand]]

    # Step through all targets that are associated with the ligand
    for(target in targets) {
      receiver_cell_types <- colnames(producible_targets_2)[producible_targets_2[target,]]

      # Step through all receiver cell-types that express the target and add
      # the sampled ligand-target-source-receiver to the expected
      # perturbed pathways
      for(receiver_cell_type in receiver_cell_types) {
        expected_perturbed_pathways[[length(expected_perturbed_pathways)+1]] <- c(
          source=source_cell_type,
          receiver=receiver_cell_type,
          ligand=ligand,
          target=target
        )
      }

      # Update the count of affected pathways
      total_number_of_affected_pathways <- total_number_of_affected_pathways + length(receiver_cell_types)

      # Remove target from available_rt
      if(length(tmp_available_rt) > 0) {
        idx_to_remove <- c()
        for(i in 1:length(tmp_available_rt)) {
          if(tmp_available_rt[[i]][1] == target) {
            idx_to_remove <- c(idx_to_remove, i)
          }
        }
        tmp_available_rt[idx_to_remove] <- NULL
      }
    }

    # Remove ligand from available_sr
    if(length(tmp_available_sl) > 0) {
      idx_to_remove <- c()
      for(i in 1:length(tmp_available_sl)) {
        if(tmp_available_sl[[i]][1] == ligand) {
          idx_to_remove <- c(idx_to_remove, i)
        }
      }
      tmp_available_sl[idx_to_remove] <- NULL
    }
  } else {
    n_sampled_targets <- n_sampled_targets + 1

    target_receiver_pair <- tmp_available_rt[[random_idx-l_sl]]
    target <- target_receiver_pair[1]
    receiver_cell_type <- target_receiver_pair[2]

    perturbations_to_perform[[length(perturbations_to_perform)+1]] <- target_receiver_pair

    # Each target can have multiple ligands
    ligands <- targets_to_ligands[[target]]

    # Step through all ligands that are associated with the targets
    for(ligand in ligands) {
      # Get all cell-types that express this ligand
      source_cell_types <- colnames(producible_ligands_2)[producible_ligands_2[ligand,]]

      # Step through all source cell-types that express this ligand and add
      # the sampled ligand-target-source-receiver to the expected
      # perturbed pathways
      for(source_cell_type in source_cell_types) {
        expected_perturbed_pathways[[length(expected_perturbed_pathways)+1]] <- c(
          source=source_cell_type,
          receiver=receiver_cell_type,
          ligand=ligand,
          target=target
        )
      }

      # Update the count of affected pathways
      total_number_of_affected_pathways <- total_number_of_affected_pathways + length(source_cell_types)

      # Remove ligand from available_sr
      if(length(tmp_available_sl) > 0) {
        idx_to_remove <- c()
        for(i in 1:length(tmp_available_sl)) {
          if(tmp_available_sl[[i]][1] == ligand) {
            idx_to_remove <- c(idx_to_remove, i)
          }
        }
        tmp_available_sl[idx_to_remove] <- NULL
      }
    }

    # Remove target from available_rt
    if(length(tmp_available_rt) > 0) {
      idx_to_remove <- c()
      for(i in 1:length(tmp_available_rt)) {
        if(tmp_available_rt[[i]][1] == target) {
          idx_to_remove <- c(idx_to_remove, i)
        }
      }
      tmp_available_rt[idx_to_remove] <- NULL
    }
  }
}

print(paste('Number of affected pathways:', total_number_of_affected_pathways))
print(paste('Number of ligand perturbations:', n_sampled_ligands))
print(paste('Number of target perturbations:', n_sampled_targets))

# Save everything to files
perturbations_performed_text <- 'Ligand_or_Target,Cell_Type'
for(i in 1:length(perturbations_to_perform)) {
  perturbations_performed_text <- paste(perturbations_performed_text, paste(perturbations_to_perform[[i]][1], perturbations_to_perform[[i]][2], sep=','), sep='\n')
}

expected_perturbed_pathways_text <- 'Source,Receiver,Ligand,Target'
for(i in 1:length(expected_perturbed_pathways)) {
  expected_perturbed_pathways_text <- paste(expected_perturbed_pathways_text, paste(expected_perturbed_pathways[[i]][[1]], expected_perturbed_pathways[[i]][[2]], expected_perturbed_pathways[[i]][[3]], expected_perturbed_pathways[[i]][[4]], sep=','), sep='\n')
}

performed_perturbations_file <- file(path_to_output_perturbations_to_perform)
writeLines(perturbations_performed_text, performed_perturbations_file)
close(performed_perturbations_file)

expected_perturbed_pathways_file <- file(path_to_output_expected_perturbed_pathways)
writeLines(expected_perturbed_pathways_text, expected_perturbed_pathways_file)
close(expected_perturbed_pathways_file)
