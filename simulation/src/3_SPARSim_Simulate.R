library(jsonlite)

library(loomR)

library(SPARSim)

# Extract args for parallel processing
args <- commandArgs(trailingOnly = TRUE)

# Args are organized as follows:
# 1. path_to_gene_perturbations_to_perform
# 2. path_to_cell_type_count_model
# 3. path_to_SPARSim_gene_count_model
# 4. path_to_datasets_loom_folder
# 5. Settings (number)

path_to_gene_perturbations_to_perform <- args[1]
path_to_cell_type_count_model <- args[2]
path_to_SPARSim_gene_count_model <- args[3]
path_to_datasets_loom_folder <- args[4]
run_settings <- as.numeric(args[5])
n_people_to_simulate <- as.numeric(args[6])
log2fc <- as.numeric(args[7])

# Update: 12.12.2024 for reproducibility
set.seed(42 * run_settings)

# Read gene perturbations to perform
gene_perturbations_to_perform <- read.csv(path_to_gene_perturbations_to_perform)

cell_type_count_model <- jsonlite::read_json(path_to_cell_type_count_model)
joined_SPARSim_model <- readRDS(path_to_SPARSim_gene_count_model)

save_to_loom <- function (data, path_to_loom) {
  cell_types <- names(data[['Data']])

  genes <- data[['Genes']]
  subject <- data[['Donor']]
  condition <- data[['Disease']]
  sex <- data[['Sex']]

  data_mat <- NULL

  df_cell_ids <- c()
  df_cell_types <- c()
  df_subjects <- c()
  df_conditions <- c()
  df_sex <- c()

  for(cell_type in cell_types) {
    count_matrix <- data[['Data']][[cell_type]]

    if(is.null(count_matrix)) {
      next
    }

    n_cells <- ncol(count_matrix)
    cell_ids <- colnames(count_matrix)

    df_cell_ids <- c(df_cell_ids, cell_ids)
    df_cell_types <- c(df_cell_types, rep(cell_type, n_cells))
    df_subjects <- c(df_subjects, rep(subject, n_cells))
    df_conditions <- c(df_conditions, rep(condition, n_cells))
    df_sex <- c(df_sex, rep(sex, n_cells))

    if(is.null(data_mat)) {
      data_mat <- count_matrix
    } else {
      data_mat <- cbind(data_mat, count_matrix)
    }
  }

  meta_data <- list(
    'cluster_name_15CTs' = df_cell_types,
    'Donor' = df_subjects,
    'Disease' = df_conditions,
    'Sex' = df_sex
  )

  loomR::create(
    filename=path_to_loom,
    data=data_mat,
    do.transpose=TRUE,
    gene.attrs=genes,
    cell.attrs=meta_data,
    overwrite=TRUE
  )
}

sample_from_gamma_distribution <- function(shape, scale, n) {
  return(rgamma(n, shape=shape, scale=scale))
}

sample_uniform <- function(n, min, max) {
  return(ceiling(runif(n, min=min, max=max)))
}

generate_cell_id <- function(n=1) {
  # Generate a random cell id
  # Format: AAAAA0000BBBBB
  # AAAAA: 5 random uppercase letters
  # 0000: 4 random digits
  # BBBBB: 5 random uppercase letters
  # Example: ABCDE1234FGHIJ
  a <- do.call(paste0, replicate(5, sample(LETTERS, n, TRUE), FALSE))
  b <- do.call(paste0, replicate(5, sample(LETTERS, n, TRUE), FALSE))
  return(paste0(a, sprintf("%04d", sample(9999, n, TRUE)), b))
}

prepare_SPARSim_parameters <- function (model_parameters, cell_count_parameters, perturbations, log2fc) {
  # Gamma settings
  shape <- cell_count_parameters$shape
  scale <- cell_count_parameters$scale

  original_intensity <- model_parameters$intensity
  original_variability <- model_parameters$variability
  original_library_size <- model_parameters$lib_size
  original_library_size_length <- length(original_library_size)

  n_cells <- round(sample_from_gamma_distribution(shape=shape, scale=scale, n=1))

  # Ensure that there are at least 20 cells (simplification of the model, as sampling from the gamma distribution
  # can result in very low cell counts)
  if (n_cells < 20) {
    n_cells <- 20
  }

  # Sample library size for the n_cells (with replacement)
  # If we would use the fitted library sizes directly, we would have the same number of cells for each
  # simulated subject.
  lib_size_samples <- sample_uniform(n=n_cells, min=1, max=original_library_size_length)

  lib_size <- original_library_size[lib_size_samples]
  names(lib_size) <- generate_cell_id(n_cells)

  intensity <- original_intensity
  for(i in seq_along(perturbations$Cell_Type)) {
    gene_to_perturbe <- perturbations[i,]$Gene
    up_or_down <- if(perturbations[i,]$Up_or_Down == "True") {log2fc} else {-log2fc}

    intensity[gene_to_perturbe] <- intensity[gene_to_perturbe] * 2^(up_or_down)
  }

  variability <- original_variability
  name <- 'SPARSim Perturbation'

  return_value <- list(
    list(
      name=name,
      intensity=intensity,
      variability=variability,
      lib_size=lib_size
    )
  )
  return(return_value)
}

simulate_perturbation <- function(
    model_parameters,
    cell_count_parameters,
    perturbation_to_perform,
    condition_name,
    path_to_datasets_loom_folder,
    condition_label = 'PERT',
    n_subjects_to_simulate=40,
    log2fc=0) {
  path_to_datasets_loom_folder_female <- paste0(path_to_datasets_loom_folder, condition_name, '/f/')
  path_to_datasets_loom_folder_male <- paste0(path_to_datasets_loom_folder, condition_name, '/m/')

  dir.create(path_to_datasets_loom_folder_female, showWarnings = FALSE, recursive = TRUE)
  dir.create(path_to_datasets_loom_folder_male, showWarnings = FALSE, recursive = TRUE)

  female_parameters <- model_parameters$f
  male_parameters <- model_parameters$m

  cell_types <- names(female_parameters)
  genes <- names(model_parameters$m$`Excitatory Layer 2-3 IT neurons I`[[1]]$intensity)

  # Simulate females
  for(i in 1:n_subjects_to_simulate) {
    female_i_data <- list(
      'Genes' = genes,
      'Donor' = paste0('F_', condition_name, '_', i),
      'Disease' = condition_label,
      'Sex' = 'f',
      'Data' = list()
    )

    for(cell_type in cell_types) {
      perturbations_for_cell_type <- perturbation_to_perform[perturbation_to_perform$Cell_Type == cell_type, ]

      female_i_parameters <- prepare_SPARSim_parameters(
        female_parameters[[cell_type]][[1]],
        cell_count_parameters[[cell_type]],
        perturbations_for_cell_type,
        log2fc = log2fc
      )

      if(is.null(female_i_parameters)) {
        # If there are no cells, continue to the next cell type
        next
      }

      sim_res <- SPARSim_simulation(
        female_i_parameters,
        count_data_simulation_seed = 22 * run_settings + i,
        gene_expr_simulation_seed = 42 * run_settings + i,
      )

      count_matrix <- sim_res$count_matrix

      female_i_data[['Data']][[cell_type]] <- count_matrix
    }

    path_to_loom <- paste0(path_to_datasets_loom_folder_female, condition_name, '_', i, '.loom')
    save_to_loom(female_i_data, path_to_loom)

    rm(female_i_data)
    gc()
  }

  # Simulate males
  for(i in 1:n_subjects_to_simulate) {
    male_i_data <- list(
      'Genes' = genes,
      'Donor' = paste0('M_', condition_name, '_', i),
      'Disease' = condition_label,
      'Sex' = 'm',
      'Data' = list()
    )

    for(cell_type in cell_types) {
      perturbations_for_cell_type <- perturbation_to_perform[perturbation_to_perform$Cell_Type == cell_type, ]

      male_i_parameters <- prepare_SPARSim_parameters(
        male_parameters[[cell_type]][[1]],
        cell_count_parameters[[cell_type]],
        perturbations_for_cell_type,
        log2fc = log2fc
      )

      if(is.null(male_i_parameters)) {
        # If there are no cells, continue to the next cell type
        next
      }

      sim_res <- SPARSim_simulation(
        male_i_parameters,
        count_data_simulation_seed = 22 * run_settings + i + n_subjects_to_simulate,
        gene_expr_simulation_seed = 42 * run_settings + i + n_subjects_to_simulate,
      )

      count_matrix <- sim_res$count_matrix

      male_i_data[['Data']][[cell_type]] <- count_matrix
    }

    path_to_loom <- paste0(path_to_datasets_loom_folder_male, condition_name, '_', i, '.loom')
    save_to_loom(male_i_data, path_to_loom)

    rm(male_i_data)
    gc()
  }
}

condition_label <- 'CASE'
if(run_settings == 1) {
  condition_label <- 'CTRL'
}
condition_name <- paste(condition_label, log2fc, sep='_')

simulate_perturbation(
  model_parameters=joined_SPARSim_model,
  cell_count_parameters=cell_type_count_model,
  perturbation_to_perform=gene_perturbations_to_perform,
  condition_name=condition_name,
  path_to_datasets_loom_folder=path_to_datasets_loom_folder,
  condition_label = condition_label,
  n_subjects_to_simulate=n_people_to_simulate,
  log2fc=log2fc
)

print('DONE!')

for(i in 1:length(gene_perturbations_to_perform[[1]])) {
  cell_type <- gene_perturbations_to_perform[i, ]$Cell_Type
  gene <- gene_perturbations_to_perform[i, ]$Gene
  up_down <- gene_perturbations_to_perform[i, ]$Up_or_Down

  if(up_down == 'False') {
    print(paste0('Cell type: ', cell_type, ' Gene: ', gene, ' Up/Down: ', up_down))
    print(paste0('Female intesity: ', joined_SPARSim_model$f[[cell_type]][[1]]$intensity[[gene]]))
    print(paste0('Female variability: ', joined_SPARSim_model$f[[cell_type]][[1]]$variability[[gene]]))
    print(paste0('Male intesity: ', joined_SPARSim_model$m[[cell_type]][[1]]$intensity[[gene]]))
    print(paste0('Male variability: ', joined_SPARSim_model$m[[cell_type]][[1]]$variability[[gene]]))
    print('')
  }
}
