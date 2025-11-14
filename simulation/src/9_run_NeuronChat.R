# Load the NeuronChat package
library(NeuronChat)

# Load loomR
library(loomR)

# Get as args the following input:
# Path to loom file
# Path to M0 ctrl rds
# Path to M0 case rds
# Path to M100 ctrl rds
# Path to M100 case rds
# Path to timing csv file
args <- commandArgs(trailingOnly = TRUE)

if (length(args) != 6) {
  stop("Usage: script.R <loom_path> <M0_ctrl_rds> <M0_case_rds> <M100_ctrl_rds> <M100_case_rds> <timing_csv>")
}

# ---- Assign arguments ----
path_to_loom_file      <- args[1]
path_to_M0_ctrl_rds    <- args[2]
path_to_M0_case_rds    <- args[3]
path_to_M100_ctrl_rds  <- args[4]
path_to_M100_case_rds  <- args[5]
path_to_timing_csv     <- args[6]

# ---- Print inputs for verification ----
cat("Input files:\n")
cat("  Loom file:        ", path_to_loom_file, "\n")
cat("  M0 Ctrl RDS:      ", path_to_M0_ctrl_rds, "\n")
cat("  M0 Case RDS:      ", path_to_M0_case_rds, "\n")
cat("  M100 Ctrl RDS:    ", path_to_M100_ctrl_rds, "\n")
cat("  M100 Case RDS:    ", path_to_M100_case_rds, "\n")
cat("  Timing CSV:       ", path_to_timing_csv, "\n")

# Create path if not existing
dir.create(dirname(path_to_M0_ctrl_rds), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(path_to_M0_case_rds), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(path_to_M100_ctrl_rds), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(path_to_M100_case_rds), showWarnings = FALSE, recursive = TRUE)
dir.create(dirname(path_to_timing_csv), showWarnings = FALSE, recursive = TRUE)

# Read the original loom file.
# This should return an object containing the count matrix and its attributes.
print("Reading loom file")
start.time.reading_loom <- Sys.time()
print(paste("Start time:", start.time.reading_loom))

loom_data <- connect(filename = path_to_loom_file, mode = "r", skip.validate=TRUE)

# Extract the count matrix and attributes.
# (Adjust the field names if your object structure differs)
data <- t(loom_data[["matrix"]][,])

cell_id <- loom_data[['col_attrs/CellID']][]
genes_id <- loom_data[['row_attrs/Gene']][]

colnames(data) <- cell_id
rownames(data) <- genes_id

cell_types <- loom_data[['col_attrs/cluster_name_15CTs']][]
disease <- loom_data[['col_attrs/Disease']][]
donor_id <- loom_data[['col_attrs/Donor']][]
sex <- loom_data[['col_attrs/Sex']][]

meta_data <- data.frame(sex, donor_id, disease, cell_types)
rownames(meta_data) <- cell_id

loom_data$close_all()

ctrl_idx <- (meta_data$disease == 'CTRL')
case_idx <- !ctrl_idx

# split matrix into control- and schizophrenic-group
ctrl_data <- data[,ctrl_idx]
ctrl_cell_type_annotation <- cell_types[ctrl_idx]
ctrl_meta_data <- meta_data[ctrl_idx,]

colnames(ctrl_data) <- cell_id[ctrl_idx]
rownames(ctrl_data) <- genes_id
rownames(ctrl_meta_data) <- cell_id[ctrl_idx]

case_data <- data[,case_idx]
case_cell_type_annotation <- cell_types[case_idx]
case_meta_data <- meta_data[case_idx,]

colnames(case_data) <- cell_id[case_idx]
rownames(case_data) <- genes_id
rownames(case_meta_data) <- cell_id[case_idx]

# Do some clean up after the data was loaded
rm(data)
rm(cell_id)
rm(genes_id)
rm(cell_types)
rm(disease)
rm(donor_id)
rm(sex)

end.time.reading_loom <- Sys.time()
print(paste("End time:", end.time.reading_loom))
print("")

print("Start M=0 NeuronChat run")
start.time.m0 <- Sys.time()
# Run for control
print("NeuronChat M=0 Control")
x_0_ctrl <- createNeuronChat(
  ctrl_data,
  DB='human',
  group.by = ctrl_cell_type_annotation,
  meta=ctrl_meta_data
)
# calculation of communication networks
x_0_ctrl <- run_NeuronChat(x_0_ctrl, M=0)

end.time.m0.ctrl <- Sys.time()

# Run for case
print("NeuronChat M=0 Case")
x_0_case <- createNeuronChat(
  case_data,
  DB='human',
  group.by = case_cell_type_annotation,
  meta=case_meta_data
)
# calculation of communication networks
x_0_case <- run_NeuronChat(x_0_case,M=0)

end.time.m0 <- Sys.time()

time.neuron_chat.m0.ctrl <- end.time.m0.ctrl - start.time.m0
time.neuron_chat.m0.scz <- end.time.m0 - end.time.m0.ctrl
time.neuron_chat.m0 <- end.time.m0 - start.time.m0

time_diff_secs <- as.numeric(difftime(end.time.m0 , start.time.m0, units = "secs"))
print(paste("Finished M=0 NeuronChat run in", time_diff_secs, "seconds"))
print("")

# save the results
save(x_0_ctrl, file=path_to_M0_ctrl_rds)
save(x_0_case, file=path_to_M0_case_rds)

print("Start M=100 NeuronChat run")
start.time.m100 <- Sys.time()

# Run for ctrl
x_100_ctrl <- createNeuronChat(
  ctrl_data,
  DB='human',
  group.by = ctrl_cell_type_annotation,
  meta=ctrl_meta_data
)
# calculation of communication networks
x_100_ctrl <- run_NeuronChat(x_100_ctrl,M=100)

end.time.m100.ctrl <- Sys.time()

# Run for case
x_100_case <- createNeuronChat(
  case_data,
  DB='human',
  group.by = case_cell_type_annotation,
  meta=case_meta_data
)
# calculation of communication networks
x_100_case <- run_NeuronChat(x_100_case,M=100)

end.time.m100 <- Sys.time()

time.neuron_chat.m100.ctrl <- end.time.m100.ctrl - start.time.m100
time.neuron_chat.m100.scz <- end.time.m100 - end.time.m100.ctrl
time.neuron_chat.m100 <- end.time.m100 - start.time.m100

time_diff_secs <- as.numeric(difftime(end.time.m100 , start.time.m100, units = "secs"))
print(paste("Finished M=100 NeuronChat run in", time_diff_secs, "seconds"))
print("")

save(x_100_ctrl, file=path_to_M100_ctrl_rds)
save(x_100_case, file=path_to_M100_case_rds)

# Save timings
time_df <- data.frame(
  start.time.reading_loom = start.time.reading_loom,
  end.time.reading_loom = end.time.reading_loom,
  diff.time.reading_loom = as.numeric(end.time.reading_loom - start.time.reading_loom),

  start.time.m0 = start.time.m0,
  end.time.m0.ctrl = end.time.m0.ctrl,
  end.time.m0 = end.time.m0,
  diff.time.m0.ctrl = as.numeric(end.time.m0.ctrl - start.time.m0),
  diff.time.m0.case = as.numeric(end.time.m0 - end.time.m0.ctrl),
  diff.time.m0 = as.numeric(end.time.m0 - start.time.m0),

  start.time.m100 = start.time.m100,
  end.time.m100.ctrl = end.time.m100.ctrl,
  end.time.m100 = end.time.m100,
  diff.time.m100.ctrl = as.numeric(end.time.m100.ctrl - start.time.m100),
  diff.time.m100.case = as.numeric(end.time.m100 - end.time.m100.ctrl),
  diff.time.m100 = as.numeric(end.time.m100 - start.time.m100)

)
write.csv(time_df, path_to_timing_csv, row.names = FALSE)
print("Done!")