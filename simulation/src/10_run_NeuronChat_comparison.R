#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(NeuronChat)
  library(ggplot2)
  library(patchwork)
})

# -----------------------------
# Helper utilities
# -----------------------------
ensure_dir <- function(path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
}

# Load either an .rds or an .RData/.rda file and return the first object
load_any_r_obj <- function(path) {
  # Try RDS first
  obj <- tryCatch(readRDS(path), error = function(e) NULL)
  if (!is.null(obj)) return(obj)
  # Fallback: RData into a temp env
  e <- new.env()
  nm <- load(path, envir = e)
  if (length(nm) == 0) stop(sprintf("No objects found in '%s'", path))
  if (length(nm) > 1) {
    warning(sprintf("Multiple objects in '%s'. Using the first: %s", path, nm[1]))
  }
  get(nm[1], envir = e)
}

# -----------------------------
# Args (15 required + 3 optional labels)
# -----------------------------
#  1  M0_ctrl_rds_in
#  2  M0_case_rds_in
#  3  M100_ctrl_rds_in
#  4  M100_case_rds_in
#  5  M0_merged_rds_out
#  6  M100_merged_rds_out
#  7  M0_comparison_png_out
#  8  M0_rankNet_png_out
#  9  M100_comparison_png_out
# 10  M100_rankNet_png_out
# 11  M0_rankNet_counts_rds_out
# 12  M0_rankNet_weights_rds_out
# 13  M100_rankNet_counts_rds_out
# 14  M100_rankNet_weights_rds_out
# 15  timing_csv_out
# 16  (optional) case_label
# 17  (optional) proportion_label
# 18  (optional) dataset_id

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 15) {
  stop(paste0(
    "Usage: 10_run_NeuronChat_comparison.R ",
    "<M0_ctrl_rds_in> <M0_case_rds_in> <M100_ctrl_rds_in> <M100_case_rds_in> ",
    "<M0_merged_rds_out> <M100_merged_rds_out> ",
    "<M0_comparison_png_out> <M0_rankNet_png_out> <M100_comparison_png_out> <M100_rankNet_png_out> ",
    "<M0_rankNet_counts_rds_out> <M0_rankNet_weights_rds_out> <M100_rankNet_counts_rds_out> <M100_rankNet_weights_rds_out> ",
    "<timing_csv_out> [case_label] [proportion_label] [dataset_id]"
  ))
}

M0_ctrl_rds_in               <- args[1]
M0_case_rds_in               <- args[2]
M100_ctrl_rds_in             <- args[3]
M100_case_rds_in             <- args[4]
M0_merged_rds_out            <- args[5]
M100_merged_rds_out          <- args[6]
M0_comparison_png_out        <- args[7]
M0_rankNet_png_out           <- args[8]
M100_comparison_png_out      <- args[9]
M100_rankNet_png_out         <- args[10]
M0_rankNet_counts_rds_out    <- args[11]
M0_rankNet_weights_rds_out   <- args[12]
M100_rankNet_counts_rds_out  <- args[13]
M100_rankNet_weights_rds_out <- args[14]
TIMING_CSV_OUT               <- args[15]

case_label        <- if (length(args) >= 16) args[16] else NA
proportion_label  <- if (length(args) >= 17) args[17] else NA
dataset_id_label  <- if (length(args) >= 18) args[18] else NA

# Make sure output dirs exist
lapply(c(
  M0_merged_rds_out, M100_merged_rds_out,
  M0_comparison_png_out, M0_rankNet_png_out,
  M100_comparison_png_out, M100_rankNet_png_out,
  M0_rankNet_counts_rds_out, M0_rankNet_weights_rds_out,
  M100_rankNet_counts_rds_out, M100_rankNet_weights_rds_out,
  TIMING_CSV_OUT
), ensure_dir)

# -----------------------------
# M0: load, merge, plot, rankNet
# -----------------------------
ctrl_m0 <- load_any_r_obj(M0_ctrl_rds_in)
case_m0 <- load_any_r_obj(M0_case_rds_in)

results_list <- list(ctrl = ctrl_m0, case = case_m0)
merged_m0 <- NeuronChat::mergeNeuronChat(results_list, add.names = names(results_list))
saveRDS(merged_m0, M0_merged_rds_out)

# Comparison plots (weight + count)
p1 <- NeuronChat::compareInteractions_Neuron(merged_m0, measure = c("weight"), comparison = c(1,2), group = c(1,2), show.legend = FALSE)
p2 <- NeuronChat::compareInteractions_Neuron(merged_m0, measure = c("count"),  comparison = c(1,2), group = c(1,2), show.legend = FALSE)

# RankNet timing start
start.time.m0 <- Sys.time()

rankNet_Neuron_count_results <- NeuronChat::rankNet_Neuron(
  merged_m0, mode = 'comparison', measure = c("count"), comparison = 1:2,
  do.stat = TRUE, tol = 0.1, stacked = FALSE, font.size = 11, return.data = TRUE
)
rankNet_Neuron_weight_results <- NeuronChat::rankNet_Neuron(
  merged_m0, mode = 'comparison', measure = c("weight"), comparison = 1:2,
  do.stat = TRUE, tol = 0.1, stacked = FALSE, font.size = 11, return.data = TRUE
)

end.time.m0 <- Sys.time()

# Save plots
try(ggsave(filename = M0_comparison_png_out, plot = p1 + p2, width = 29.7, height = 21, dpi = 300, units = "cm"), silent = TRUE)
try(ggsave(filename = M0_rankNet_png_out, plot = rankNet_Neuron_count_results$gg.obj + rankNet_Neuron_weight_results$gg.obj,
           width = 21, height = 29.7, dpi = 300, units = "cm"), silent = TRUE)

# Save rankNet data
saveRDS(rankNet_Neuron_count_results$signaling.contribution,  M0_rankNet_counts_rds_out)
saveRDS(rankNet_Neuron_weight_results$signaling.contribution, M0_rankNet_weights_rds_out)

# -----------------------------
# M100: load, merge, plot, rankNet
# -----------------------------
ctrl_m100 <- load_any_r_obj(M100_ctrl_rds_in)
case_m100 <- load_any_r_obj(M100_case_rds_in)

results_list_100 <- list(ctrl = ctrl_m100, case = case_m100)
merged_m100 <- NeuronChat::mergeNeuronChat(results_list_100, add.names = names(results_list_100))
saveRDS(merged_m100, M100_merged_rds_out)

p1_100 <- NeuronChat::compareInteractions_Neuron(merged_m100, measure = c("weight"), comparison = c(1,2), group = c(1,2), show.legend = FALSE)
p2_100 <- NeuronChat::compareInteractions_Neuron(merged_m100, measure = c("count"),  comparison = c(1,2), group = c(1,2), show.legend = FALSE)

start.time.m100 <- Sys.time()

rankNet_count_100 <- NeuronChat::rankNet_Neuron(
  merged_m100, mode = 'comparison', measure = c("count"), comparison = 1:2,
  do.stat = TRUE, tol = 0.1, stacked = FALSE, font.size = 11, return.data = TRUE
)
rankNet_weight_100 <- NeuronChat::rankNet_Neuron(
  merged_m100, mode = 'comparison', measure = c("weight"), comparison = 1:2,
  do.stat = TRUE, tol = 0.1, stacked = FALSE, font.size = 11, return.data = TRUE
)

end.time.m100 <- Sys.time()

try(ggsave(filename = M100_comparison_png_out, plot = p1_100 + p2_100, width = 29.7, height = 21, dpi = 300, units = "cm"), silent = TRUE)
try(ggsave(filename = M100_rankNet_png_out, plot = rankNet_count_100$gg.obj + rankNet_weight_100$gg.obj,
           width = 21, height = 29.7, dpi = 300, units = "cm"), silent = TRUE)

saveRDS(rankNet_count_100$signaling.contribution,  M100_rankNet_counts_rds_out)
saveRDS(rankNet_weight_100$signaling.contribution, M100_rankNet_weights_rds_out)

# -----------------------------
# Timing CSV
# -----------------------------

timing_results <- data.frame(
  case = case_label,
  proportion = proportion_label,
  dataset_id = dataset_id_label,
  start_time_m0 = start.time.m0,
  end_time_m0 = end.time.m0,
  start_time_m100 = start.time.m100,
  end_time_m100 = end.time.m100,
  duration_m0_secs = as.numeric(difftime(end.time.m0, start.time.m0, units = "secs")),
  duration_m100_secs = as.numeric(difftime(end.time.m100, start.time.m100, units = "secs"))
)

write.csv(timing_results, file = TIMING_CSV_OUT, row.names = FALSE)

cat("Done. Wrote:\n",
    M0_merged_rds_out, "\n",
    M100_merged_rds_out, "\n",
    M0_comparison_png_out, "\n",
    M0_rankNet_png_out, "\n",
    M100_comparison_png_out, "\n",
    M100_rankNet_png_out, "\n",
    M0_rankNet_counts_rds_out, "\n",
    M0_rankNet_weights_rds_out, "\n",
    M100_rankNet_counts_rds_out, "\n",
    M100_rankNet_weights_rds_out, "\n",
    TIMING_CSV_OUT, "\n")