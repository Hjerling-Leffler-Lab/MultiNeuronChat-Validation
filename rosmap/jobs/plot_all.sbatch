#!/bin/bash
# All ROSMAP figures and the Excel summary (Snakemake target plot_all) on Bianca, in one job.
# Snakemake runs inside the pipeline container (it ships Snakemake), so no executor plugin and no docker:// pull are
# needed; every rule runs locally in this job. Submit from rosmap/ after `mkdir -p logs`:
#   sbatch jobs/plot_all.sbatch
#
# Needs results/coarse/mnc_grouping_by.pkl (step 5), data/raw/ROSMAP_clinical.csv and data/raw/cell-annotation.n437.csv.
# Step 5 is NOT rerun: its code changed for the freeze (coarse annotation only), which would otherwise trigger a 48 h
# rerun. The stored object already holds the raw CVM p-values and every plot recomputes BH from them, so the object is
# touched and the rerun triggers are limited to file times. For a clean step-5 rerun, use a separate long job instead.
#
# Outputs and logs stay on Bianca. Only aggregated files may be exported, after checking each one: the circle plot,
# clinical_covariates/summary_heatmap, results/tables/*.xlsx and the *_tests.csv / *_stats.csv / significant_tetrads.csv.
# Box plots with donor dots, the AD-only clustermap, covariate_plots/ and the donor-order CSV are individual-level.
#SBATCH -A sens2025558
#SBATCH -p core
#SBATCH -J rosmap_plot_all
#SBATCH -t 02:00:00
#SBATCH -n 1
#SBATCH -c 4                          # 4 rules in parallel, ~14 GB per core
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

export TMPDIR="${SNIC_TMP:-/tmp}"

CONTAINER=/proj/sens2025558/MultiNeuronChat/containers/multi-neuron-chat-validation-pipeline_2026.9.5.dev0.sif

echo "Job ${SLURM_JOB_ID} on $(hostname), started $(date)"

run_snakemake() {
    apptainer exec --bind /proj/sens2025558 --bind "${TMPDIR}:/tmp" "${CONTAINER}" \
        snakemake --snakefile workflow/Snakefile --cores "${SLURM_CPUS_PER_TASK}" --rerun-triggers mtime "$@"
}

# Mark the existing step-5 outputs as up to date instead of recomputing them
run_snakemake --touch results/coarse/mnc_grouping_by.pkl results/coarse/timings.pkl

run_snakemake --rerun-incomplete --printshellcmds plot_all

echo "Finished $(date)"
