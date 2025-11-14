# MultiNeuronChat — Validation Code

This repository contains the computational workflows and scripts used to evaluate and validate the MultiNeuronChat method.
It includes the code that is required to reproduce the analyses, figures, and tables presented in the MultiNeuronChat manuscript.

The validation encompasses simulated datasets and reference data analyses, automated through Snakemake workflows executed inside a fully reproducible container environment.

## Environment
The code was executed on KTH's HPC cluster [Dardel](https://www.pdc.kth.se/), using Docker containers loaded via Apptainer for reproducibility. The full pipeline should be executable on any system supporting Apptainer and Snakemake.

## Data Availability
Raw and processed data from ROSMAP can be accessed through the Synapse repository under controlled access (Synapse ID: [syn3219045](https://www.synapse.org/Synapse:syn3219045) ), as part of the AMP-AD Knowledge Portal.
Access requires data use approval from the Rush Alzheimer’s Disease Center and adherence to the AMP-AD data use terms.

All simulated datasets used for benchmarking were generated within this repository using the provided scripts. A model for generating simulated data was fitted on the control group of [Bast _et al._ (2025)](https://www.medrxiv.org/content/10.1101/2025.03.14.25323827v1).