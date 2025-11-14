# ========= Base with micromamba =========
FROM mambaorg/micromamba:1.5.10
LABEL authors="Gianluca Volkmer <gianluca.volkmer@ki.se>"
LABEL version=0.9

# Ensure micromamba env is active in subsequent RUN/CMD/ENTRYPOINT
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin
ENV R_LIBS_USER=/opt/R/site-library
SHELL ["/bin/bash", "-lc"]

# System deps: compilers, HDF5, git, etc.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      libhdf5-dev \
      hdf5-tools \
      git \
      ca-certificates \
      curl \
      pkg-config \
      libz-dev \
      libbz2-dev \
      liblzma-dev \
      libcurl4-openssl-dev \
      libssl-dev \
      libxml2-dev \
      libaec-dev \
      zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# ========= Conda env with pinned versions =========
# - Pin Python 3.12.3 exactly
# - Pin R-base 4.4.3 exactly
# - Install snakemake, pip, and common build helpers

RUN micromamba install -y -n base -c conda-forge -c bioconda \
    python=3.12.3 r-base=4.4.3 pip snakemake=9.9.0 anndata scikit-learn scanpy \
    r-devtools r-remotes r-tidyverse r-xml2 r-data.table r-rcpp r-cpp11 r-httr r-jsonlite r-r.utils \
    r-igraph r-svglite r-uwot r-hdf5r r-seurat r-ggpubr \
    r-rcurl r-png r-cairo r-viridis r-ggrepel r-gridextra \
    r-rcpphnsw r-rcppml r-rtsne r-pheatmap r-kableextra r-mclust \
    bioconductor-s4vectors bioconductor-iranges bioconductor-genomeinfodb \
    bioconductor-genomicranges bioconductor-sparsearray \
    bioconductor-delayedarray bioconductor-delayedmatrixstats \
    bioconductor-summarizedexperiment bioconductor-singlecellexperiment \
    bioconductor-scaledmatrix bioconductor-biocsingular \
    bioconductor-bluster bioconductor-scuttle bioconductor-scran \
    bioconductor-complexheatmap \
  && micromamba clean -a -y \
  && mkdir -p /opt/R/site-library \
  && chmod -R 777 /opt/R \
  && ln -sf /opt/conda/bin/python   /usr/local/bin/python \
  && ln -sf /opt/conda/bin/pip      /usr/local/bin/pip \
  && ln -sf /opt/conda/bin/R        /usr/local/bin/R \
  && ln -sf /opt/conda/bin/Rscript  /usr/local/bin/Rscript

RUN micromamba run -n base python -m pip install multineuronchat

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    R_LIBS_USER=/opt/R/site-library

# ========= Install R packages =========
# RUN micromamba run -n base R -q -e 'devtools::install_github("hhoeflin/hdf5r", upgrade="never", lib="/opt/R/site-library")' \
#     && micromamba run -n base R -q -e 'library(hdf5r); packageVersion("hdf5r"); print(.libPaths())'

RUN micromamba run -n base R -q -e 'devtools::install_gitlab("sysbiobig/sparsim", upgrade="never", lib="/opt/R/site-library")' \
    && micromamba run -n base R -q -e 'library(SPARSim); packageVersion("SPARSim"); print(.libPaths())'

RUN micromamba run -n base R -q -e 'devtools::install_github("mojaveazure/loomR", ref="develop", upgrade="never", lib="/opt/R/site-library")' \
    && micromamba run -n base R -q -e 'library(loomR); packageVersion("loomR"); print(.libPaths())'

# Install CellChat
RUN micromamba run -n base R -q -e 'devtools::install_github("jinworks/CellChat", upgrade="never", lib="/opt/R/site-library")' \
    && micromamba run -n base R -q -e 'library(CellChat); packageVersion("CellChat"); print(.libPaths())'

# Install NeuronChat package
RUN micromamba run -n base R -q -e 'devtools::install_github("Wei-BioMath/NeuronChat", upgrade="never", lib="/opt/R/site-library")' \
    && micromamba run -n base R -q -e 'library(NeuronChat); packageVersion("NeuronChat"); print(.libPaths())'

WORKDIR /work
ENTRYPOINT ["micromamba", "run", "-n", "base", "bash", "-lc"]