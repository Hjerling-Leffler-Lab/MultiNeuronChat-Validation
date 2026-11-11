# ========= Revision release image =========
# Layered on the previous image so that everything except the multineuronchat package stays byte-identical:
# Dockerfile.base installs R packages from GitHub HEAD (CellChat, NeuronChat, loomR, SPARSim) and unpinned
# conda packages, so rebuilding it from scratch could change the NeuronChat and simulation results.
# Dockerfile.base is the full recipe of the base image (2026.9.5.dev0).
FROM ghcr.io/siedler/multi-neuron-chat-validation-pipeline:2026.9.5.dev0
LABEL authors="Gianluca Volkmer <gianluca.volkmer@ki.se>"
LABEL version=2026.11.11

# The dependency pins of multineuronchat are unchanged since 2026.8.18.dev0, so --no-deps leaves the environment as is
RUN micromamba run -n base python -m pip install --no-deps multineuronchat==2026.11.11 \
    && micromamba run -n base python -c "import importlib.metadata as m; assert m.version('multineuronchat') == '2026.11.11'"

WORKDIR /work
ENTRYPOINT ["micromamba", "run", "-n", "base", "bash", "-lc"]
