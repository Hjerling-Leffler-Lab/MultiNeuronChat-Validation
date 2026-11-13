import os
import re

import numpy as np
from scipy import stats

cm: float = 1/2.54

cell_type_to_short: dict[str, str] = {
    'Astrocyte': 'Astro',
    'Endothelial': 'Endo',
    'Exc L2-4 RORB': 'L2-4 RORB',
    'Exc L2-5 LINC00507': 'L2-5 LINC00507',
    'Exc L3-5 FEZF2': 'L3-5 FEZF2',
    'Exc L3-5 RORB CMAHP or CD24': 'L3-5 \n RORB CMAHP \n or CD24',
    'Exc L3-5 THEMIS UBE2F or L4-6 RORB': 'L3-5 \n THEMIS UBE2F \n or L4-6 RORB',
    'Exc L5-6 FEZF2': 'L5-6 FEZF2',
    'Exc L5-6 THEMIS': 'L5-6 THEMIS',
    'Exc mixed Cells': 'Exc mixed',
    'Fibroblast': 'Fibro',
    'Inh LAMP5': 'LAMP5',
    'Inh PVALB': 'PVALB',
    'Inh SST': 'SST',
    'Inh VIP': 'VIP',
    'Macrophages': 'Macro',
    'Microglia': 'Microglia',
    'Monocytes': 'Monocytes',
    'OPCs': 'OPCs',
    'Oligodendrocytes': 'Oligo',
    'Pericytes': 'Peri',
    'SMC': 'SMC',
    'mixed cells': 'mixed',
    'Excitatory Neurons': 'Exc',
    'Inhibitory Neurons': 'Inh',
    'Vascular Niche': 'Vascular'
}
condition_colors: dict[str, str] = {
    'NCI': '#64729C',
    'AD': '#8F5450',
}


def get_adjusted_p_values(mnc_object, statistical_test: str):
    """BH-adjusted p-values over all tested tetrads, recomputed from the raw p.

    Every ROSMAP plot selects its tetrads through this function rather than the object's stored p_values_adj, so all
    panels agree even for objects computed before the switch from BY to BH.
    """
    p_values = mnc_object.p_values[statistical_test]
    tested = np.isfinite(p_values.values)
    adjusted = np.full(p_values.shape, np.nan)
    adjusted[tested] = stats.false_discovery_control(p_values.values[tested], method='bh')
    return p_values.copy(data=adjusted)


def get_significant_tetrads(mnc_object, statistical_test: str, significance_threshold: float):
    """Return a DataFrame (source, receiver, interaction, p_adj) of tetrads below the BH threshold, sorted by p_adj."""
    p_values_adj = get_adjusted_p_values(mnc_object, statistical_test).to_dataframe('p_adj').dropna()
    significant = p_values_adj[p_values_adj['p_adj'] < significance_threshold].sort_values('p_adj').reset_index()
    return significant


def save_no_significant_tetrads_figure(output_paths: list[str], statistical_test: str, significance_threshold: float):
    """Write a placeholder figure so per-tetrad plotting rules still produce their outputs when nothing is significant."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6 * cm, 2 * cm))
    ax.text(0.5, 0.5, f'No tetrads with adj. p < {significance_threshold}\n({statistical_test})', ha='center', va='center', fontsize=7)
    ax.axis('off')
    for output_path in output_paths:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def format_p_value(p: float, prefix: str | None = 'adj. p') -> str:
    value = f'{p:.1e}' if p < 0.001 else f'{p:.3f}'
    if prefix is None:
        return value
    return f'{prefix} = {value}'


def tetrad_file_stem(source: str, receiver: str, interaction: str) -> str:
    """File-name-safe identifier of a tetrad, used to name its single-panel figures."""
    return '__'.join(re.sub(r'[^A-Za-z0-9.-]+', '_', name).strip('_') for name in [source, receiver, interaction])


def save_single_panel(fig, single_panel_dir_png: str, single_panel_dir_pdf: str, file_stem: str):
    """Save a one-panel figure as png and pdf, so main and supplementary figures can be assembled panel by panel."""
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(os.path.join(single_panel_dir_png, f'{file_stem}.png'), dpi=300)
    fig.savefig(os.path.join(single_panel_dir_pdf, f'{file_stem}.pdf'), dpi=300)
    plt.close(fig)
