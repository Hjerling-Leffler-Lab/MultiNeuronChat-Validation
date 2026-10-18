cm: float = 1/2.54

statistical_test_short: dict[str, str] = {
    'KS': 'KS',
    'CVM': 'CVM',
    'MannWhitneyU': 'MWU'
}

statistical_test_colors: dict[str, str] = {
    'KS': '#ca0020',
    'CVM': '#92c5de',
    'MannWhitneyU': '#0571b0'
}

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


def get_significant_tetrads(mnc_object, statistical_test: str, significance_threshold: float):
    """Return a DataFrame (source, receiver, interaction, p_adj) of tetrads below the threshold, sorted by p_adj."""
    p_values_adj = mnc_object.p_values_adj[statistical_test].to_dataframe('p_adj').dropna()
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


def format_p_value(p: float, prefix: str = 'adj. p') -> str:
    if p < 0.001:
        return f'{prefix} = {p:.1e}'
    return f'{prefix} = {p:.3f}'
