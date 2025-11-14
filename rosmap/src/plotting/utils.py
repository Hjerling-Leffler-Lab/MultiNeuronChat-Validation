import seaborn as sns


cm: float = 1/2.54

statistical_test_short: dict[str, str] = {
    'KS': 'KS',
    'Anderson': 'AD',
    'CVM': 'CVM',
    'MannWhitneyU': 'MWU'
}

statistical_test_colors: dict[str, str] = {
    'KS': '#ca0020',
    'Anderson': '#f4a582',
    'CVM': '#92c5de',
    'MannWhitneyU': '#0571b0'
}
wasserstein_statistical_test_colors: dict[str, str] = {
    x: y
    for x, y in zip(['KS', 'Anderson', 'CVM', 'MannWhitneyU'], sns.color_palette('husl', n_colors=4))
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