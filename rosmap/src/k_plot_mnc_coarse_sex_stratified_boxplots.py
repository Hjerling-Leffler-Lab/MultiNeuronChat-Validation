"""Sex-stratified NCI vs AD permutation tests and box plots for every significant tetrad.

Standalone: deliberately not part of the Snakefile, because the permutations take ~12 min (50,000 shuffles x 4 tests x 8
tetrads). The test results are cached in --path_to_tests_csv; later runs only replot from that CSV unless --recompute is
given (or the significant tetrads / number of permutations changed). Needs the clinical data, so it runs on Bianca.

Usage (inside the validation container, from code/validation/rosmap):
    python src/k_plot_mnc_coarse_sex_stratified_boxplots.py \
        --path_to_mnc_object results/coarse/mnc_grouping_by.pkl \
        --path_to_clinical_data data/raw/ROSMAP_clinical.csv \
        --path_to_output_dir results/figs/coarse/CVM/padj_0.05

Tests per tetrad (CVM statistic throughout; labels = NCI/AD, shuffled, add-one p-values):
    p_perm_unstratified    labels shuffled over all donors (reference)
    p_perm_sex_stratified  labels shuffled within women and within men only; statistic CVM(women) + CVM(men), the "direct"
                           combination of Pesarin & Salmaso (2010). Null: no NCI vs AD difference once sex is held fixed
    p_perm_F / p_perm_M    NCI vs AD within women / within men, labels shuffled within that sex
Also asymptotic CVM and MWU, medians and zero fractions within each sex, and Spearman score vs MMSE within AD per sex.

Outputs, under --path_to_output_dir:
    sex_stratified_tests.csv, sex_by_condition.csv
    {png,pdf}/sex_stratified_boxplots.{png,pdf}                     all significant tetrads in one grid
    {png,pdf}/single_panels/sex_stratified_boxplots/<tetrad>.{png,pdf}  one figure per tetrad
Boxes F-NCI, F-AD, M-NCI, M-AD; bracket = within-sex permutation p; line under the title = sex-stratified p.
"""
import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt

from multineuronchat import MultiNeuronChatObject

from plotting.utils import cm, cell_type_to_short, condition_colors, tetrad_file_stem, save_single_panel

matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in Illustrator / Inkscape

parser = argparse.ArgumentParser(description='Sex-stratified NCI vs AD permutation tests and box plots for the significant ROSMAP tetrads.')

parser.add_argument(
    '--path_to_mnc_object',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_clinical_data',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_output_dir',
    type=str,
    required=True,
    help='Plot directory of one test/threshold, e.g. results/figs/coarse/CVM/padj_0.05.'
)
parser.add_argument(
    '--statistical_test',
    type=str,
    default='CVM',
)
parser.add_argument(
    '--significance_threshold',
    type=float,
    default=0.05,
)
parser.add_argument(
    '--n_permutations',
    type=int,
    default=50_000,
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
)
parser.add_argument(
    '--recompute',
    action='store_true',
    help='Rerun the permutation tests even if the cached tests CSV matches.'
)
parser.add_argument(
    '--n_columns',
    type=int,
    default=4,
)
parser.add_argument(
    '--width_in_cm_per_panel',
    type=float,
    default=5,
)
parser.add_argument(
    '--height_in_cm_per_panel',
    type=float,
    default=5.5,
)

args = parser.parse_args()

path_to_output_dir: str = args.path_to_output_dir
path_to_tests_csv: str = os.path.join(path_to_output_dir, 'sex_stratified_tests.csv')
path_to_sex_by_condition_csv: str = os.path.join(path_to_output_dir, 'sex_by_condition.csv')
single_panel_dir_png: str = os.path.join(path_to_output_dir, 'png', 'single_panels', 'sex_stratified_boxplots')
single_panel_dir_pdf: str = os.path.join(path_to_output_dir, 'pdf', 'single_panels', 'sex_stratified_boxplots')

statistical_test: str = args.statistical_test
significance_threshold: float = args.significance_threshold
n_permutations: int = args.n_permutations
width_in_cm_per_panel: float = args.width_in_cm_per_panel
height_in_cm_per_panel: float = args.height_in_cm_per_panel

subject_label_column: str = 'individualID'
sex_labels: dict[int, str] = {0: 'F', 1: 'M'}


def tetrad_title(source: str, receiver: str, interaction: str) -> str:
    short = lambda cell_type: cell_type_to_short.get(cell_type, cell_type)
    return f'{short(source)} → {short(receiver)}  {interaction.replace("_", "–")}'


def format_permutation_p(p: float) -> str:
    """Bare p-value; at the add-one floor (no shuffle reached the observed statistic) it is shown as an upper bound."""
    floor = 1 / (n_permutations + 1)
    if p <= floor * (1 + 1e-9):
        return f'< {1 / n_permutations:.0e}'
    return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'


def load_clinical_data() -> pd.DataFrame:
    clinical_data = pd.read_csv(args.path_to_clinical_data)
    clinical_data = clinical_data.drop_duplicates(subset=subject_label_column).set_index(subject_label_column)
    return clinical_data[['cts_mmse30_lv', 'msex']]


def get_significant_tetrads(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    """BH over all tested tetrads, re-adjusted from the raw p, so objects from before the BH switch (storing BY) work too."""
    p_values = mnc_object.p_values[statistical_test]
    tested = np.isfinite(p_values.values)
    adjusted = np.full(p_values.shape, np.nan)
    adjusted[tested] = stats.false_discovery_control(p_values.values[tested], method='bh')
    p_values_adj = p_values.copy(data=adjusted).to_dataframe('p_adj').dropna()
    raw = p_values.to_dataframe('p_raw')['p_raw']
    significant = p_values_adj[p_values_adj['p_adj'] < significance_threshold].sort_values('p_adj')
    return significant.join(raw).reset_index()


def get_tetrad_df(mnc_object, clinical_data, source: str, receiver: str, interaction: str) -> pd.DataFrame:
    """One row per donor with a score and a known sex: score, condition, sex, MMSE."""
    frames = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_series().dropna()
        frames.append(pd.DataFrame({'score': scores, 'condition': condition}))
    df = pd.concat(frames).join(clinical_data, how='left')
    n_missing = df['msex'].isna().sum()
    if n_missing > 0:
        print(f'Warning: dropping {n_missing} donors without sex for {source} -> {receiver} {interaction}')
    return df.dropna(subset=['msex'])


def cvm_statistic(scores: np.ndarray, is_ad: np.ndarray) -> float:
    return stats.cramervonmises_2samp(scores[~is_ad], scores[is_ad]).statistic


def stratified_statistic(scores: np.ndarray, is_ad: np.ndarray, strata: list[np.ndarray]) -> float:
    return sum(cvm_statistic(scores[stratum], is_ad[stratum]) for stratum in strata)


def permutation_p(scores: np.ndarray, is_ad: np.ndarray, strata: list[np.ndarray], statistic, rng) -> float:
    """Shuffle the condition labels within each stratum (one stratum = unstratified). Add-one estimate, never 0."""
    observed = statistic(scores, is_ad)
    n_at_least = 0
    permuted = is_ad.copy()
    for _ in range(n_permutations):
        for stratum in strata:
            permuted[stratum] = rng.permutation(is_ad[stratum])
        n_at_least += statistic(scores, permuted) >= observed - 1e-12
    return (n_at_least + 1) / (n_permutations + 1)


def within_sex(df: pd.DataFrame, nci: str, ad: str, sex: int) -> dict:
    sex_df = df[df['msex'] == sex]
    nci_scores, ad_scores = sex_df.loc[sex_df['condition'] == nci, 'score'], sex_df.loc[sex_df['condition'] == ad, 'score']
    label = sex_labels[sex]
    ad_df = sex_df[sex_df['condition'] == ad].dropna(subset=['cts_mmse30_lv'])
    rho, rho_p = stats.spearmanr(ad_df['score'], ad_df['cts_mmse30_lv']) if len(ad_df) > 2 else (np.nan, np.nan)
    return {
        f'n_NCI_{label}': len(nci_scores), f'n_AD_{label}': len(ad_scores),
        f'median_NCI_{label}': nci_scores.median(), f'median_AD_{label}': ad_scores.median(),
        f'frac_zero_NCI_{label}': (nci_scores == 0).mean(), f'frac_zero_AD_{label}': (ad_scores == 0).mean(),
        f'p_cvm_{label}': stats.cramervonmises_2samp(nci_scores, ad_scores).pvalue,
        f'p_mwu_{label}': stats.mannwhitneyu(nci_scores, ad_scores).pvalue,
        f'rho_mmse_AD_{label}': rho, f'p_rho_mmse_AD_{label}': rho_p,
    }


def test_tetrad(df: pd.DataFrame, nci: str, ad: str, rng) -> dict:
    scores = df['score'].to_numpy(dtype=float)
    is_ad = (df['condition'] == ad).to_numpy()
    sex = df['msex'].to_numpy()
    everyone = [np.ones(len(df), dtype=bool)]
    by_sex = [sex == 0, sex == 1]
    within_sex_p = {
        f'p_perm_{sex_labels[sex_code]}': permutation_p(
            scores[stratum], is_ad[stratum], [np.ones(stratum.sum(), dtype=bool)], cvm_statistic, rng,
        )
        for sex_code, stratum in zip((0, 1), by_sex)
    }
    return {
        'p_perm_unstratified': permutation_p(scores, is_ad, everyone, cvm_statistic, rng),
        'p_perm_sex_stratified': permutation_p(
            scores, is_ad, by_sex, lambda s, a: stratified_statistic(s, a, by_sex), rng,
        ),
        **within_sex_p,
        **within_sex(df, nci, ad, 0),
        **within_sex(df, nci, ad, 1),
    }


def load_cached_tests(titles: list[str]) -> pd.DataFrame | None:
    """The cached tests, if they cover exactly these tetrads with the same number of permutations."""
    if args.recompute or not os.path.exists(path_to_tests_csv):
        return None
    cached = pd.read_csv(path_to_tests_csv, index_col='tetrad')
    matches = (
        {'n_permutations', 'seed'}.issubset(cached.columns) and sorted(cached.index) == sorted(titles)
        and (cached['n_permutations'] == n_permutations).all() and (cached['seed'] == args.seed).all()
    )
    if not matches:
        print(f'Cached {path_to_tests_csv} does not match the current tetrads / n_permutations / seed, recomputing')
        return None
    print(f'Reusing permutation tests from {path_to_tests_csv} (pass --recompute to rerun them)')
    return cached.loc[titles]


def run_tests(tetrad_dfs: dict[str, pd.DataFrame], significant: pd.DataFrame, nci: str, ad: str) -> pd.DataFrame:
    rng = np.random.default_rng(args.seed)
    rows = []
    for (title, df), tetrad in zip(tetrad_dfs.items(), significant.itertuples()):
        print(f'Testing {title} ({n_permutations} permutations x 4)')
        rows.append({
            'tetrad': title, 'p_raw_mnc': tetrad.p_raw, 'p_adj_bh_mnc': tetrad.p_adj, 'n_donors': len(df),
            'n_permutations': n_permutations, 'seed': args.seed,
            **test_tetrad(df, nci, ad, rng),
        })
    results = pd.DataFrame(rows).set_index('tetrad')
    results.to_csv(path_to_tests_csv)
    return results


def draw_bracket(ax, x_left: float, x_right: float, y: float, height: float, label: str):
    ax.plot([x_left, x_left, x_right, x_right], [y, y + height, y + height, y], color='black', linewidth=0.5)
    ax.text((x_left + x_right) / 2, y + 1.4 * height, label, ha='center', va='bottom', fontsize=6)


def plot_sex_condition_boxes(ax, df: pd.DataFrame, result: pd.Series, title: str, nci: str, ad: str):
    """Boxes for F-NCI, F-AD, M-NCI, M-AD; one bracket per sex with the within-sex permutation p."""
    rng = np.random.default_rng(args.seed)
    groups = [(0, nci), (0, ad), (1, nci), (1, ad)]
    positions = np.array([0, 1, 2.5, 3.5])
    values_per_group = [df.loc[(df['msex'] == sex) & (df['condition'] == condition), 'score'].to_numpy() for sex, condition in groups]

    box = ax.boxplot(
        values_per_group,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8},
        whiskerprops={'linewidth': 0.6},
        capprops={'linewidth': 0.6},
        boxprops={'linewidth': 0.6},
    )
    for patch, (_, condition) in zip(box['boxes'], groups):
        patch.set_facecolor(condition_colors.get(condition, 'grey'))
        patch.set_alpha(0.6)
    for position, values in zip(positions, values_per_group):
        jitter = rng.uniform(-0.18, 0.18, size=values.shape[0])
        ax.scatter(position + jitter, values, s=1.5, color='black', alpha=0.5, linewidths=0, zorder=3)

    y_max = max(values.max() for values in values_per_group)
    y_min = min(values.min() for values in values_per_group)
    y_range = y_max - y_min if y_max > y_min else 1.0
    bracket_y, bracket_height = y_max + 0.05 * y_range, 0.025 * y_range
    for sex_index, label in enumerate(('F', 'M')):
        draw_bracket(ax, positions[2 * sex_index], positions[2 * sex_index + 1], bracket_y, bracket_height,
                     format_permutation_p(result[f'p_perm_{label}']))
    ax.set_ylim(y_min - 0.04 * y_range, y_max + 0.22 * y_range)

    ax.set_xticks(positions)
    ax.set_xticklabels([f'{sex_labels[sex]}\n{condition}' for sex, condition in groups], fontsize=6)
    ax.tick_params(axis='y', labelsize=6)
    p_text = format_permutation_p(result['p_perm_sex_stratified'])
    p_text = f'p {p_text}' if p_text.startswith('<') else f'p = {p_text}'
    # Tetrad as the title; the combined test on a smaller line below it, worded so it doesn't read as an F vs M test
    ax.set_title(title, fontsize=7, pad=10)
    ax.annotate(f'NCI vs AD, sex-stratified {p_text}', xy=(0.5, 1), xycoords='axes fraction', xytext=(0, 2),
                textcoords='offset points', ha='center', va='bottom', fontsize=6)
    ax.spines[['top', 'right']].set_visible(False)


def main():
    for directory in [os.path.join(path_to_output_dir, 'png'), os.path.join(path_to_output_dir, 'pdf'), single_panel_dir_png, single_panel_dir_pdf]:
        os.makedirs(directory, exist_ok=True)

    mnc_object = MultiNeuronChatObject.load(args.path_to_mnc_object)
    nci, ad = mnc_object.condition_names
    clinical_data = load_clinical_data()

    significant = get_significant_tetrads(mnc_object)
    print(f'{len(significant)} tetrads with {statistical_test} BH-adjusted p < {significance_threshold}')
    if len(significant) == 0:
        return

    tetrad_dfs = {
        tetrad_title(row.source, row.receiver, row.interaction):
            get_tetrad_df(mnc_object, clinical_data, row.source, row.receiver, row.interaction)
        for row in significant.itertuples()
    }

    # Sex balance between the groups (donors of the first tetrad; all tetrads share the donor set up to missing scores)
    first_df = next(iter(tetrad_dfs.values()))
    sex_by_condition = pd.crosstab(first_df['msex'].map(sex_labels), first_df['condition'])[[nci, ad]]
    sex_by_condition.to_csv(path_to_sex_by_condition_csv)
    _, fisher_p = stats.fisher_exact(sex_by_condition.to_numpy())
    print(f'Donors by sex and condition (Fisher exact p = {fisher_p:.3g}):\n{sex_by_condition}')

    results = load_cached_tests(list(tetrad_dfs))
    if results is None:
        results = run_tests(tetrad_dfs, significant, nci, ad)
    with pd.option_context('display.width', 200, 'display.max_columns', 10):
        print(results[['p_raw_mnc', 'p_perm_unstratified', 'p_perm_sex_stratified', 'p_perm_F', 'p_perm_M']])

    # All tetrads in one grid
    titles = list(tetrad_dfs)
    n_columns = min(args.n_columns, len(titles))
    n_rows = int(np.ceil(len(titles) / n_columns))
    fig, axes = plt.subplots(
        n_rows, n_columns,
        figsize=(n_columns * width_in_cm_per_panel * cm, n_rows * height_in_cm_per_panel * cm),
        squeeze=False,
    )
    for ax, title in zip(axes.flat, titles):
        plot_sex_condition_boxes(ax, tetrad_dfs[title], results.loc[title], title, nci, ad)
    for ax in axes.flat[len(titles):]:
        ax.set_visible(False)
    for row_axes in axes:
        row_axes[0].set_ylabel('Communication score', fontsize=7)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(os.path.join(path_to_output_dir, extension, f'sex_stratified_boxplots.{extension}'), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # One figure per tetrad
    for row, title in zip(significant.itertuples(), titles):
        fig, ax = plt.subplots(1, 1, figsize=(width_in_cm_per_panel * cm, height_in_cm_per_panel * cm))
        plot_sex_condition_boxes(ax, tetrad_dfs[title], results.loc[title], title, nci, ad)
        ax.set_ylabel('Communication score', fontsize=7)
        save_single_panel(fig, single_panel_dir_png, single_panel_dir_pdf, tetrad_file_stem(row.source, row.receiver, row.interaction))

    print(f'Results written to {path_to_output_dir}')


if __name__ == '__main__':
    main()
