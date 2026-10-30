"""NCI vs AD communication scores of every significant tetrad against disease severity and potential confounders.

Standalone: deliberately not part of the Snakefile, because it needs the controlled-access clinical data and so runs on
Bianca. Run it before k_plot_mnc_coarse_sex_stratified_boxplots.py, which follows up on the sex effects found here.

Usage (inside the validation container, from code/validation/rosmap):
    python src/j_plot_mnc_coarse_clinical_covariates.py \
        --path_to_mnc_object results/coarse/mnc_grouping_by.pkl \
        --path_to_clinical_data data/raw/ROSMAP_clinical.csv \
        --path_to_output_dir results/figs/coarse/CVM/padj_0.05

Outputs, under --path_to_output_dir (every plot as png and pdf):
    clinical_covariates_stats.csv                                   numbers behind the heatmap (effect, nominal p, n,
                                                                    plus BH over the whole heatmap)
    {png,pdf}/clinical_covariates/summary_heatmap                   all significant tetrads x covariates: Spearman rho
                                                                    (sex: rank-biserial r), nominal p as text
    {png,pdf}/clinical_covariates/covariate_plots/<tetrad>/
        mmse_vs_score     MMSE (last visit) vs score, Spearman for AD and all donors
        braak_vs_score    Braak stage vs score (box plot), Spearman for AD and all donors
        age_vs_score      Age at death vs score, Spearman within each condition
        pmi_vs_score      Post-mortem interval vs score, Spearman within each condition
        sex_vs_score      Score by sex within each condition (box plot, Mann-Whitney U)
Box plots get a bracket with the p-value for every comparison with p < 0.05 (Braak: adjacent stages, BH).
"""
import os
import argparse

import numpy as np
import pandas as pd

from scipy import stats

import matplotlib
import matplotlib.pyplot as plt

from multineuronchat import MultiNeuronChatObject

matplotlib.rcParams['pdf.fonttype'] = 42  # editable text in Illustrator / Inkscape

from plotting.utils import cm, cell_type_to_short, condition_colors, tetrad_file_stem

parser = argparse.ArgumentParser(description='Communication scores of the significant ROSMAP tetrads vs clinical covariates.')

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
    '--p_value_correction',
    type=str,
    default='bh',
    choices=['bh', 'by', 'stored'],
    help="Re-adjust the raw p-values over all tested tetrads ('bh'/'by'), or use the object's stored adjusted p ('stored')",
)
parser.add_argument(
    '--panel_size_in_cm',
    type=float,
    default=5.0,
)
parser.add_argument(
    '--summary_heatmap_width_in_cm',
    type=float,
    default=10.0,
)
args = parser.parse_args()

path_to_output_dir: str = args.path_to_output_dir
path_to_stats_csv: str = os.path.join(path_to_output_dir, 'clinical_covariates_stats.csv')
plot_dir: str = 'clinical_covariates'
covariate_plot_dir: str = os.path.join(plot_dir, 'covariate_plots')
subject_label_column: str = 'individualID'

braak_labels: dict[int, str] = {0: '0', 1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
sex_labels: dict[int, str] = {0: 'F', 1: 'M'}


def format_p_value(p: float) -> str:
    if np.isnan(p):
        return 'n.a.'
    return f'{p:.1e}' if p < 0.001 else f'{p:.3f}'


def short(cell_type: str) -> str:
    return cell_type_to_short.get(cell_type, cell_type)


def load_clinical_data() -> pd.DataFrame:
    clinical_data = pd.read_csv(args.path_to_clinical_data)
    clinical_data = clinical_data.drop_duplicates(subset=subject_label_column).set_index(subject_label_column)
    # age_death is censored at 90 and stored as '90+'
    clinical_data['age_death'] = pd.to_numeric(clinical_data['age_death'].astype(str).str.replace('+', '', regex=False), errors='coerce')
    return clinical_data[['braaksc', 'cts_mmse30_lv', 'age_death', 'pmi', 'msex']]



def get_adjusted_p_values(mnc_object: MultiNeuronChatObject):
    """Adjusted p-values for the chosen test. Objects computed before the BH switch store BY, so re-adjust the raw p."""
    if args.p_value_correction == 'stored':
        return mnc_object.p_values_adj[args.statistical_test]
    p_values = mnc_object.p_values[args.statistical_test]
    tested = np.isfinite(p_values.values)
    adjusted = np.full(p_values.shape, np.nan)
    adjusted[tested] = stats.false_discovery_control(p_values.values[tested], method=args.p_value_correction)
    return p_values.copy(data=adjusted)


def get_significant_tetrads(mnc_object: MultiNeuronChatObject) -> pd.DataFrame:
    p_values_adj = get_adjusted_p_values(mnc_object).to_dataframe('p_adj').dropna()
    return p_values_adj[p_values_adj['p_adj'] < args.significance_threshold].sort_values('p_adj').reset_index()


def get_tetrad_df(mnc_object, clinical_data, source: str, receiver: str, interaction: str) -> pd.DataFrame:
    """One row per donor: communication score, condition and clinical covariates."""
    frames = []
    for condition in mnc_object.condition_names:
        scores = mnc_object.communication_score_dict[condition].loc[:, source, receiver, interaction].to_series().dropna()
        frames.append(pd.DataFrame({'score': scores, 'condition': condition}))
    df = pd.concat(frames).join(clinical_data, how='left')
    n_missing = df['cts_mmse30_lv'].isna().sum()
    if n_missing > 0:
        print(f'Warning: {n_missing} donors without MMSE for {source} -> {receiver} {interaction}')
    return df


def spearman(df: pd.DataFrame, x_column: str) -> tuple[float, float, int]:
    df = df.dropna(subset=[x_column, 'score'])
    if len(df) < 3 or df[x_column].nunique() < 2:
        return np.nan, np.nan, len(df)
    rho, p = stats.spearmanr(df[x_column], df['score'])
    return rho, p, len(df)


def sex_test(df: pd.DataFrame) -> tuple[float, float, int]:
    """Mann-Whitney U male vs female, effect as rank-biserial r (> 0: higher scores in males)."""
    male, female = df.loc[df['msex'] == 1, 'score'], df.loc[df['msex'] == 0, 'score']
    if len(male) == 0 or len(female) == 0:
        return np.nan, np.nan, len(male) + len(female)
    result = stats.mannwhitneyu(male, female, alternative='two-sided')
    return 2 * result.statistic / (len(male) * len(female)) - 1, result.pvalue, len(male) + len(female)


def new_panel(width_factor: float = 1.0, height_factor: float = 1.0):
    return plt.subplots(1, 1, figsize=(width_factor * args.panel_size_in_cm * cm, height_factor * args.panel_size_in_cm * cm))


def save_panel(fig, name: str, sub_dir: str):
    """Save as {png,pdf}/<sub_dir>/<name>.{png,pdf} under the output directory."""
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        os.makedirs(os.path.join(path_to_output_dir, extension, sub_dir), exist_ok=True)
        fig.savefig(os.path.join(path_to_output_dir, extension, sub_dir, f'{name}.{extension}'), dpi=300)
    plt.close(fig)


def style_axis(ax, x_label: str, y_label: str = 'Communication score'):
    ax.set_xlabel(x_label, fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    ax.spines[['top', 'right']].set_visible(False)


def annotate(ax, lines: list[str]):
    """Statistics in the top-left corner; a line starting with a condition name is drawn in its colour (doubles as legend)."""
    # Extra headroom above the data so the statistics never cover points
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.3 * (top - bottom))
    for line_index, line in enumerate(lines):
        color = next((c for condition, c in condition_colors.items() if line.startswith(f'{condition}')), 'black')
        ax.text(0.02, 0.98 - 0.075 * line_index, line, transform=ax.transAxes, ha='left', va='top', fontsize=5.5, color=color)


def scatter(ax, df: pd.DataFrame, x_column: str, condition: str, jitter: float = 0.0, alpha: float = 0.7, y_column: str = 'score'):
    rng = np.random.default_rng(0)
    condition_df = df[df['condition'] == condition].dropna(subset=[x_column, y_column])
    x = condition_df[x_column].to_numpy(dtype=float) + rng.uniform(-jitter, jitter, size=len(condition_df))
    ax.scatter(x, condition_df[y_column], s=4, color=condition_colors.get(condition, 'grey'), alpha=alpha, linewidths=0, label=condition, zorder=3)


def boxes(ax, values: list[np.ndarray], positions: list[float]):
    ax.boxplot(
        values, positions=positions, widths=0.6, showfliers=False,
        medianprops={'color': 'black', 'linewidth': 0.8}, whiskerprops={'linewidth': 0.6},
        capprops={'linewidth': 0.6}, boxprops={'linewidth': 0.6},
    )


def add_brackets(ax, comparisons: list[tuple[float, float, float]], threshold: float = 0.05):
    """Brackets above the data for comparisons (x_left, x_right, p) with p < threshold, stacked so they never overlap."""
    significant = sorted([c for c in comparisons if c[2] < threshold], key=lambda c: c[1] - c[0])
    if len(significant) == 0:
        return
    bottom, top = ax.get_ylim()
    y_range = top - bottom
    bracket_height = 0.025 * y_range
    step = 0.11 * y_range
    levels: list[list[tuple[float, float]]] = []
    for x_left, x_right, p in significant:
        # Lowest level whose brackets don't overlap this one horizontally
        # Brackets sharing an end point stay on one level, shortened a little so their ticks don't touch
        x_left, x_right = x_left + 0.06, x_right - 0.06
        level = next((i for i, spans in enumerate(levels) if all(x_right < l or x_left > r for l, r in spans)), len(levels))
        if level == len(levels):
            levels.append([])
        levels[level].append((x_left, x_right))
        y = top + 0.02 * y_range + level * step
        ax.plot([x_left, x_left, x_right, x_right], [y, y + bracket_height, y + bracket_height, y], color='black', linewidth=0.6, clip_on=False)
        ax.text((x_left + x_right) / 2, y + 1.4 * bracket_height, format_p_value(p), ha='center', va='bottom', fontsize=5)
    ax.set_ylim(bottom, top + 0.02 * y_range + len(levels) * step)


def spearman_line(df: pd.DataFrame, x_column: str, label: str) -> str:
    rho, p, _ = spearman(df, x_column)
    return f'{label}: ρ = {rho:.2f}, p = {format_p_value(p)}'


def plot_mmse(df: pd.DataFrame, nci: str, ad: str, title: str, sub_dir: str):
    fig, ax = new_panel()
    # NCI donors are faded: by definition they cluster at high MMSE, the within-AD trend is the relevant one
    scatter(ax, df, 'cts_mmse30_lv', nci, alpha=0.25)
    scatter(ax, df, 'cts_mmse30_lv', ad)
    annotate(ax, [spearman_line(df[df['condition'] == ad], 'cts_mmse30_lv', ad), spearman_line(df, 'cts_mmse30_lv', 'All')])
    ax.set_title(title, fontsize=7)
    style_axis(ax, 'MMSE (last visit)')
    save_panel(fig, 'mmse_vs_score', sub_dir)



def braak_pairwise_tests(df: pd.DataFrame, min_donors: int = 3) -> list[tuple[float, float, float]]:
    """MWU between adjacent Braak stages with >= min_donors donors (both conditions pooled), BH over the pairs.
    Adjacent pairs only: Braak is ordinal, and all 21 pairs would be unreadable; the overall trend is the Spearman ρ."""
    stages = [stage for stage in braak_labels if (df['braaksc'] == stage).sum() >= min_donors]
    pairs = list(zip(stages[:-1], stages[1:]))
    if len(pairs) == 0:
        return []
    p = [stats.mannwhitneyu(df.loc[df['braaksc'] == a, 'score'], df.loc[df['braaksc'] == b, 'score'], alternative='two-sided').pvalue for a, b in pairs]
    return [(a, b, p_adj) for (a, b), p_adj in zip(pairs, stats.false_discovery_control(p, method='bh'))]


def plot_braak(df: pd.DataFrame, nci: str, ad: str, title: str, sub_dir: str):
    fig, ax = new_panel()
    present = [stage for stage in braak_labels if (df['braaksc'] == stage).any()]
    boxes(ax, [df.loc[df['braaksc'] == stage, 'score'] for stage in present], present)
    scatter(ax, df, 'braaksc', nci, jitter=0.2)
    scatter(ax, df, 'braaksc', ad, jitter=0.2)
    ax.set_xticks(list(braak_labels))
    ax.set_xticklabels(list(braak_labels.values()), fontsize=6)
    add_brackets(ax, braak_pairwise_tests(df))
    annotate(ax, [spearman_line(df[df['condition'] == ad], 'braaksc', ad), spearman_line(df, 'braaksc', 'All')])
    ax.set_title(title, fontsize=7)
    style_axis(ax, 'Braak stage')
    save_panel(fig, 'braak_vs_score', sub_dir)


def plot_continuous_confounder(df: pd.DataFrame, conditions: list[str], x_column: str, x_label: str, title: str, name: str, sub_dir: str, jitter: float = 0.0):
    # Confounders are assessed within each condition: a pooled correlation would only re-detect the group difference
    fig, ax = new_panel()
    for condition in conditions:
        scatter(ax, df, x_column, condition, jitter=jitter)
    annotate(ax, [spearman_line(df[df['condition'] == condition], x_column, condition) for condition in conditions])
    ax.set_title(title, fontsize=7)
    style_axis(ax, x_label)
    save_panel(fig, name, sub_dir)


def plot_sex(df: pd.DataFrame, conditions: list[str], title: str, sub_dir: str):
    fig, ax = new_panel()
    rng = np.random.default_rng(0)
    positions, tick_labels, lines, comparisons = [], [], [], []
    for condition_index, condition in enumerate(conditions):
        condition_df = df[df['condition'] == condition]
        for sex_index, (sex, sex_label) in enumerate(sex_labels.items()):
            position = 3 * condition_index + sex_index
            values = condition_df.loc[condition_df['msex'] == sex, 'score']
            boxes(ax, [values], [position])
            ax.scatter(position + rng.uniform(-0.2, 0.2, size=len(values)), values, s=4, color=condition_colors.get(condition, 'grey'), alpha=0.7, linewidths=0, zorder=3)
            positions.append(position)
            tick_labels.append(f'{sex_label}\n{condition}')
        _, p, _ = sex_test(condition_df)
        lines.append(f'{condition}: MWU p = {format_p_value(p)}')
        comparisons.append((3 * condition_index, 3 * condition_index + 1, p))
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=6)
    add_brackets(ax, comparisons)
    annotate(ax, lines)
    ax.set_title(title, fontsize=7)
    style_axis(ax, '')
    save_panel(fig, 'sex_vs_score', sub_dir)


def summary_stats(tetrad_dfs: dict[str, pd.DataFrame], nci: str, ad: str) -> pd.DataFrame:
    """Long table: one row per (tetrad, covariate column of the heatmap)."""
    columns = [
        ('MMSE', ad, 'cts_mmse30_lv'),
        ('Braak', ad, 'braaksc'),
        ('Age', nci, 'age_death'), ('Age', ad, 'age_death'),
        ('PMI', nci, 'pmi'), ('PMI', ad, 'pmi'),
        ('Sex', nci, 'msex'), ('Sex', ad, 'msex'),
    ]
    rows = []
    for tetrad_label, df in tetrad_dfs.items():
        for covariate, condition, x_column in columns:
            condition_df = df[df['condition'] == condition]
            effect, p, n = sex_test(condition_df) if x_column == 'msex' else spearman(condition_df, x_column)
            rows.append({
                'tetrad': tetrad_label, 'column': f'{covariate}\n({condition})', 'covariate': covariate, 'condition': condition,
                'effect': effect, 'effect_type': 'rank-biserial r (male vs female)' if x_column == 'msex' else 'Spearman rho',
                'p': p, 'n': n,
            })
    summary = pd.DataFrame(rows)
    tested = summary['p'].notna()
    summary['p_adj_bh_over_heatmap'] = np.nan
    summary.loc[tested, 'p_adj_bh_over_heatmap'] = stats.false_discovery_control(summary.loc[tested, 'p'], method='bh')
    return summary


def plot_summary_heatmap(summary: pd.DataFrame):
    effect = summary.pivot(index='tetrad', columns='column', values='effect')
    p = summary.pivot(index='tetrad', columns='column', values='p')
    row_order = list(dict.fromkeys(summary['tetrad']))
    column_order = list(dict.fromkeys(summary['column']))
    effect, p = effect.loc[row_order, column_order], p.loc[row_order, column_order]

    limit = max(0.3, float(np.nanmax(np.abs(effect.to_numpy()))))
    fig, ax = plt.subplots(1, 1, figsize=(args.summary_heatmap_width_in_cm * cm, (2.5 + 0.7 * len(row_order)) * cm))
    image = ax.imshow(effect.to_numpy(dtype=float), cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto')
    for i in range(len(row_order)):
        for j in range(len(column_order)):
            value = p.iat[i, j]
            ax.text(j, i, format_p_value(value), ha='center', va='center', fontsize=5, fontweight='bold' if value < 0.05 else 'normal')
    ax.set_xticks(range(len(column_order)))
    ax.set_xticklabels(column_order, fontsize=6, rotation=90)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=6)
    ax.tick_params(length=0)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    colorbar.set_label('Spearman ρ (sex: rank-biserial r)', fontsize=6)
    colorbar.ax.tick_params(labelsize=5)
    ax.set_title('Nominal p in cells (bold: p < 0.05)', fontsize=6)
    save_panel(fig, 'summary_heatmap', plot_dir)



def main():
    mnc_object = MultiNeuronChatObject.load(args.path_to_mnc_object)
    nci, ad = mnc_object.condition_names
    clinical_data = load_clinical_data()

    significant = get_significant_tetrads(mnc_object)
    print(f'{len(significant)} tetrads with {args.statistical_test} {args.p_value_correction.upper()}-adjusted p < {args.significance_threshold}:')
    print(significant.to_string(index=False))
    if len(significant) == 0:
        return

    tetrad_dfs = {}
    for row in significant.itertuples():
        title = f'{short(row.source)} → {short(row.receiver)}  {row.interaction.replace("_", "–")}'
        df = get_tetrad_df(mnc_object, clinical_data, row.source, row.receiver, row.interaction)
        tetrad_dfs[title] = df

        # Per-covariate plots: one subfolder per significant tetrad
        sub_dir = os.path.join(covariate_plot_dir, tetrad_file_stem(row.source, row.receiver, row.interaction))
        plot_mmse(df, nci, ad, title, sub_dir)
        plot_braak(df, nci, ad, title, sub_dir)
        plot_continuous_confounder(df, [nci, ad], 'age_death', 'Age at death (90 = 90+)', title, 'age_vs_score', sub_dir, jitter=0.2)
        plot_continuous_confounder(df, [nci, ad], 'pmi', 'Post-mortem interval (h)', title, 'pmi_vs_score', sub_dir)
        plot_sex(df, [nci, ad], title, sub_dir)

    # Heatmap: all significant tetrads
    summary = summary_stats(tetrad_dfs, nci, ad)
    summary.drop(columns='column').to_csv(path_to_stats_csv, index=False)
    plot_summary_heatmap(summary)
    print(f'Results written to {path_to_output_dir}')


if __name__ == '__main__':
    main()
