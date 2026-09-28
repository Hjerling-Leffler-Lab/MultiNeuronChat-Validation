import math

import numpy as np
import xarray as xr
import pandas as pd

from multineuronchat import MultiNeuronChatObject
from multineuronchat.masks import compute_variance_mask, compute_abundance_mask
from multineuronchat.power import minimum_detectable_p

from utils import get_expected_perturbations_array, make_test_label
from plot_precision_recall_a import top_percentile_mask, by_correct, precision_recall_from_p_values_adj

import argparse


# Prefilters evaluated on the donor-count sweep. The legacy Wasserstein filter is left out: its statistic is not
# saved locally for the sweep, and it is only kept as a comparator in the full-cohort benchmark.
n_sweep_filter_keys: list[str] = [
    'None',
    'Variance-top10',
    'Variance-min0',
    'Abundance-top10',
    'Abundance-min0',
]

statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']


def triples_needed_at_floor(p_min: float, n_hypotheses: int, alpha: float = 0.05) -> int:
    """
    Smallest number of discoveries the Benjamini-Yekutieli procedure can make in a given design.

    The k-th smallest p-value is significant only if p_(k) <= k * alpha / (m * c(m)). Since no p-value can fall below
    the permutation floor p_min, a significance call needs k >= p_min * m * c(m) / alpha: below this many triples at
    (or near) the floor, nothing can be called, whatever the effect size.

    :param p_min: uncorrected minimal attainable p-value of the design (minimum_detectable_p)
    :param n_hypotheses: number of hypotheses entering the correction (m)
    :param alpha: significance level of the corrected p-values
    :return: the minimal number of triples that must reach the floor together (at least 1)
    """
    # Harmonic factor of the BY correction, as in multineuronchat.power.minimum_detectable_p.
    correction_factor: float = float(np.sum(1 / np.arange(1, n_hypotheses + 1)))
    return max(1, int(math.ceil(p_min * n_hypotheses * correction_factor / alpha)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_mnc_object', nargs='+', required=True)
    parser.add_argument('--path_to_expected_perturbations', type=str, required=True)
    parser.add_argument('--path_to_summary_df', type=str, required=True)
    parser.add_argument('--mean_type', type=str, required=True)
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--n_donors', type=int, required=True, help='Donors per sex per group')
    parser.add_argument('--alpha', type=float, default=0.05)

    args = parser.parse_args()

    expected_perturbations: pd.DataFrame = pd.read_csv(args.path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]

    # Both sexes are sampled per group, so a group holds 2 * n_donors subjects.
    n_per_group: int = 2 * args.n_donors
    p_min, _ = minimum_detectable_p(n_per_group, n_per_group)

    rows: list[dict[str, str | int | float]] = []

    reference_matrix: xr.DataArray | None = None

    for dataset_index, mnc_path in enumerate(args.paths_to_mnc_object):
        mnc_object = MultiNeuronChatObject.load(mnc_path)

        if reference_matrix is None:
            reference_matrix = get_expected_perturbations_array(
                expected_perturbations=expected_perturbations,
                mnc_object=mnc_object,
            )

        # The label-blind prefilter statistics are recomputed from the saved object (step 7 computes them from the
        # same communication and abundance scores). They are independent of the condition labels, so recomputing
        # them does not change the p-values; only the set of hypotheses entering the BY correction.
        _, variances = compute_variance_mask(mnc_object, min_value=0.0, statistic='cv2', return_variances=True)
        _, abundances = compute_abundance_mask(mnc_object, min_value=0.0, return_abundances=True)

        masks: dict[str, xr.DataArray | None] = {
            'None': None,
            'Variance-top10': top_percentile_mask(variances, keep_fraction=0.1),
            'Variance-min0': variances > 0,
            'Abundance-top10': top_percentile_mask(abundances, keep_fraction=0.1),
            'Abundance-min0': abundances > 0,
        }

        for statistical_test in statistical_tests:
            raw_p_values: xr.DataArray = mnc_object.p_values[statistical_test]

            for filter_key in n_sweep_filter_keys:
                mask: xr.DataArray | None = masks[filter_key]
                selected: xr.DataArray = raw_p_values if mask is None else raw_p_values.where(mask)
                p_values_adj: xr.DataArray = by_correct(raw_p_values, mask=mask)

                precision, recall = precision_recall_from_p_values_adj(
                    reference_matrix=reference_matrix,
                    p_values_adj=p_values_adj,
                )

                finite: np.ndarray = np.isfinite(selected.values)
                n_hypotheses: int = int(finite.sum())
                significant: np.ndarray = (p_values_adj < args.alpha).values
                true_positive: np.ndarray = significant & (reference_matrix.values == 1)

                rows.append({
                    'Mean Type': args.mean_type,
                    'Case': args.case,
                    'N Donors Per Group': n_per_group,
                    'Dataset Index': dataset_index,
                    'Statistical Test': make_test_label(filter_key, statistical_test),
                    'Filter': filter_key,
                    'Test': statistical_test,
                    'Precision': precision,
                    'Recall': recall,
                    'N Significant': int(significant.sum()),
                    'N True Positives': int(true_positive.sum()),
                    'N Ground Truth Tested': int((finite & (reference_matrix.values == 1)).sum()),
                    'N Hypotheses': n_hypotheses,
                    'Min Raw P': float(np.nanmin(selected.values)) if n_hypotheses > 0 else np.nan,
                    'P Floor': p_min,
                    # Relative tolerance: exact p-values at the floor can differ from 2 / C(2n, n) in the last bits.
                    'N At Floor': int((selected.values[finite] <= p_min * (1 + 1e-6)).sum()),
                    'N Needed At Floor': triples_needed_at_floor(p_min, n_hypotheses, alpha=args.alpha)
                    if n_hypotheses > 0 else np.nan,
                })

        del mnc_object

    pd.DataFrame(rows).to_pickle(args.path_to_summary_df)


if __name__ == '__main__':
    main()
