import math

import numpy as np
import xarray as xr
import pandas as pd

from scipy import stats

from multineuronchat import MultiNeuronChatObject
from multineuronchat.power import minimum_detectable_p

from utils import get_expected_perturbations_array

import argparse


def compute_confusion_matrix(
        reference_matrix: xr.DataArray,
        test_matrix: xr.DataArray
) -> xr.DataArray:
    confusion_matrix: xr.DataArray = xr.DataArray(
        np.zeros((2, 2)),
        dims=('actual', 'predicted'),
        coords={
            'actual': ['positive', 'negative'],
            'predicted': ['positive', 'negative']
        }
    )

    confusion_matrix.loc[{'actual': 'positive', 'predicted': 'positive'}] = np.sum(
        np.logical_and(reference_matrix == 1, test_matrix == 1)
    )
    confusion_matrix.loc[{'actual': 'positive', 'predicted': 'negative'}] = np.sum(
        np.logical_and(reference_matrix == 1, test_matrix == 0)
    )
    confusion_matrix.loc[{'actual': 'negative', 'predicted': 'positive'}] = np.sum(
        np.logical_and(reference_matrix == 0, test_matrix == 1)
    )
    confusion_matrix.loc[{'actual': 'negative', 'predicted': 'negative'}] = np.sum(
        np.logical_and(reference_matrix == 0, test_matrix == 0)
    )

    return confusion_matrix


def precision_recall_from_p_values_adj(
        reference_matrix: xr.DataArray,
        p_values_adj: xr.DataArray,
) -> tuple[float, float]:
    """
    Compute precision and recall of the significant triples (adjusted p < 0.05) against the ground truth.

    :param reference_matrix: the ground-truth perturbation matrix (1 for perturbed triples)
    :param p_values_adj: BH-corrected p-values for one statistical test
    :return: (precision, recall)
    """
    confusion_matrix: xr.DataArray = compute_confusion_matrix(
        reference_matrix=reference_matrix,
        test_matrix=p_values_adj < 0.05,
    )

    true_positives: xr.DataArray = confusion_matrix.loc[{'actual': 'positive', 'predicted': 'positive'}]

    precision: xr.DataArray = true_positives / np.sum(confusion_matrix.loc[{'predicted': 'positive'}])
    recall: xr.DataArray = true_positives / np.sum(confusion_matrix.loc[{'actual': 'positive'}])

    return float(precision.values), float(recall.values)


def fdr_correct(raw_p_values: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """
    Benjamini-Hochberg correction of a p-value cube, optionally restricted to a prefilter mask.

    The prefilter only affects which triples enter the correction: applying it sets the non-selected
    triples to NaN, so the correction runs over fewer hypotheses (a smaller denominator, hence more
    power). It does not change any triple's raw p-value, so we reuse the raw p-values computed once in
    step 7 rather than recomputing significance per filter. This matches the package's own correction
    (MultiNeuronChatObject.__correct_p_values), which drops NaNs before scipy's false_discovery_control.

    :param raw_p_values: uncorrected per-triple p-values (NaN where untestable), dims (source, receiver, interaction)
    :param mask: boolean DataArray selecting the triples to keep; None keeps every testable triple
    :return: BH-corrected p-values, NaN outside the retained/testable set
    """
    selected: xr.DataArray = raw_p_values if mask is None else raw_p_values.where(mask)

    values: np.ndarray = selected.values
    corrected: np.ndarray = np.full_like(values, np.nan)

    finite: np.ndarray = ~np.isnan(values)
    if np.any(finite):
        corrected[finite] = stats.false_discovery_control(values[finite], method='bh')

    return xr.DataArray(corrected, dims=selected.dims, coords=selected.coords)


def triples_needed_at_floor(p_min: float, n_hypotheses: int, alpha: float = 0.05) -> int:
    """
    Smallest number of discoveries the Benjamini-Hochberg procedure can make in a given design.

    The k-th smallest p-value is significant only if p_(k) <= k * alpha / m. Since no p-value can fall below the
    permutation floor p_min, a significance call needs k >= p_min * m / alpha: below this many triples at
    (or near) the floor, nothing can be called, whatever the effect size.

    :param p_min: uncorrected minimal attainable p-value of the design (minimum_detectable_p)
    :param n_hypotheses: number of hypotheses entering the correction (m)
    :param alpha: significance level of the corrected p-values
    :return: the minimal number of triples that must reach the floor together (at least 1)
    """
    return max(1, int(math.ceil(p_min * n_hypotheses / alpha)))


def dataset_metrics(
        raw_p_values: xr.DataArray,
        reference_matrix: xr.DataArray,
        p_min: float,
        mask: xr.DataArray | None = None,
        alpha: float = 0.05,
) -> dict[str, int | float]:
    """
    Precision, recall and the p-value-floor counts of one dataset and one statistical test.

    :param raw_p_values: uncorrected per-triple p-values (NaN where untestable)
    :param reference_matrix: the ground-truth perturbation matrix (1 for perturbed triples)
    :param p_min: minimal attainable p-value of the design (minimum_detectable_p)
    :param mask: optional prefilter restricting the triples that enter the BH correction
    :param alpha: significance level of the corrected p-values
    :return: one row of the precision/recall summary
    """
    selected: xr.DataArray = raw_p_values if mask is None else raw_p_values.where(mask)
    p_values_adj: xr.DataArray = fdr_correct(raw_p_values, mask=mask)

    precision, recall = precision_recall_from_p_values_adj(
        reference_matrix=reference_matrix,
        p_values_adj=p_values_adj,
    )

    finite: np.ndarray = np.isfinite(selected.values)
    n_hypotheses: int = int(finite.sum())
    significant: np.ndarray = (p_values_adj < alpha).values
    true_positive: np.ndarray = significant & (reference_matrix.values == 1)

    return {
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
        'N Needed At Floor': triples_needed_at_floor(p_min, n_hypotheses, alpha=alpha) if n_hypotheses > 0 else np.nan,
    }


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paths_to_mnc_object',
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--path_to_expected_perturbations',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_summary_df',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--statistical_test',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--mean_type',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--case',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--proportion',
        type=float,
        required=True,
    )
    parser.add_argument(
        '--n_per_group',
        type=int,
        required=True,
        help='Donors per group (cases = controls)',
    )

    args = parser.parse_args()

    expected_perturbations: pd.DataFrame = pd.read_csv(args.path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]

    p_min, _ = minimum_detectable_p(args.n_per_group, args.n_per_group)

    rows: list[dict[str, str | int | float]] = []

    reference_matrix: xr.DataArray | None = None

    for dataset_index, mnc_path in enumerate(args.paths_to_mnc_object):
        mnc_object = MultiNeuronChatObject.load(mnc_path)

        if reference_matrix is None:
            reference_matrix = get_expected_perturbations_array(
                expected_perturbations=expected_perturbations,
                mnc_object=mnc_object,
            )

        rows.append({
            'Mean Type': args.mean_type,
            'Case': args.case,
            'Proportion': args.proportion,
            'N Donors Per Group': args.n_per_group,
            'Dataset Index': dataset_index,
            'Statistical Test': f'{args.statistical_test}_adj',
            'Test': args.statistical_test,
            **dataset_metrics(
                raw_p_values=mnc_object.p_values[args.statistical_test],
                reference_matrix=reference_matrix,
                p_min=p_min,
            ),
        })

        del mnc_object

    pd.DataFrame(rows).to_pickle(args.path_to_summary_df)


if __name__ == '__main__':
    main()
