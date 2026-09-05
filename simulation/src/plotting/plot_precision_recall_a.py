import pickle

import numpy as np
import xarray as xr
import pandas as pd

from scipy import stats

from multineuronchat import MultiNeuronChatObject

from utils import get_expected_perturbations_array, filter_display_names, make_test_label

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


def top_percentile_mask(statistic: xr.DataArray, keep_fraction: float = 0.1) -> xr.DataArray:
    """
    Build a boolean mask keeping the triples with the largest statistic values, at the requested
    retained fraction. NaN triples (untestable) never pass, matching the package mask semantics.

    :param statistic: per-triple statistic with dims (source, receiver, interaction), NaN where undefined
    :param keep_fraction: fraction of triples to keep (0.1 keeps the top 10%)
    :return: boolean DataArray, True for the retained triples
    """
    threshold: float = np.nanpercentile(statistic, 100 * (1 - keep_fraction))
    return statistic >= threshold


def precision_recall_from_p_values_adj(
        reference_matrix: xr.DataArray,
        p_values_adj: xr.DataArray,
) -> tuple[float, float]:
    """
    Compute precision and recall of the significant triples (adjusted p < 0.05) against the ground truth.

    :param reference_matrix: the ground-truth perturbation matrix (1 for perturbed triples)
    :param p_values_adj: BY-corrected p-values for one statistical test
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


def by_correct(raw_p_values: xr.DataArray, mask: xr.DataArray | None = None) -> xr.DataArray:
    """
    Benjamini-Yekutieli correction of a p-value cube, optionally restricted to a prefilter mask.

    The prefilter only affects which triples enter the correction: applying it sets the non-selected
    triples to NaN, so the correction runs over fewer hypotheses (a smaller denominator, hence more
    power). It does not change any triple's raw p-value, so we reuse the raw p-values computed once in
    step 7 rather than recomputing significance per filter. This matches the package's own correction
    (MultiNeuronChatObject.__correct_p_values), which drops NaNs before scipy's false_discovery_control.

    :param raw_p_values: uncorrected per-triple p-values (NaN where untestable), dims (source, receiver, interaction)
    :param mask: boolean DataArray selecting the triples to keep; None keeps every testable triple
    :return: BY-corrected p-values, NaN outside the retained/testable set
    """
    selected: xr.DataArray = raw_p_values if mask is None else raw_p_values.where(mask)

    values: np.ndarray = selected.values
    corrected: np.ndarray = np.full_like(values, np.nan)

    finite: np.ndarray = ~np.isnan(values)
    if np.any(finite):
        corrected[finite] = stats.false_discovery_control(values[finite], method='by')

    return xr.DataArray(corrected, dims=selected.dims, coords=selected.coords)


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--paths_to_mnc_object',
        nargs='+',
        required=True,
    )
    parser.add_argument(
        '--paths_to_wasserstein_object',
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

    args = parser.parse_args()

    paths_to_mnc_object: str = args.paths_to_mnc_object
    paths_to_wasserstein_object: str = args.paths_to_wasserstein_object
    path_to_expected_perturbations: str = args.path_to_expected_perturbations
    path_to_summary_df: str = args.path_to_summary_df

    statistical_test: str = args.statistical_test

    mean_type = args.mean_type
    case = args.case
    proportion = args.proportion

    expected_perturbations: pd.DataFrame = pd.read_csv(path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]
    reference_mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(paths_to_mnc_object[0])
    reference_matrix: xr.DataArray = get_expected_perturbations_array(
        expected_perturbations=expected_perturbations,
        mnc_object=reference_mnc_object
    )
    del reference_mnc_object


    precision_recall_dict: dict[str, list[str | float]] = {
        'Mean Type': [],
        'Case': [],
        'Proportion': [],
        'Statistical Test': [],
        'Precision': [],
        'Recall': []
    }

    for mnc_path, wasserstein_path in zip(paths_to_mnc_object, paths_to_wasserstein_object):
        mnc_object = MultiNeuronChatObject.load(mnc_path)
        wasserstein_object: dict[str, xr.DataArray] = pickle.load(
            open(
                wasserstein_path,
                'rb'
            )
        )

        # Build one BY-corrected p-value cube per prefilter. 'None' is the unfiltered baseline, which was
        # already computed and corrected in step 7, so it is reused directly. The legacy Wasserstein filter
        # and the label-blind Variance/Abundance filters are each applied at two retentions: 'top10' keeps
        # the top 10% of triples by the statistic, 'min0' keeps every triple whose statistic is strictly
        # positive (dropping only untestable, all-zero triples).
        wasserstein_distances: xr.DataArray = wasserstein_object['wasserstein_distances']
        variances: xr.DataArray = wasserstein_object['variances']
        abundances: xr.DataArray = wasserstein_object['abundances']

        # 'None' keeps every testable triple; the other filters restrict the correction to a subset.
        # The raw p-values are the same for every filter (reused from step 7); only the BY-correction
        # denominator differs.
        masks: dict[str, xr.DataArray | None] = {
            'None': None,
            'Wasserstein': top_percentile_mask(wasserstein_distances, keep_fraction=0.1),
            'Variance-top10': top_percentile_mask(variances, keep_fraction=0.1),
            'Variance-min0': variances > 0,
            'Abundance-top10': top_percentile_mask(abundances, keep_fraction=0.1),
            'Abundance-min0': abundances > 0,
        }

        raw_p_values: xr.DataArray = mnc_object.p_values[statistical_test]

        p_values_adj_per_filter: dict[str, xr.DataArray] = {
            filter_key: by_correct(raw_p_values, mask=mask)
            for filter_key, mask in masks.items()
        }

        # Append one row per filter, in the canonical order defined in utils.filter_display_names.
        for filter_key in filter_display_names:
            precision, recall = precision_recall_from_p_values_adj(
                reference_matrix=reference_matrix,
                p_values_adj=p_values_adj_per_filter[filter_key],
            )

            precision_recall_dict['Mean Type'].append(mean_type)
            precision_recall_dict['Case'].append(case)
            precision_recall_dict['Proportion'].append(proportion)
            precision_recall_dict['Statistical Test'].append(make_test_label(filter_key, statistical_test))
            precision_recall_dict['Precision'].append(precision)
            precision_recall_dict['Recall'].append(recall)

    precision_recall_df: pd.DataFrame = pd.DataFrame(precision_recall_dict)
    precision_recall_df.to_pickle(path_to_summary_df)

if __name__ == '__main__':
    main()