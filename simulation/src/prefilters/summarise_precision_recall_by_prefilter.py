import pickle

import xarray as xr
import pandas as pd

from multineuronchat import MultiNeuronChatObject
from multineuronchat.power import minimum_detectable_p

from prefilter_utils import filter_display_names, make_test_label, top_percentile_mask
from utils import get_expected_perturbations_array
from plot_precision_recall_a import dataset_metrics

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths_to_mnc_object', nargs='+', required=True)
    parser.add_argument(
        '--paths_to_prefilter_statistics', nargs='+', required=True,
        help='compute_prefilter_statistics.py output, one per MNC object and in the same order',
    )
    parser.add_argument('--path_to_expected_perturbations', type=str, required=True)
    parser.add_argument('--path_to_summary_df', type=str, required=True)
    parser.add_argument('--statistical_test', type=str, required=True)
    parser.add_argument('--mean_type', type=str, required=True)
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--proportion', type=float, required=True)
    parser.add_argument('--n_per_group', type=int, required=True, help='Donors per group (cases = controls)')

    args = parser.parse_args()

    expected_perturbations: pd.DataFrame = pd.read_csv(args.path_to_expected_perturbations)
    expected_perturbations['Interaction'] = [
        '_'.join([ligand, target])
        for ligand, target in zip(expected_perturbations['Ligand'], expected_perturbations['Target'])
    ]

    p_min, _ = minimum_detectable_p(args.n_per_group, args.n_per_group)

    rows: list[dict[str, str | int | float]] = []

    reference_matrix: xr.DataArray | None = None

    for dataset_index, (mnc_path, statistics_path) in enumerate(
            zip(args.paths_to_mnc_object, args.paths_to_prefilter_statistics)
    ):
        mnc_object = MultiNeuronChatObject.load(mnc_path)
        with open(statistics_path, 'rb') as f:
            prefilter_statistics: dict[str, xr.DataArray] = pickle.load(f)

        if reference_matrix is None:
            reference_matrix = get_expected_perturbations_array(
                expected_perturbations=expected_perturbations,
                mnc_object=mnc_object,
            )

        # One BH correction per prefilter over the same raw p-values; only the set of hypotheses entering the
        # correction differs. The legacy Wasserstein filter and the label-blind Variance/Abundance filters are each
        # applied at two retentions: 'top10' keeps the top 10% of triples by the statistic, 'min0' keeps every triple
        # whose statistic is strictly positive (dropping only untestable, all-zero triples).
        variances: xr.DataArray = prefilter_statistics['variances']
        abundances: xr.DataArray = prefilter_statistics['abundances']
        masks: dict[str, xr.DataArray | None] = {
            'None': None,
            'Wasserstein': top_percentile_mask(prefilter_statistics['wasserstein_distances'], keep_fraction=0.1),
            'Variance-top10': top_percentile_mask(variances, keep_fraction=0.1),
            'Variance-min0': variances > 0,
            'Abundance-top10': top_percentile_mask(abundances, keep_fraction=0.1),
            'Abundance-min0': abundances > 0,
        }

        # One row per filter, in the canonical order defined in prefilter_utils.filter_display_names.
        for filter_key in filter_display_names:
            rows.append({
                'Mean Type': args.mean_type,
                'Case': args.case,
                'Proportion': args.proportion,
                'N Donors Per Group': args.n_per_group,
                'Dataset Index': dataset_index,
                'Statistical Test': make_test_label(filter_key, args.statistical_test),
                **dataset_metrics(
                    raw_p_values=mnc_object.p_values[args.statistical_test],
                    reference_matrix=reference_matrix,
                    p_min=p_min,
                    mask=masks[filter_key],
                ),
            })

        del mnc_object

    pd.DataFrame(rows).to_pickle(args.path_to_summary_df)


if __name__ == '__main__':
    main()
