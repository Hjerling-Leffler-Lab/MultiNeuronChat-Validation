import xarray as xr
import pandas as pd

from multineuronchat import MultiNeuronChatObject
from multineuronchat.power import minimum_detectable_p

from utils import get_expected_perturbations_array
from plot_precision_recall_a import dataset_metrics

import argparse


statistical_tests: list[str] = ['KS', 'CVM', 'MannWhitneyU']


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

        for statistical_test in statistical_tests:
            rows.append({
                'Mean Type': args.mean_type,
                'Case': args.case,
                'N Donors Per Group': n_per_group,
                'Dataset Index': dataset_index,
                'Statistical Test': f'{statistical_test}_adj',
                'Test': statistical_test,
                **dataset_metrics(
                    raw_p_values=mnc_object.p_values[statistical_test],
                    reference_matrix=reference_matrix,
                    p_min=p_min,
                    alpha=args.alpha,
                ),
            })

        del mnc_object

    pd.DataFrame(rows).to_pickle(args.path_to_summary_df)


if __name__ == '__main__':
    main()
