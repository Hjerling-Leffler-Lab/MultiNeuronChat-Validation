import os
import pickle
import warnings

import argparse

import xarray as xr

from multineuronchat import MultiNeuronChatObject
from multineuronchat.masks import (
    compute_wasserstein_mask,
    compute_variance_mask,
    compute_abundance_mask,
)


def main():
    parser = argparse.ArgumentParser(
        description='Per-triple prefilter statistics of one saved MultiNeuronChat object (step 7 output).'
    )
    parser.add_argument('--path_to_mnc_object', type=str, required=True)
    parser.add_argument('--path_to_prefilter_statistics', type=str, required=True)
    args = parser.parse_args()

    mnc_object: MultiNeuronChatObject = MultiNeuronChatObject.load(args.path_to_mnc_object)

    # Legacy Wasserstein prefilter, kept only for the revision comparison. The distance is not independent of the
    # significance-test statistics under the null (Bourgon, Gentleman & Huber 2010), so it is deprecated as a
    # prefilter in the package; the DeprecationWarning is silenced deliberately.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        mask_xr, wasserstein_distances_xr = compute_wasserstein_mask(
            mnc_object,
            top_percentile=99,
            p=10,
            normalized=False,
            exclude_zero_distributions=True,
            return_wasserstein_distances=True,
        )

    # Label-blind replacements (independent of the test statistics under the null). The raw per-triple statistics
    # are saved rather than a mask at a fixed retention, so the benchmark can apply both the top-10% and the > 0
    # cutoffs. compute_variance_mask uses the squared coefficient of variation ('cv2') of the pooled communication
    # scores; compute_abundance_mask uses the bottleneck (min) of the pooled mean ligand and target abundances.
    _, variances_xr = compute_variance_mask(
        mnc_object,
        min_value=0.0,
        statistic='cv2',
        return_variances=True,
    )
    _, abundances_xr = compute_abundance_mask(
        mnc_object,
        min_value=0.0,
        return_abundances=True,
    )

    prefilter_statistics: dict[str, xr.DataArray] = {
        'mask': mask_xr,
        'wasserstein_distances': wasserstein_distances_xr,
        'variances': variances_xr,
        'abundances': abundances_xr,
    }

    os.makedirs(os.path.dirname(args.path_to_prefilter_statistics), exist_ok=True)
    with open(args.path_to_prefilter_statistics, 'wb') as f:
        pickle.dump(prefilter_statistics, f)


if __name__ == '__main__':
    main()
