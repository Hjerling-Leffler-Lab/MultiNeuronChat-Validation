import os
import sys

import numpy as np
import xarray as xr

# The prefilter scripts reuse the helpers of the main benchmark (src/plotting). Importing this module first puts
# that directory on the path, so `from utils import ...` and `from plot_precision_recall_a import ...` resolve there.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'plotting'))


# Prefiltering methods evaluated in the precision/recall benchmark. Keys are the label prefixes used in
# the 'Statistical Test' column of the summary dataframes (e.g. 'Variance-top10_KS_adj'); the 'None'
# baseline carries no prefix (just '<test>_adj'). Wasserstein is retained only as a deprecated legacy
# comparator: it is not independent of the test statistics under the null (Bourgon et al. 2010). The
# label-blind replacements are the squared-coefficient-of-variation ('Variance') and bottleneck
# abundance ('Abundance') filters, each shown at two retentions: 'top10' keeps the top 10% of triples,
# 'min0' keeps every triple with a strictly positive statistic (drops only untestable, all-zero triples).
filter_display_names: dict[str, str] = {
    'None': 'No filter',
    'Wasserstein': 'Wasserstein (top 10%, legacy)',
    'Variance-top10': 'Variance (top 10%)',
    'Variance-min0': 'Variance (> 0)',
    'Abundance-top10': 'Abundance (top 10%)',
    'Abundance-min0': 'Abundance (> 0)',
}

# Styling of the prefilters when they are drawn together in one panel. Hue encodes the filter family
# (colour-blind-safe categorical slots, neutral grey for the unfiltered baseline) and the retention is a
# secondary encoding, so identity never relies on colour alone: 'top10' is a solid line with filled
# markers, 'min0' a dashed line with hollow markers.
filter_colors: dict[str, str] = {
    'None': '#3d3d3a',
    'Wasserstein': '#eb6834',
    'Variance-top10': '#2a78d6',
    'Variance-min0': '#2a78d6',
    'Abundance-top10': '#1baf7a',
    'Abundance-min0': '#1baf7a',
}
filter_line_styles: dict[str, str] = {
    'None': '-',
    'Wasserstein': '-',
    'Variance-top10': '-',
    'Variance-min0': '--',
    'Abundance-top10': '-',
    'Abundance-min0': '--',
}
filter_markers: dict[str, str] = {
    'None': 'o',
    'Wasserstein': 'D',
    'Variance-top10': 's',
    'Variance-min0': 's',
    'Abundance-top10': '^',
    'Abundance-min0': '^',
}
filter_hollow_markers: set[str] = {'Variance-min0', 'Abundance-min0'}


def make_test_label(filter_key: str, statistical_test: str) -> str:
    """
    Build the 'Statistical Test' label used in the precision/recall summary dataframes from a filter key
    and a statistical test name. The unfiltered baseline ('None') carries no prefix.

    :param filter_key: one of the keys of filter_display_names
    :param statistical_test: e.g. 'KS', 'CVM', 'MannWhitneyU'
    :return: the label string, e.g. 'KS_adj' or 'Variance-top10_KS_adj'
    """
    if filter_key == 'None':
        return f'{statistical_test}_adj'
    return f'{filter_key}_{statistical_test}_adj'


def parse_test_label(label: str) -> tuple[str, str]:
    """
    Inverse of make_test_label: split a 'Statistical Test' label into its filter key and statistical
    test name. A label with no recognised filter prefix is treated as the unfiltered baseline ('None').

    :param label: e.g. 'KS_adj' or 'Variance-top10_KS_adj'
    :return: (filter_key, statistical_test), e.g. ('None', 'KS') or ('Variance-top10', 'KS')
    """
    core: str = label[:-len('_adj')] if label.endswith('_adj') else label
    for filter_key in filter_display_names:
        if filter_key == 'None':
            continue
        prefix: str = f'{filter_key}_'
        if core.startswith(prefix):
            return filter_key, core[len(prefix):]
    return 'None', core


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
