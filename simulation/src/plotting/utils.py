import os

import numpy as np
import pandas as pd
import xarray as xr

import pickle

from multineuronchat import MultiNeuronChatObject

from matplotlib.axes import Axes


cm: float = 1/2.54

statistical_test_short: dict[str, str] = {
    'KS': 'KS',
    'CVM': 'CVM',
    'MannWhitneyU': 'MWU'
}

statistical_test_colors: dict[str, str] = {
    'KS': '#ca0020',
    'CVM': '#92c5de',
    'MannWhitneyU': '#0571b0'
}


def extract_expected_perturbations_df(
        path_to_expected_perturbations: str,
        cell_types: list[str],
        lt_interactions: list[str],
) -> xr.DataArray:
    df = pd.read_csv(path_to_expected_perturbations)

    expected_perturbations = xr.DataArray(
        data=np.zeros((len(cell_types), len(cell_types), len(lt_interactions))),
        coords={
            'source': cell_types,
            'receiver': cell_types,
            'interaction': lt_interactions,
        },
        dims=['source', 'receiver', 'interaction'],
    )

    for idx, row in df.iterrows():
        source = row['Source']
        receiver = row['Receiver']
        ligand = row['Ligand']
        target = row['Target']

        interaction = f'{ligand}_{target}'

        expected_perturbations.loc[source, receiver, interaction] = 1

    return expected_perturbations

def get_expected_perturbations_array(
        expected_perturbations: pd.DataFrame,
        mnc_object: MultiNeuronChatObject
) -> xr.DataArray:

    reference_matrix: xr.DataArray = xr.DataArray(
        np.zeros(shape=(
            len(mnc_object.source_cell_types),
            len(mnc_object.receiver_cell_types),
            len(mnc_object.interaction_names)
        )),
        dims=[
            'source',
            'receiver',
            'interaction',
        ],
        coords={
            'source': mnc_object.source_cell_types,
            'receiver': mnc_object.receiver_cell_types,
            'interaction': mnc_object.interaction_names
        }
    )

    for source, receiver, interaction in zip(
        expected_perturbations['Source'],
        expected_perturbations['Receiver'],
        expected_perturbations['Interaction']
    ):
        reference_matrix.loc[{'source': source, 'receiver': receiver, 'interaction': interaction}] = 1

    return reference_matrix

def load_curve_data(
    path_to_curves: str,
    path_to_pr_aucs: str,
    path_to_error_bands_dict: str,
) -> tuple[dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]], dict[str, list[float]], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    curves: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]
    with open(path_to_curves, 'rb') as f:
        curves = pickle.load(f)

    pr_aucs: dict[str, list[float] | np.array]
    with open(path_to_pr_aucs, 'rb') as f:
        pr_aucs = pickle.load(f)

    error_bands_dict: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    with open(path_to_error_bands_dict, 'rb') as f:
        error_bands_dict = pickle.load(f)

    return curves, pr_aucs, error_bands_dict


def convert_mean_type_label(mean_type_label: str) -> str:
    if mean_type_label == 'mean_0':
        return 'Mean'
    elif mean_type_label == 'tri_mean_0':
        return 'Trimean'
    elif mean_type_label == 'trim_mean_0.1':
        return 'Trimmed mean 10%'
    elif mean_type_label == 'trim_mean_0.05':
        return 'Trimmed mean 5%'
    else:
        raise ValueError(f'Invalid mean type label: {mean_type_label}')


def convert_mean_type_labels(mean_type_labels: list[str]) -> list[str]:
    converted_mean_type_labels: list[str] = []

    for mean_type_label in mean_type_labels:
        converted_mean_type_labels.append(convert_mean_type_label(mean_type_label))

    return converted_mean_type_labels


def convert_cases_l2fc_label(cases_l2fc_label: str) -> str:
    if cases_l2fc_label == 'CASE_0.3':
        return 'Case log2FC 0.3'
    elif cases_l2fc_label == 'CASE_0.5':
        return 'Case log2FC 0.5'
    elif cases_l2fc_label == 'CASE_1':
        return 'Case log2FC 1.0'
    elif cases_l2fc_label == 'CASE_1.5':
        return 'Case log2FC 1.5'
    else:
        raise ValueError(f'Invalid cases l2fc label: {cases_l2fc_label}')


def convert_cases_l2fc_labels(cases_l2fc_labels: list[str]) -> list[str]:
    converted_cases_l2fc_labels: list[str] = []

    for cases_l2fc_label in cases_l2fc_labels:
        converted_cases_l2fc_labels.append(convert_cases_l2fc_label(cases_l2fc_label))

    return converted_cases_l2fc_labels


def convert_p_value_string_to_stars(p_value: float) -> str:
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s.'

def median_and_iqr_errors(values_per_x: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Median and interquartile range of the values at each x position, in the form plt.errorbar expects.

    Precision, recall and PR AUC are bounded by [0, 1] and often skewed towards a bound, so mean +- std can reach
    outside the valid range; the quartiles are actual quantiles of the data and always stay inside it.

    :param values_per_x: the finite values at each x position; an empty array gives NaN
    :return: medians, and a (2, n) array of the distances from the median down to Q1 and up to Q3 (yerr)
    """
    medians: list[float] = []
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for values in values_per_x:
        if len(values) == 0:
            medians.append(np.nan)
            lower_errors.append(np.nan)
            upper_errors.append(np.nan)
            continue
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        medians.append(float(median))
        lower_errors.append(float(median - q1))
        upper_errors.append(float(q3 - median))
    return np.array(medians), np.array([lower_errors, upper_errors])


def with_points_path(path: str) -> str:
    """
    Path of the variant of a figure that overlays the individual data points, next to the original figure.

    :param path: e.g. 'results/figs/pr_auc_comparison/CASE_1.pdf'
    :return: e.g. 'results/figs/pr_auc_comparison/CASE_1_with_points.pdf'
    """
    root, extension = os.path.splitext(path)
    return f'{root}_with_points{extension}'


def beeswarm_offsets(y: np.ndarray, diameter: float) -> np.ndarray:
    """
    Horizontal offsets that place markers of the given diameter without overlap, each as close to the centre line as
    possible (greedy, from the lowest value up). y and diameter share one unit (display pixels).

    :param y: marker positions along the value axis
    :param diameter: marker diameter
    :return: one horizontal offset per marker, in the unit of y
    """
    offsets: np.ndarray = np.zeros(len(y))
    placed: list[int] = []
    for idx in np.argsort(y, kind='stable'):
        forbidden: list[tuple[float, float]] = []
        for other in placed:
            dy: float = y[idx] - y[other]
            if abs(dy) < diameter:
                dx: float = np.sqrt(diameter ** 2 - dy ** 2)
                forbidden.append((offsets[other] - dx, offsets[other] + dx))
        candidates: list[float] = [0.0] + [bound for interval in forbidden for bound in interval]
        free: list[float] = [
            c for c in candidates if not any(lower + 1e-9 < c < upper - 1e-9 for lower, upper in forbidden)
        ]
        # Closest to the centre line; ties alternate sides so the swarm stays symmetric.
        offsets[idx] = min(free, key=lambda c: (round(abs(c), 6), (c < 0) == (len(placed) % 2 == 0)))
        placed.append(idx)
    return offsets


def draw_beeswarm(
        ax: Axes,
        x: float,
        values,
        color,
        max_half_width: float,
        marker_size: float = 2.2,
        alpha: float = 0.6,
        edgecolor='none',
        zorder: float = 1,
) -> None:
    """
    Draw the individual data points of one group as a beeswarm centred on x. Non-finite values (e.g. the precision of
    a dataset without any significant call) and, on a log axis, non-positive values are skipped.

    The swarm is laid out in display space, so call this after the axis limits and the figure layout are final (right
    before saving). The axis limits are frozen so the points do not rescale the axes.

    :param x: centre of the swarm, in data units
    :param values: the data points
    :param max_half_width: the swarm is compressed horizontally if it would extend further than this from x (data units)
    :param marker_size: marker diameter in points
    """
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())

    # Points are drawn unclipped so that values on the axis limits show in full; values outside the limits (e.g. in
    # the other panel of a broken axis) are dropped instead.
    y_lower, y_upper = sorted(ax.get_ylim())
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values >= y_lower) & (values <= y_upper)]
    if ax.get_yscale() == 'log':
        values = values[values > 0]
    if len(values) == 0:
        return

    ax.figure.draw_without_rendering()

    display: np.ndarray = ax.transData.transform(np.column_stack([np.full(len(values), x), values]))
    diameter_px: float = marker_size * ax.figure.dpi / 72
    offsets_px: np.ndarray = beeswarm_offsets(display[:, 1], diameter_px)

    max_offset_px: float = abs(ax.transData.transform((x + max_half_width, 0))[0] - display[0, 0])
    widest: float = np.max(np.abs(offsets_px))
    if widest > max_offset_px:
        offsets_px = offsets_px * max_offset_px / widest

    x_data: np.ndarray = ax.transData.inverted().transform(
        np.column_stack([display[:, 0] + offsets_px, display[:, 1]])
    )[:, 0]

    ax.scatter(
        x_data, values, s=marker_size ** 2, color=color, alpha=alpha, edgecolors=edgecolor, linewidths=0.3,
        zorder=zorder, clip_on=False,
    )
