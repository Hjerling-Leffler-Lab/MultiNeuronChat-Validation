import numpy as np
import pandas as pd
import xarray as xr

import pickle

from multineuronchat import MultiNeuronChatObject

import seaborn as sns


cm: float = 1/2.54

statistical_test_short: dict[str, str] = {
    'KS': 'KS',
    'Anderson': 'AD',
    'CVM': 'CVM',
    'MannWhitneyU': 'MWU'
}

statistical_test_colors: dict[str, str] = {
    'KS': '#ca0020',
    'Anderson': '#f4a582',
    'CVM': '#92c5de',
    'MannWhitneyU': '#0571b0'
}
wasserstein_statistical_test_colors: dict[str, str] = {
    x: y
    for x, y in zip(['KS', 'Anderson', 'CVM', 'MannWhitneyU'], sns.color_palette('husl', n_colors=4))
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