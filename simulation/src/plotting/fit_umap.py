import os

import scanpy

import numpy as np
import numpy.typing as npt

from anndata import AnnData

import loompy

import argparse


def fit_umap(ann_data: AnnData, path_to_output: str, meta_data: dict[str, npt.NDArray]):
    # normalize cell-wise data
    print('Normalizing data...')
    scanpy.pp.normalize_total(ann_data)
    scanpy.pp.log1p(ann_data)

    # compute PCA
    print('Computing PCA...')
    scanpy.pp.pca(ann_data)

    # compute UMAP
    print('Computing UMAP...')
    scanpy.pp.neighbors(ann_data)
    scanpy.tl.umap(ann_data)

    row_attrs: dict[str, npt.NDArray] = {
        'UMAP': npt.NDArray([0, 1])
    }

    print('Saving UMAP data...')
    with loompy.new(path_to_output) as new_loom:
        new_loom.add_columns(ann_data.obsm['X_umap'].T, row_attrs=row_attrs, col_attrs=meta_data)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--path_to_original_data',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_simulated_data',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_original_data_umap',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_simulated_data_umap',
        type=str,
        required=True,
    )
    parser.add_argument(
        '--path_to_combined_umap',
        type=str,
        required=True,
    )

    args = parser.parse_args()

    path_to_original_data: str = args.path_to_original_data
    path_to_simulated_data: str = args.path_to_simulated_data
    path_to_original_data_umap: str = args.path_to_original_data_umap
    path_to_simulated_data_umap: str = args.path_to_simulated_data_umap
    path_to_combined_umap: str = args.path_to_combined_umap

    np.random.seed(42)

    os.makedirs(os.path.dirname(path_to_original_data_umap), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_simulated_data_umap), exist_ok=True)
    os.makedirs(os.path.dirname(path_to_combined_umap), exist_ok=True)

    db_genes = [
        'GABRE', 'HMOX2', 'NPY1R', 'CHRND', 'HTR6', 'VIPR1', 'NPY5R', 'GJC3', 'HTR3D', 'CHRNB4', 'GLRA3',
        'GABBR1', 'CHRNG', 'DRD5', 'SSTR4', 'GAD2', 'GABRA3', 'PNOC', 'GLRA4', 'GABRA2', 'NTS', 'GRIA1',
        'RXFP2', 'RXFP1', 'HTR3B', 'OPRK1', 'PNMT', 'TAC1', 'GRIN3A', 'GRM5', 'GABRA5', 'TPH1', 'GRIK4',
        'SLC18A2', 'GJA5', 'DRD1', 'DRD3', 'CCKBR', 'GJA8', 'SLC6A4', 'ADCYAP1R1', 'GRIA4', 'GRIK3',
        'GABRA1', 'SLC17A7', 'GJB1', 'ADRA2B', 'GABRA4', 'GRIK2', 'GUCY1A1', 'GRM2', 'NPY', 'SST', 'HTR3C',
        'OPRM1', 'OPRD1', 'RLN1', 'GRIN2B', 'GJB4', 'SLC6A2', 'CHRNA5', 'GJE1', 'NOS3', 'GJD4', 'SLC18A3',
        'HTR4', 'NMB', 'GJA4', 'GLRA2', 'SSTR3', 'GLRB', 'GRIA2', 'TRHR', 'NLGN4X', 'ADRB1', 'PDYN',
        'GUCY1A2', 'CHAT', 'TH', 'CHRNA7', 'DRD4', 'CRHR1', 'GJB5', 'HTR1A', 'CHRNA1', 'GRM7', 'PENK',
        'SLC6A3', 'GJB2', 'HTR2C', 'GRIN2C', 'GJC2', 'DDC', 'GJA10', 'GJC1', 'TACR1', 'GRPR', 'HTR1F',
        'GRIN2D', 'SSTR2', 'NOS2', 'GLS', 'CHRNB2', 'TPH2', 'CHRNA3', 'CHRM1', 'GRM3', 'CHRNB1', 'GRM4',
        'GAD1', 'NMBR', 'HTR1D', 'SLC18A1', 'HTR1E', 'SHMT2', 'HMOX1', 'CHRNA10', 'NLGN1', 'ADRA1D',
        'GRIN2A', 'GJA1', 'NOS1', 'ADRA2C', 'GRP', 'SSTR1', 'ADRA1A', 'GABRD', 'HTR3E', 'DBH', 'NRXN2',
        'ACHE', 'ADCYAP1', 'PTHLH', 'GUCY1B2', 'CCK', 'GRM8', 'GRIA3', 'COMT', 'PTH1R', 'GJA3', 'NRXN1',
        'GRIN1', 'SHMT1', 'NRXN3', 'HTR2A', 'GABRA6', 'GRM6', 'GRIN3B', 'DRD2', 'ADRA2A', 'HTR1B', 'TAC3',
        'ADRB3', 'GLRA1', 'NLGN2', 'GJD2', 'GABRB3', 'NLGN4Y', 'TRH', 'GJD3', 'GABBR2', 'GRIK5', 'OPRL1',
        'CHRNA4', 'NPY2R', 'HTR3A', 'CHRM5', 'GABRG2', 'GABRQ', 'CHRM3', 'HTR5A', 'HTR7', 'CRHR2', 'CHRM4',
        'ADRB2', 'GJB3', 'CHRNB3', 'TACR3', 'SLC6A5', 'GRIK1', 'RXFP3', 'CHRNA9', 'SLC17A6', 'SLC17A8',
        'VIPR2', 'CHRNA6', 'GABRB2', 'NLGN3', 'SLC6A9', 'GRM1', 'SLC32A1', 'CHRNE', 'CRH', 'ADRA1B',
        'CHRNA2', 'HTR2B', 'NTSR1', 'CHRM2', 'GJB6', 'NTSR2', 'VIP', 'GUCY1B1', 'CORT'
    ]

    print('Loading original data...')
    with loompy.connect(path_to_original_data, mode='r') as original_loom:
        original_genes: npt.NDArray = original_loom.ra['Gene']

        genes_mask: npt.NDArray = [gene in db_genes for gene in original_genes]
        genes = original_genes[genes_mask]

        genes_sorted_idx = np.argsort(genes)

        non_removed_mask: npt.NDArray = original_loom.ca['cluster_name_15CTs'] != 'none (removed)'
        ctrl_mask: npt.NDArray = original_loom.ca['Disease'] == 'CTRL'
        mask = non_removed_mask & ctrl_mask

        n_original_ctrl_cells: int = np.sum(mask)

        sub_sampled_row_attrs = original_loom.ra['Gene'][genes_mask]
        original_data: npt.NDArray = original_loom[genes_mask, :]
        original_data = original_data[genes_sorted_idx, :]
        original_data: npt.NDArray = original_data[:, mask]

        original_meta_data: dict[str, npt.NDArray] = {
            'CellID': original_loom.ca['CellID'][mask],
            'Donor': original_loom.ca['Donor'][mask],
            'Disease': original_loom.ca['Disease'][mask],
            'cluster_name_15CTs': original_loom.ca['cluster_name_15CTs'][mask],
            'Sex': original_loom.ca['Sex'][mask],
            'Simulated': npt.NDArray(['No'] * n_original_ctrl_cells)
        }

    print('Fitting UMAP for original data...')
    ann_data_original: AnnData = AnnData(
        X=original_data.T
    )
    fit_umap(
        ann_data_original,
        path_to_original_data_umap,
        original_meta_data
    )
    del ann_data_original

    # Load simulated data
    print('Loading simulated data...')
    with loompy.connect(path_to_simulated_data, mode='r') as simulated_loom:
        n_simulated_cells: int = simulated_loom.shape[1]

        simulated_genes = simulated_loom.ra['Gene']

        genes_mask: npt.NDArray = [gene in db_genes for gene in simulated_genes]
        genes = simulated_genes[genes_mask]

        genes_sorted_idx = np.argsort(genes)

        disease_mask: npt.NDArray = simulated_loom.ca['Disease'] == 'CASE'
        ctrl_mask: npt.NDArray = simulated_loom.ca['Disease'] == 'CTRL'
        n_simulated_ctrl_cells: int = np.sum(ctrl_mask)

        if n_simulated_ctrl_cells > n_original_ctrl_cells:
            # Subsample the simulated data
            print('Subsampling the simulated data...')

            sampled_ctrl_idx: npt.NDArray = np.random.choice(np.where(ctrl_mask)[0], n_original_ctrl_cells, replace=False)
            ctrl_mask = np.zeros(n_simulated_cells, dtype=bool)
            ctrl_mask[sampled_ctrl_idx] = True

            n_simulated_ctrl_cells: int = np.sum(ctrl_mask)

        # This works! I checked it manually for the NRXN1 example
        # The order is correct!
        simulated_data: npt.NDArray = simulated_loom[genes_mask, :]
        simulated_data: npt.NDArray = simulated_data[genes_sorted_idx, :]
        simulated_data: npt.NDArray = simulated_data[:, ctrl_mask]

        simulated_meta_data: dict[str, npt.NDArray] = {
            'CellID': simulated_loom.ca['CellID'][ctrl_mask],
            'Donor': simulated_loom.ca['Donor'][ctrl_mask],
            'Disease': simulated_loom.ca['Disease'][ctrl_mask],
            'cluster_name_15CTs': simulated_loom.ca['cluster_name_15CTs'][ctrl_mask],
            'Sex': simulated_loom.ca['Sex'][ctrl_mask],
            'Simulated': npt.NDArray(['Yes'] * n_simulated_ctrl_cells)
        }

    print('Fitting UMAP for simulated data...')
    ann_data_simulated: AnnData = AnnData(
        X=simulated_data.T
    )
    fit_umap(
        ann_data_simulated,
        path_to_simulated_data_umap,
        simulated_meta_data
    )
    del ann_data_simulated

    print('Fitting UMAP to combined data...')
    joined_meta_data: dict[str, npt.NDArray] = {
        key: np.hstack([original_meta_data[key], simulated_meta_data[key]])
        for key in original_meta_data.keys()
    }
    ann_data_combined: AnnData = AnnData(
        X=np.hstack([original_data, simulated_data]).T
    )
    fit_umap(
        ann_data_combined,
        path_to_combined_umap,
        joined_meta_data
    )

    print('Done!')


if __name__ == '__main__':
    main()
