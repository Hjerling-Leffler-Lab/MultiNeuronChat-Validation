import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--path_to_clinical_data',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_437_cell_type_annotations',
    type=str,
    required=True,
)
parser.add_argument(
    '--path_to_output_figure',
    type=str,
    required=True,
)

parser.add_argument(
    '--width_in_cm',
    type=float,
    default=5.0,
    help='Width of the output figure in centimeters',
)
parser.add_argument(
    '--height_in_cm',
    type=float,
    default=5.0,
    help='Height of the output figure in centimeters',
)

args = parser.parse_args()

path_to_clinical_data: str = args.path_to_clinical_data
path_to_437_cell_type_annotations: str = args.path_to_437_cell_type_annotations
path_to_output_figure: str = args.path_to_output_figure

width_in_cm: float = args.width_in_cm
height_in_cm: float = args.height_in_cm

def cm_to_inches(cm: float) -> float:
    """Convert centimeters to inches."""
    return cm / 2.54

def main():
    clinical_data: pd.DataFrame = pd.read_csv(path_to_clinical_data)
    cell_type_annotations: pd.DataFrame = pd.read_csv(path_to_437_cell_type_annotations)

    included_donors: set[str] = set(cell_type_annotations['individualID'])

    clinical_data = clinical_data[clinical_data['individualID'].isin(included_donors)]

    fig, axs = plt.subplots(1, 1, figsize=(cm_to_inches(width_in_cm), cm_to_inches(height_in_cm)))

    colors = sns.color_palette('pastel', 6)
    stage_to_name = {
        1.0: 'NCI',
        2.0: 'MCI',
        3.0: 'MCI+',
        4.0: 'AD',
        5.0: 'AD+',
        6.0: 'Others'
    }

    # Stacked bar plot of cogdx
    cogdx = clinical_data['cogdx'].value_counts()

    cogdx.plot(
        kind='bar',
        stacked=True,
        color=colors,
        ax=axs
    )
    axs.set_xticklabels([stage_to_name[x] for x in cogdx.index])
    axs.set_xlabel('Diagnosis')
    axs.set_ylabel('Number of participants')
    axs.set_title('Diagnosis distribution')

    fig.tight_layout()
    fig.savefig(path_to_output_figure)
    plt.close(fig)


if __name__ == '__main__':
    main()
