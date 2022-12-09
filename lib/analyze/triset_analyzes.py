import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from numpy import ndarray
from sklearn.cluster import SpectralBiclustering

from lib.dataloader.folding import DTAFolding, Fold
from lib.dataset.affinity import AffinityData
from lib.dataset.dta import DTA
from lib.dataset.minimal_dta import MinimalDTA
from lib.dataset.representation import RepresentationData
from lib.util.anchor_util import PLOTS
from lib.util.log_util import log_func
from lib.util.output_util import plot_and_save


def report_one(report: dict):
    for title, value in report.items():
        print(f'{title : <30} {round(value, 4) : >8}')


def report_two(report_1: dict, report_2: dict):
    for title, value_1 in report_1.items():
        value_2 = report_2[title]
        ratio = np.nan if value_1 == 0 else round(value_2 / value_1 * 100)
        print(f'{title : <30} {value_1 : >10} -> {value_2 : <10} = {ratio : >4} %')


###########################################################

@log_func
def analyze_fusions(
        minimal_dta: MinimalDTA,
        drug_reprs: list[RepresentationData],
        target_reprs: list[RepresentationData]
):
    dataset_report = minimal_dta.triset.report(minimal_dta.affinities)

    print(f'Complete fusion statistics:')
    dta = DTA.from_minimal_dta(minimal_dta, drug_reprs=drug_reprs, target_reprs=target_reprs)
    fusion = dta.triset

    report_two(dataset_report, fusion.report(minimal_dta.affinities))
    print(f'Total drug width: {dta.drug_width}')
    print(f'Total target width: {dta.target_width}')
    for repr_ in drug_reprs:
        print(f'Contribution of drug repr "{repr_.name}"')
        fusion = DTA.from_minimal_dta(
            minimal_dta,
            drug_reprs=[r for r in drug_reprs if r != repr_],
            target_reprs=target_reprs
        ).triset
        report_two(dataset_report, fusion.report(minimal_dta.affinities))
        print(f'Width: {repr_.width}')
    for repr_ in target_reprs:
        print(f'Contribution of target repr "{repr_.name}"')
        fusion = DTA.from_minimal_dta(
            minimal_dta,
            drug_reprs=drug_reprs,
            target_reprs=[r for r in target_reprs if r != repr_]
        ).triset
        report_two(dataset_report, fusion.report(minimal_dta.affinities))
        print(f'Width: {repr_.width}')

    # TODO export to file


@log_func
def analyze_distribution(
        name: str,
        minimal_dta: MinimalDTA,
        partition_num: int = 5
):
    triset = minimal_dta.triset
    drug_labels = {value: key for key, value in enumerate(triset.drugs)}
    target_labels = {value: key for key, value in enumerate(triset.targets)}

    affinity_array = np.zeros((triset.drug_num, triset.target_num))

    for drug, target in triset.affinities:
        affinity_array[drug_labels[drug], target_labels[target]] = 1

    with mpl.rc_context({'figure.figsize': [10, 10]}):
        fig, ax = plt.subplots()
        ax.imshow(affinity_array, aspect='auto')
        ax.set(title=f'Distribution of dataset "{name}"')
        plot_and_save(PLOTS, f'{name}_distribution', fig)

        model = SpectralBiclustering(n_clusters=(partition_num, partition_num), method="log", random_state=0)
        model.fit(affinity_array)

        fit_array = affinity_array[np.argsort(model.row_labels_)][:, np.argsort(model.column_labels_)]
        fig, ax = plt.subplots()
        ax.imshow(fit_array, aspect='auto')
        ax.set(title=f'Reordered distribution of dataset "{name}" ({partition_num ** 2} cells)')
        plot_and_save(PLOTS, f'{name}_distribution_reordered_{partition_num}', fig)


def plot_matrix(
        title: str,
        column_title: str,
        row_title: str,
        column_labels: list[str],
        row_labels: list[str],
        array: ndarray,
        format_string: str
):
    with mpl.rc_context({'figure.figsize': [10, 10]}):
        fig, ax = plt.subplots()

        im_ = ax.imshow(array, interpolation='nearest', cmap=plt.cm.Blues)
        min_, max_ = im_.cmap(0), im_.cmap(1.0)
        threshold = (np.max(array) + np.min(array)) / 2.0

        for i, row_label in enumerate(row_labels):
            for j, column_label in enumerate(column_labels):
                ax.text(
                    j, i,
                    format(array[i, j], format_string),
                    ha="center",
                    va="center",
                    color=max_ if array[i, j] < threshold else min_
                )
        fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(len(column_labels)),
            yticks=np.arange(len(row_labels)),
            xticklabels=column_labels,
            yticklabels=row_labels,
            ylabel=row_title,
            xlabel=column_title,
            title=title,
            ylim=(len(row_labels) - 0.5, -0.5),
            xlim=(len(column_labels) - 0.5, -0.5)
        )
        plt.setp(ax.get_yticklabels(), rotation='vertical')
        return fig


@log_func
def analyze_folding(
        name: str,
        folding: tuple[DTAFolding, list[Fold]],
        affinities: AffinityData
):
    dta_folding = folding[0]

    row_title = 'Drugs'
    column_title = 'Targets'
    row_labels = ['Train/Validate', 'Only representation', 'De novo']
    column_labels = row_labels

    areas = []
    positives = []
    measurements = []
    for drugs in [
        set(dta_folding.drug_with_repr_affinity),
        set(dta_folding.drug_with_repr),
        set(dta_folding.drug_with_nothing)
    ]:
        for targets in [
            set(dta_folding.target_with_repr_affinity),
            set(dta_folding.target_with_repr),
            set(dta_folding.target_with_nothing)
        ]:
            areas.append(len(drugs) * len(targets))
            positive = 0
            measurement = 0
            for drug, target in affinities.true:
                if drug in drugs and target in targets:
                    positive += 1
                    measurement += 1
            for drug, target in affinities.false:
                if drug in drugs and target in targets:
                    measurement += 1
            positives.append(positive)
            measurements.append(measurement)

    all_area = sum(area for area in areas)

    plot_and_save(PLOTS, f'{name}_region_num', plot_matrix(
        'Size of regions',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array(areas).reshape((3, 3)),
        'd'
    ))

    plot_and_save(PLOTS, f'{name}_measurements_num', plot_matrix(
        'Number of measurements',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array(measurements).reshape((3, 3)),
        'd'
    ))

    plot_and_save(PLOTS, f'{name}_positives_num', plot_matrix(
        'Number of positives',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array(positives).reshape((3, 3)),
        'd'
    ))

    plot_and_save(PLOTS, f'{name}_region_num', plot_matrix(
        'Ratio of regions',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array([area / all_area for area in areas]).reshape((3, 3)),
        '.2%'
    ))

    plot_and_save(PLOTS, f'{name}_measurements_ratio', plot_matrix(
        'Ratio of measurements / region areas',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array([measurement / area for measurement, area in zip(measurements, areas)]).reshape((3, 3)),
        '.2%'
    ))

    plot_and_save(PLOTS, f'{name}_positives_ratio', plot_matrix(
        'Ratio of positives / region areas',
        column_title,
        row_title,
        column_labels,
        row_labels,
        np.array([positive / area for positive, area in zip(positives, areas)]).reshape((3, 3)),
        '.2%'
    ))
