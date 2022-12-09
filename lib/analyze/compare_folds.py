from math import sqrt, ceil
from typing import Iterable, Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, cm
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, auc

from lib.analyze.predict import run_prediction
from lib.analyze.prediction import Prediction
from lib.util.manager_util import ExperimentManager
import matplotlib as mpl


def get_fold_predictions_from_experiment(
        experiment: Callable[[], Iterable[ExperimentManager]],
        drug_split_name: str,
        target_split_name: str,
        mode: str
) -> list[Prediction]:
    return [run_prediction(mgr, drug_split_name, target_split_name, mode) for mgr in experiment()]


def plot_scores_of_folds(fold_predictions: list[Prediction]):
    df = pd.concat([p.get_scores() for p in fold_predictions])

    num_axes = len(df.columns)
    col_num = ceil(sqrt(num_axes))
    row_num = ceil(num_axes / col_num)

    with mpl.rc_context({'figure.figsize': [10, 10]}):
        fig, axes = plt.subplots(row_num, col_num)

        for ax, key in zip(axes.flat, df.columns):
            values = df.loc[:, key]
            ax.bar(
                0.5 + np.arange(len(values)),
                values,
                width=1,
                edgecolor='white',
                linewidth=0.7,
                color=[cm.get_cmap('jet')(e) for e in np.linspace(0.0, 1.0, len(values))]
            )
            ax.set(
                title=key,
                xticks=[],
                ylim=(np.min(values) * 0.9, np.max(values) * 1.1)
            )

        fig.suptitle(f"Statistical features of {fold_predictions[0].experiment_name}")
        fig.subplots_adjust(wspace=0.4, hspace=0.4)
        return fig


def precision_recall_of_folds(fold_predictions: list[Prediction]):
    with mpl.rc_context({'figure.figsize': [5, 5]}):
        fig, ax = plt.subplots()

        interpolations = []
        average_precisions = []
        interpolation_base = np.linspace(0.0, 1.0, 100)

        for p in fold_predictions:
            precisions, recalls, _ = precision_recall_curve(p.ground_truth, p.predictions, pos_label=1.0)
            average_precision = average_precision_score(p.ground_truth, p.predictions, pos_label=1.0)

            ax.plot(
                precisions,
                recalls,
                lw=1,
                label=f'Fold {p.fold_num} (Average Precision = {average_precision : 0.2})',
                alpha=0.3
            )
            interpolated = np.interp(interpolation_base, precisions, recalls)
            interpolated[0] = 1.0
            interpolations.append(interpolated)
            average_precisions.append(average_precision)

        ax.plot(
            [0, 1],
            [1, 0],
            color='red',
            lw=2,
            linestyle='--',
            label='Chance',
            alpha=0.8
        )

        mean = np.mean(interpolations, axis=0)
        std = np.std(interpolations, axis=0)
        mean[-1] = 0.0

        ax.plot(
            interpolation_base,
            mean,
            color="blue",
            label=rf"Mean (Average Precision = {np.mean(average_precisions):0.2} $\pm$ {np.std(average_precisions) : 0.2})",
            lw=2,
            alpha=0.8,
        )

        ax.fill_between(
            interpolation_base,
            np.maximum(mean - std, 0),
            np.minimum(mean + std, 1),
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[0.0, 1.05],
            ylim=[0.0, 1.05],
            xlabel="Recall",
            ylabel="Precision",
            title=f"Precision recall characteristics of {fold_predictions[0].experiment_name}"
        )

        ax.legend(loc="lower left")

        return fig


def roc_of_folds(fold_predictions: list[Prediction]):
    with mpl.rc_context({'figure.figsize': [5, 5]}):
        fig, ax = plt.subplots()

        interpolations = []
        aucs = []
        interpolation_base = np.linspace(0.0, 1.0, 100)

        for p in fold_predictions:
            fp_rates, tp_rates, _ = roc_curve(p.ground_truth, p.predictions, pos_label=1.0)
            roc_auc = auc(fp_rates, tp_rates)

            ax.plot(
                fp_rates,
                tp_rates,
                lw=1,
                label=f'Fold {p.fold_num} (AUC = {roc_auc : 0.2})',
                alpha=0.3
            )
            interpolated = np.interp(interpolation_base, fp_rates, tp_rates)
            interpolated[0] = 0.0
            interpolations.append(interpolated)
            aucs.append(roc_auc)

        ax.plot(
            [0, 1],
            [0, 1],
            color='red',
            lw=2,
            linestyle='--',
            label='Chance',
            alpha=0.8
        )

        mean = np.mean(interpolations, axis=0)
        std = np.std(interpolations, axis=0)
        mean[-1] = 1.0

        ax.plot(
            interpolation_base,
            mean,
            color="blue",
            label=rf"Mean (AUC = {np.mean(aucs):0.2} $\pm$ {np.std(aucs) : 0.2})",
            lw=2,
            alpha=0.8,
        )

        ax.fill_between(
            interpolation_base,
            np.maximum(mean - std, 0),
            np.minimum(mean + std, 1),
            color="grey",
            alpha=0.2,
            label=r"$\pm$ 1 std. dev.",
        )

        ax.set(
            xlim=[0.0, 1.0],
            ylim=[0.0, 1.05],
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            title=f"Receiver operating characteristics of {fold_predictions[0].experiment_name}"
        )

        ax.legend(loc="lower right")

        return fig
