from typing import Callable, Iterable

from lib.analyze.compare_folds import get_fold_predictions_from_experiment, plot_scores_of_folds, \
    precision_recall_of_folds, roc_of_folds
from lib.analyze.single import confusion_matrix_single
from lib.util.anchor_util import PLOTS
from lib.util.log_util import log_func
from lib.util.manager_util import ExperimentManager
from lib.util.output_util import plot_and_save


@log_func
def experiment_analyzes(
        drug_split_name: str,
        target_split_name: str,
        mode: str,
        experiments: Callable[[], Iterable[Callable[[], Iterable[ExperimentManager]]]]
):
    for experiment in experiments():
        fold_predictions = get_fold_predictions_from_experiment(
            experiment,
            drug_split_name=drug_split_name,
            target_split_name=target_split_name,
            mode=mode
        )
        plot_path = PLOTS / f'{drug_split_name}_drug' / f'{target_split_name}_target' / f'{mode}_mode'
        name = fold_predictions[0].experiment_name
        for fold_prediction in fold_predictions:
            plot_and_save(plot_path, f'{name}_{fold_prediction.fold_num}', confusion_matrix_single(fold_prediction))
        plot_and_save(plot_path, f'{name}_scores', plot_scores_of_folds(fold_predictions))
        plot_and_save(plot_path, f'{name}_pr', precision_recall_of_folds(fold_predictions))
        plot_and_save(plot_path, f'{name}_roc', roc_of_folds(fold_predictions))
