from lib.analyze.main_analyzes import experiment_analyzes
from lib.analyze.triset_analyzes import analyze_distribution
from lib.analyze.triset_analyzes import analyze_folding
from lib.analyze.triset_analyzes import analyze_fusions
from lib.experiments.main_experiments import all_experiments, main_experiment, drug_exclusion_experiment, \
    target_exclusion_experiment
from lib.experiments.prepare_data import load_common_folds
from lib.load.drug_repr import drug_repr_loaders
from lib.load.minimal_dta import load_kiba
from lib.load.target_repr import target_repr_loaders
from lib.util.load_util import free


def ensure_kiba():
    load_kiba()
    print(f'Kiba is ready')


def ensure_loading():
    for loader in drug_repr_loaders:
        print(f'Drug loader {loader().name} is ready')

    for loader in target_repr_loaders:
        print(f'Target loader {loader().name} is ready')


def analyze():
    analyze_fusions(
        load_kiba(),
        list(loader() for loader in drug_repr_loaders),
        list(loader() for loader in target_repr_loaders)
    )
    analyze_distribution('kiba', load_kiba())
    analyze_folding('common_folding', load_common_folds(), load_kiba().affinities)


def execute_experiments():
    for i, experiment in enumerate(all_experiments()):
        for j, fold_mgr in enumerate(experiment()):
            print(f'Experiment {fold_mgr.experiment_name}, fold {j} done')
            del fold_mgr
            free()
        del experiment
        free()


def analyze_experiments():
    MODE = 'raw'  # or 'balanced' or 'all'
    # Drugs and targets with representations and affinities
    experiment_analyzes('train_validate', 'train_validate', MODE, all_experiments)
    # Drugs and targets with only representations
    experiment_analyzes('represented', 'represented', MODE, all_experiments)
    # De novo drugs and targets
    experiment_analyzes('de_novo', 'de_novo', MODE, all_experiments)


if __name__ == '__main__':
    ensure_kiba()
    ensure_loading()
    analyze()
    execute_experiments()
    analyze_experiments()
