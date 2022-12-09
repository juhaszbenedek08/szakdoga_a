from functools import wraps
from typing import Callable

from lib.dataset.representation import RepresentationData
from lib.experiments.prepare_model import add_common_model, add_common_scheme
from lib.experiments.prepare_manager import get_default_manager
from lib.experiments.prepare_data import add_kiba, add_dta, add_common_folding, add_drugs, add_targets, \
    add_constant_drugs, add_constant_targets, add_random_drugs, add_random_targets, add_control_kiba
from lib.load.drug_repr import drug_repr_loaders
from lib.load.target_repr import target_repr_loaders
from lib.util.manager_util import NO_KEY, ExperimentManager


def common_experiment(experiment_name: str):
    def outer(add_data: Callable[[ExperimentManager], ExperimentManager]):
        @wraps(add_data)
        def inner():
            mgr = get_default_manager()
            mgr = mgr.choose(
                experiment_name=experiment_name
            )
            mgr = add_data(mgr)
            mgr = add_dta(mgr)
            mgr = add_common_folding(mgr)

            for i, fold in enumerate(mgr.splits):
                fold = mgr.splits[i]
                local_mgr = mgr.choose(
                    fold_num=i,
                    fold=(NO_KEY, fold)
                )

                local_mgr = add_common_model(local_mgr)
                local_mgr = add_common_scheme(local_mgr)

                local_mgr.scheme()

                local_mgr = [local_mgr]
                yield local_mgr.pop()

        return inner

    return outer


@common_experiment('Main Experiment')
def main_experiment(mgr):
    mgr = add_kiba(mgr)
    mgr = add_drugs(mgr, mgr.all_drug_reprs)
    mgr = add_targets(mgr, mgr.all_target_reprs)
    return mgr


@common_experiment('Shuffled Labels Experiment')
def shuffled_labels_experiment(mgr):
    mgr = add_control_kiba(mgr)
    mgr = add_drugs(mgr, mgr.all_drug_reprs)
    mgr = add_targets(mgr, mgr.all_target_reprs)
    return mgr


@common_experiment('Constant Drug Experiment')
def constant_drug_experiment(mgr):
    mgr = add_kiba(mgr)
    mgr = add_constant_drugs(mgr)
    mgr = add_targets(mgr, mgr.all_target_reprs)
    return mgr


@common_experiment('Constant Target Experiment')
def constant_target_experiment(mgr):
    mgr = add_kiba(mgr)
    mgr = add_drugs(mgr, mgr.all_drug_reprs)
    mgr = add_constant_targets(mgr)
    return mgr


@common_experiment('Random Drug Experiment')
def random_drug_experiment(mgr):
    mgr = add_kiba(mgr)
    mgr = add_random_drugs(mgr)
    mgr = add_targets(mgr, mgr.all_target_reprs)
    return mgr


@common_experiment('Random Target Experiment')
def random_target_experiment(mgr):
    mgr = add_kiba(mgr)
    mgr = add_drugs(mgr, mgr.all_drug_reprs)
    mgr = add_random_targets(mgr)
    return mgr


def drug_exclusion_experiment(drug_repr: RepresentationData):
    @common_experiment(f'Excluded {drug_repr.name} Experiment')
    def helper(mgr):
        mgr = add_kiba(mgr)
        mgr = add_drugs(mgr, [repr_
                              for repr_
                              in mgr.all_drug_reprs
                              if repr_.name != drug_repr.name])
        mgr = add_targets(mgr, mgr.all_target_reprs)
        return mgr

    return helper


def target_exclusion_experiment(target_repr: RepresentationData):
    @common_experiment(f'Excluded {target_repr.name} Experiment')
    def helper(mgr):
        mgr = add_kiba(mgr)
        mgr = add_drugs(mgr, mgr.all_drug_reprs)
        mgr = add_targets(mgr, [repr_
                                for repr_
                                in mgr.all_target_reprs
                                if repr_.name != target_repr.name])
        return mgr

    return helper


def single_drug_experiment(drug_repr: RepresentationData):
    @common_experiment(f'Only {drug_repr.name} Experiment')
    def helper(mgr):
        mgr = add_kiba(mgr)
        mgr = add_drugs(mgr, [drug_repr])
        mgr = add_targets(mgr, mgr.all_target_reprs)
        return mgr

    return helper


def single_target_experiment(target_repr: RepresentationData):
    @common_experiment(f'Only {target_repr.name} Experiment')
    def helper(mgr):
        mgr = add_kiba(mgr)
        mgr = add_drugs(mgr, mgr.all_drug_reprs)
        mgr = add_targets(mgr, [target_repr])
        return mgr

    return helper


def all_experiments():
    yield main_experiment

    for loader in drug_repr_loaders:
        yield drug_exclusion_experiment(loader())
        # yield single_drug_experiment(loader())
    for loader in target_repr_loaders:
        yield target_exclusion_experiment(loader())
        # yield single_target_experiment(loader())

    # yield constant_drug_experiment
    # yield constant_target_experiment
    yield shuffled_labels_experiment
    yield random_drug_experiment
    yield random_target_experiment
