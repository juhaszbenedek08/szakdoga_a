import torch

from lib.load.drug_repr import drug_repr_loaders
from lib.load.target_repr import target_repr_loaders
from lib.util.manager_util import ExperimentManager, NO_KEY


def get_default_manager():
    all_drug_reprs = [loader() for loader in drug_repr_loaders]
    all_target_reprs = [loader() for loader in target_repr_loaders]
    all_drug_names = [repr_.name for repr_ in all_drug_reprs]
    all_target_names = [repr_.name for repr_ in all_target_reprs]
    total_drug_width = sum(repr_.width for repr_ in all_drug_reprs)
    total_target_width = sum(repr_.width for repr_ in all_target_reprs)

    return ExperimentManager().choose(
        device=torch.device('cuda'),
        dtype=torch.float,
        MAIN_SEED=29000,
        TRAIN_SEED=430,
        BALANCE_SEED=450,
        VALIDATE_SEED=0,
        PREDICTION_SEED=23430,
        all_drug_reprs=(all_drug_names, all_drug_reprs),
        all_target_reprs=(all_target_names, all_target_reprs),
        total_drug_width=(NO_KEY, total_drug_width),
        total_target_width=(NO_KEY, total_target_width),
        version=1,
    )
