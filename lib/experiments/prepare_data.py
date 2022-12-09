from lib.dataloader.folding import train_test_split, train_validate_splits
from lib.dataset.dta import DTA
from lib.dataset.representation import ConstantRepresentationData, RandomRepresentationData, RepresentationData
from lib.load.drug_repr import drug_repr_loaders
from lib.load.minimal_dta import load_kiba
from lib.load.target_repr import target_repr_loaders
from lib.util.anchor_util import EXPERIMENTS, KIBA
from lib.util.load_util import load
from lib.util.log_util import log_func
from lib.util.manager_util import ExperimentManager, NO_KEY
from lib.util.random_util import generator


def add_kiba(mgr: ExperimentManager):
    return mgr.choose(
        minimal_dta=load_kiba()
    )


def add_control_kiba(
        mgr: ExperimentManager,
        KIBA_CONTROL_SEED: int = 34000
):
    @load(KIBA, f'kiba_control_{KIBA_CONTROL_SEED}.pickle')
    def load_kiba_control():
        return load_kiba().remapped(KIBA_CONTROL_SEED)

    return mgr.choose(
        minimal_dta=load_kiba_control()
    )


########################################

def add_drugs(mgr: ExperimentManager, drug_reprs: list[RepresentationData]):
    return mgr.choose(
        drug_reprs=drug_reprs
    )


def add_targets(mgr: ExperimentManager, target_reprs: list[RepresentationData]):
    return mgr.choose(
        target_reprs=target_reprs
    )


def add_constant_drugs(mgr: ExperimentManager):
    return mgr.choose(
        drug_reprs=[
            ConstantRepresentationData(
                f'constant_{mgr.total_drug_width}',
                mgr.total_drug_width,
                set(mgr.minimal_dta.drug_ids.keys()))
        ]
    )


def add_constant_targets(mgr: ExperimentManager):
    return mgr.choose(
        target_reprs=[
            ConstantRepresentationData(
                f'constant_{mgr.total_target_width}',
                mgr.total_target_width,
                set(mgr.minimal_dta.target_ids.keys()))
        ]
    )


def add_random_drugs(
        mgr: ExperimentManager,
        RANDOM_DRUG_SEED: int = 30000
):
    return mgr.choose(
        drug_reprs=[
            RandomRepresentationData(
                f'random_{mgr.total_drug_width}_{RANDOM_DRUG_SEED}',
                mgr.total_drug_width,
                set(mgr.minimal_dta.drug_ids.keys()),
                RANDOM_DRUG_SEED
            )
        ]
    )


def add_random_targets(
        mgr: ExperimentManager,
        RANDOM_TARGET_SEED: int = 40000
):
    return mgr.choose(
        target_reprs=[
            RandomRepresentationData(
                f'random_{mgr.total_target_width}_{RANDOM_TARGET_SEED}',
                mgr.total_target_width,
                set(mgr.minimal_dta.target_ids.keys()),
                RANDOM_TARGET_SEED
            )
        ]
    )


########################################


def add_dta(mgr: ExperimentManager):
    return mgr.choose(
        dta=(
            NO_KEY,
            DTA.from_minimal_dta(
                mgr.minimal_dta,
                mgr.drug_reprs,
                mgr.target_reprs
            )
        )
    )


########################################

FOLD_SEED: int = 10000
SPLIT_SEED: int = 20000


@log_func
@load(EXPERIMENTS, f'common_folds_{FOLD_SEED}_{SPLIT_SEED}.pickle')
def load_common_folds():
    dta = DTA.from_minimal_dta(
        minimal_dta=load_kiba(),
        drug_reprs=[repr_() for repr_ in drug_repr_loaders],
        target_reprs=[repr_() for repr_ in target_repr_loaders]
    )
    dta_folding = train_test_split(
        list(dta.drugs),
        list(dta.targets),
        generator(FOLD_SEED)
    )

    drug_set = set(dta_folding.drug_with_repr_affinity)
    target_set = set(dta_folding.target_with_repr_affinity)
    drug_target_list = [
        (drug, target)
        for drug, target
        in dta.minimal_dta.affinities.keys
        if drug in drug_set and target in target_set
    ]

    splits = train_validate_splits(
        drug_target_list,
        generator(FOLD_SEED)
    )
    return dta_folding, splits


def add_common_folding(mgr: ExperimentManager):
    dta_folding, splits = load_common_folds()
    return mgr.choose(
        FOLD_SEED=FOLD_SEED,
        SPLIT_SEED=SPLIT_SEED,
        folding=(NO_KEY, dta_folding),
        splits=(NO_KEY, splits)
    )
