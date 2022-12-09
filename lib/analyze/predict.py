import torch

from lib.dataloader.affinity_balancer import all_possible_drug_targets
from lib.dataloader.raw_dataloaders import simple_predictor_dataloader
from lib.util.log_util import log_func
from lib.util.learner_util import predict
from lib.util.manager_util import ExperimentManager
from lib.analyze.prediction import Prediction
from lib.util.random_util import generator


@log_func
def run_prediction(
        mgr: ExperimentManager,
        drug_split_name: str,
        target_split_name: str,
        mode: str
):
    mgr.initialize()

    prediction = mgr.get_prediction(drug_split_name, target_split_name, mode)
    if prediction is not None:
        return prediction

    if drug_split_name == 'train_validate':
        drug_list = mgr.folding.drug_with_repr_affinity
    elif drug_split_name == 'represented':
        drug_list = mgr.folding.drug_with_repr
    elif drug_split_name == 'de_novo':
        drug_list = mgr.folding.drug_with_nothing
    else:
        raise

    if target_split_name == 'train_validate':
        target_list = mgr.folding.target_with_repr_affinity
    elif target_split_name == 'represented':
        target_list = mgr.folding.target_with_repr
    elif target_split_name == 'de_novo':
        target_list = mgr.folding.target_with_nothing
    else:
        raise

    if mode == 'raw':
        drug_target_list = list(mgr.dta.affinities)
    elif mode == 'balanced':
        drug_set = set(drug_list)
        target_set = set(target_list)
        drug_target_list = [
            (drug, target)
            for drug, target
            in mgr.dta.affinities.true
            if drug in drug_set and target in target_set
        ]
    elif mode == 'all':
        drug_target_list = all_possible_drug_targets(
            drug_list,
            target_list
        )
    else:
        raise

    predictions, ground_truth = predict(
        mgr.network.module.online_predictor_with_sigmoid,
        simple_predictor_dataloader(
            drug_map=mgr.dta.fused_balanced_drug_cacher,
            target_map=mgr.dta.fused_balanced_target_cacher,
            affinities=mgr.dta.minimal_dta.affinities,
            drug_target_list=drug_target_list,
            batch_size=mgr.score_batch_size,
            generator=generator(mgr.PREDICTION_SEED),
            device=mgr.device,
            dtype=mgr.dtype
        ),
        mgr.score_batch_size
    )

    prediction = Prediction(
        mgr.experiment_name,
        mgr.fold_num,
        drug_split_name,
        target_split_name,
        mode,
        predictions.to(torch.device('cpu')).numpy(),
        ground_truth.to(torch.device('cpu')).numpy()
    )

    mgr.add_prediction(prediction)

    return prediction
