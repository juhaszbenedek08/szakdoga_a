import torch
from torch.linalg import vecdot
from torch.nn import Sequential, Dropout, ReLU, ConstantPad1d, Linear
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits
from torch.nn.utils import prune
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, SWALR

from lib.model.dta_network import DTANetwork
from lib.model.dta_scheme import scheme_simple_parallel
from lib.util.manager_util import ExperimentManager, NO_KEY
from lib.util.network_util import Slicer


def add_common_model(mgr: ExperimentManager):
    drug_hidden_size = 100
    target_hidden_size = 100

    mgr = mgr.choose(
        drug_hidden_size=drug_hidden_size,
        target_hidden_size=target_hidden_size,
        single_network=(
            'common_model',
            DTANetwork(
                drug_encoder=Sequential(
                    ConstantPad1d((0, mgr.total_drug_width - mgr.dta.drug_width), 0),
                    Dropout(),
                    Linear(mgr.total_drug_width, drug_hidden_size),
                    ReLU(),
                ),
                drug_decoder=Sequential(
                    Linear(drug_hidden_size, mgr.total_drug_width),
                    Slicer(slice(mgr.dta.drug_width)),
                ),
                target_encoder=Sequential(
                    ConstantPad1d((0, mgr.total_target_width - mgr.dta.target_width), 0),
                    Dropout(),
                    Linear(mgr.total_target_width, target_hidden_size),
                    ReLU(),
                ),
                target_decoder=Sequential(
                    Dropout(),
                    Linear(target_hidden_size, mgr.total_target_width),
                    Slicer(slice(mgr.dta.target_width)),
                ),
                predictor=Linear(drug_hidden_size + target_hidden_size, 1),
            ).to(mgr.device, mgr.dtype)),
        pruning=0.0
    )

    for module in mgr.single_network.modules():
        if isinstance(module, Linear):
            prune.random_unstructured(module, 'weight', mgr.pruning)
            prune.remove(module, 'weight')

    return mgr


def custom_loss(weights, mgr):
    d_w, t_w, a_w, r_w = tuple(
        torch.tensor(w, device=mgr.device, dtype=mgr.dtype)
        for w in weights
    )
    r_w /= mgr.drug_hidden_size + mgr.target_hidden_size
    pos_weight = torch.tensor(mgr.pos_weight, device=mgr.device, dtype=mgr.dtype)

    def helper(y_p, y):
        y_p_d, y_p_t, y_p_a, y_p_r = y_p
        y_d, y_t, y_a = y

        return d_w * mse_loss(y_p_d, y_d) \
            + t_w * mse_loss(y_p_t, y_t) \
            + a_w * binary_cross_entropy_with_logits(y_p_a, y_a, pos_weight=pos_weight) \
            + r_w * vecdot(y_p_r, y_p_r).mean()

    return helper


def add_common_scheme(mgr: ExperimentManager):
    mgr = mgr.choose(
        train_drugs=(NO_KEY, [*mgr.folding.drug_with_repr_affinity, *mgr.folding.drug_with_repr]),
        train_targets=(NO_KEY, [*mgr.folding.target_with_repr_affinity, *mgr.folding.target_with_repr]),
        positive_train_drug_targets=(
            NO_KEY,
            {item for item in mgr.fold.train if item in mgr.minimal_dta.affinities.true}
        ),
        validate_drug_targets=(NO_KEY, mgr.fold.validate),
        window=100,
        optimizer=('AdamW', AdamW(mgr.single_network.combined_predictor.parameters())),
        pos_weight=10.0,
        weights=(NO_KEY, (0.1, 0.1, 1.0, 0.1)),
        train_batch_size=2 ** 5,
        score_batch_size=2 ** 5,
        average_ratio=0.5,
        network=('averaged 50/50 cuda', AveragedModel(
            mgr.single_network,
            device=mgr.device,
            avg_fn=lambda averaged_model_parameter, model_parameter, _:
            0.5 * averaged_model_parameter + 0.5 * model_parameter
        ))
    )
    mgr = mgr.choose(
        train_loss_fn=(f'MSE, MSE, BCEWIthLogit, L2, {mgr.weights}', custom_loss(mgr.weights, mgr)),
        validate_loss_fn=(NO_KEY, custom_loss((0.0, 0.0, mgr.weights[2], 0.0), mgr)),
        scheduler=(f'SWALR 0.001', SWALR(mgr.optimizer, swa_lr=0.001, anneal_epochs=3, anneal_strategy="linear"))
    )
    return mgr.choose(
        scheme=(NO_KEY, lambda: scheme_simple_parallel(mgr))
    )
