import torch

from lib.dataloader.affinity_balancer import balanced_drug_targets
from lib.dataloader.raw_dataloaders import combined_predictor_dataloader
from lib.util.manager_util import ExperimentManager, PersistenceWrapper, NO_KEY
from lib.util.learner_util import EarlyStoppingLearner, Learner, simple_train, simple_score
from lib.util.random_util import aggregate, generator


def scheme_simple_parallel(mgr: ExperimentManager):
    mgr = mgr.choose(
        epoch=(NO_KEY, PersistenceWrapper(0)),
        is_minimum=(NO_KEY, PersistenceWrapper(False)),
        is_stopped=(NO_KEY, PersistenceWrapper(False)),
    )

    def on_train(score):
        mgr.log_history(['Aggregated', 'Train'], mgr.epoch.value, score)
        mgr.epoch.value += 1
        mgr.network.update_parameters(mgr.single_network)
        mgr.scheduler.step()

    def on_validate(score):
        mgr.log_history(['Only', 'Validate'], mgr.epoch.value, score)

    def on_minimum():
        mgr.log_history(['Only', 'Validate'], mgr.epoch.value, 'New minimum')
        mgr.save_state()

    mgr.initialize()

    mgr.load_state()

    # mgr.is_stopped.value = True  # for manual stopping
    # mgr.save_state()

    if not mgr.is_minimum.value and not mgr.is_stopped.value:
        mgr.open_writer()
        dir_path = mgr.dir
        dir_path.mkdir(parents=True, exist_ok=True)
        full_path = dir_path / 'watch.sh'
        if not full_path.exists():
            with open(full_path, 'wt') as f:
                f.write('#! /bin/bash\n')
                f.write('source /home/$USER/miniconda3/etc/profile.d/conda.sh\n')
                f.write('conda activate thesis-core\n')
                f.write(f'tensorboard --logdir={dir_path}\n')
        EarlyStoppingLearner(
            learner=Learner(
                train=lambda: simple_train(
                    model=mgr.single_network.combined_predictor,
                    dataloader=combined_predictor_dataloader(
                        drug_map=mgr.dta.fused_balanced_drug_cacher,
                        target_map=mgr.dta.fused_balanced_target_cacher,
                        affinities=mgr.minimal_dta.affinities,
                        drug_target_list=balanced_drug_targets(
                            possible_drugs=mgr.train_drugs,
                            possible_targets=mgr.train_targets,
                            positive_drug_targets=mgr.positive_train_drug_targets,
                            generator=generator(aggregate(mgr.epoch.value, mgr.BALANCE_SEED)),
                            ratio=1.0 / mgr.pos_weight
                        ),
                        batch_size=mgr.train_batch_size,
                        generator=generator(aggregate(mgr.epoch.value, mgr.TRAIN_SEED)),
                        device=mgr.device,
                        dtype=mgr.dtype
                    ),
                    loss_fn=mgr.train_loss_fn,
                    optimizer=mgr.optimizer,
                    batch_size=mgr.train_batch_size,
                    device=mgr.device,
                    dtype=mgr.dtype
                ),
                on_train=on_train,
                validate=lambda: simple_score(
                    model=mgr.network.module.combined_predictor,
                    dataloader=combined_predictor_dataloader(
                        drug_map=mgr.dta.fused_balanced_drug_cacher,
                        target_map=mgr.dta.fused_balanced_target_cacher,
                        affinities=mgr.minimal_dta.affinities,
                        drug_target_list=mgr.validate_drug_targets,
                        batch_size=mgr.score_batch_size,
                        generator=generator(aggregate(mgr.epoch.value, mgr.VALIDATE_SEED)),
                        device=mgr.device,
                        dtype=mgr.dtype
                    ),
                    loss_fn=mgr.validate_loss_fn,
                    batch_size=mgr.score_batch_size,
                    device=mgr.device,
                    dtype=mgr.dtype
                ),
                on_validate=on_validate
            ),
            window=mgr.window,
            on_minimum=on_minimum
        ).train()
        mgr.is_minimum.value = True
        mgr.save_state()

    return mgr
