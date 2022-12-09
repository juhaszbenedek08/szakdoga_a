from functools import cache, reduce
from typing import Iterable

import torch

from lib.dataset.minimal_dta import MinimalDTA
from lib.dataset.representation import RepresentationData
from lib.dataset.triset import Triset
from lib.util.dataset_util import Balancer


class DTA:

    def __init__(
            self,
            name: str,
            minimal_dta: MinimalDTA,
            drug_reprs: list[RepresentationData],
            target_reprs: list[RepresentationData]
    ):
        self.name = name
        self.minimal_dta = minimal_dta
        self.drug_reprs = drug_reprs
        self.target_reprs = target_reprs

        def helper(
                ids: Iterable[int],
                reprs: list[RepresentationData]
        ):
            return set(
                reduce(
                    lambda a, b: a.intersection(b),
                    map(
                        lambda it: set(it.keys),
                        reprs
                    ),
                    set(ids)
                )
            )

        self.drugs = helper(self.minimal_dta.drug_ids.keys(), drug_reprs)
        self.targets = helper(self.minimal_dta.target_ids.keys(), target_reprs)

        self.affinities = set(
            (drug, target)
            for drug, target
            in self.minimal_dta.affinities.keys
            if drug in self.drugs and target in self.targets
        )

        def get_fused_drug(item):
            return torch.concat(tuple(torch.atleast_1d(repr_[item]) for repr_ in self.drug_reprs))

        def get_fused_target(item):
            return torch.concat(tuple(torch.atleast_1d(repr_[item]) for repr_ in self.target_reprs))

        self.drug_width = sum(repr_.width for repr_ in self.drug_reprs)
        self.target_width = sum(repr_.width for repr_ in self.target_reprs)

        self.fused_drug_balancer = Balancer(get_fused_drug, self.drugs)
        self.fused_target_balancer = Balancer(get_fused_target, self.targets)

        self.fused_balanced_drug_cacher = cache(self.fused_drug_balancer)
        self.fused_balanced_target_cacher = cache(self.fused_target_balancer)

    @staticmethod
    def from_minimal_dta(
            minimal_dta: MinimalDTA,
            drug_reprs: list[RepresentationData],
            target_reprs: list[RepresentationData]
    ):
        return DTA(
            name=minimal_dta.name + '_fused',
            minimal_dta=minimal_dta,
            drug_reprs=drug_reprs,
            target_reprs=target_reprs
        )

    @property
    def triset(self):
        return Triset(
            self.drugs,
            self.targets,
            self.affinities
        )
