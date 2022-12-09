from dataclasses import dataclass
from typing import Any

from lib.dataset.affinity import AffinityData
from lib.dataset.triset import Triset
from lib.util.log_util import log_func
from lib.util.random_util import shuffle_mapper, aggregate


@dataclass
class MinimalDTA:
    name: str
    drug_ids: dict[int, Any]
    target_ids: dict[int, Any]
    affinities: AffinityData

    @property
    def triset(self):
        return Triset(
            set(self.drug_ids.keys()),
            set(self.target_ids.keys()),
            set(self.affinities.keys)
        )

    @log_func
    def remapped(self, seed: int):
        drug_remapping = shuffle_mapper(list(self.drug_ids.keys()), aggregate(seed, 0))
        target_remapping = shuffle_mapper(list(self.target_ids.keys()), aggregate(seed, 1000))
        mapping = {
            (drug_remapping[drug], target_remapping[target]): affinity
            for (drug, target), affinity
            in self.affinities.items
        }

        return MinimalDTA(
            f'{self.name}_control_{seed}',
            self.drug_ids,
            self.target_ids,
            AffinityData(
                self.name + '_control_affinities',
                mapping
            )
        )

    def __repr__(self):
        return self.name
