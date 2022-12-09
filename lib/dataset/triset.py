from typing import Any

from lib.dataset.affinity import AffinityData


class Triset:
    def __init__(
            self,
            drugs: set[int],
            targets: set[int],
            affinities: set[tuple[int, int]]
    ):
        self.drugs = drugs
        self.targets = targets
        self.affinities = affinities

        self.drug_num = len(self.drugs)
        self.target_num = len(self.targets)
        self.affinity_num = len(self.affinities)
        self.possible_affinity_num = self.target_num * self.drug_num

    def report(self, affinities: AffinityData) -> dict[str, Any]:
        true = 0
        false = 0
        for item in self.affinities:
            if item in affinities.true:
                true += 1
            else:
                false += 1
        unknown = self.possible_affinity_num - true - false

        filled_drugs = set()
        filled_targets = set()
        for drug, target in self.affinities:
            filled_drugs.add(drug)
            filled_targets.add(target)

        return {
            'Drug number': self.drug_num,
            'Target number': self.target_num,
            'Filled drug number': len(filled_drugs),
            'Filled target number': len(filled_targets),
            'Filled drug ratio': round(len(filled_drugs) / self.drug_num, 6),
            'Filled target ratio': round(len(filled_targets) / self.target_num, 6),
            'Affinity number': self.affinity_num,
            'Affinity grid size': self.possible_affinity_num,
            'Positive affinities': true,
            'Positive affinities ratio': round(true / self.possible_affinity_num, 6),
            'Negative affinities': false,
            'Negative affinities ratio': round(false / self.possible_affinity_num, 6),
            'Unknown affinities': unknown,
            'Unknown affinities ratio': round(unknown / self.possible_affinity_num, 6)
        }
