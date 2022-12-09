from functools import reduce
from itertools import product
from operator import mul

import torch

from lib.util.log_util import logger, log_func
from lib.util.random_util import generate_picks


@log_func
def balanced_drug_targets(
        possible_drugs: list[int],
        possible_targets: list[int],
        positive_drug_targets: set[tuple[int, int]],
        generator: torch.Generator,
        max_tries: int = 100,
        ratio: float = 0.1
) -> list[tuple[int, int]]:
    result = positive_drug_targets.copy()
    drug_picks = generate_picks(possible_drugs, generator)
    target_picks = generate_picks(possible_targets, generator)

    goal = round(len(positive_drug_targets) / ratio)

    for i in range(goal):
        for _ in range(max_tries):
            item = next(drug_picks), next(target_picks)
            if item not in result:
                result.add(item)
                break
        else:
            logger.warn(f'Could not balance a difference of {goal}, only {i}')
            break
    return list(result)


def except_drug_targets(
        possible_drugs,
        possible_targets,
        except_drug_target_list
):
    except_set = set(except_drug_target_list)
    return [item
            for item
            in product(possible_drugs, possible_targets)
            if item not in except_set]


class Cartesian:
    def __init__(self, *lists: list):
        self.lists = lists
        self.total_length = reduce(mul, (len(list_) for list_ in lists))

    def __getitem__(self, item):
        result = []
        for list_ in reversed(self.lists):
            item, rem = divmod(item, len(list_))
            result.append(rem)
        return reversed(result)

    def __len__(self):
        return self.total_length


def all_possible_drug_targets(
        possible_drugs: list[int],
        possible_targets: list[int]
):
    return Cartesian(possible_drugs, possible_targets)
