from dataclasses import dataclass

import torch


@dataclass
class DTAFolding:
    drug_with_repr_affinity: list[int]
    drug_with_repr: list[int]
    drug_with_nothing: list[int]
    target_with_repr_affinity: list[int]
    target_with_repr: list[int]
    target_with_nothing: list[int]


@dataclass
class Fold:
    train: list[tuple[int, int]]
    validate: list[tuple[int, int]]


def train_test_split(
        drugs: list[int],
        targets: list[int],
        generator: torch.Generator,
        with_affinity_ratio: float = 0.95,
        with_representation_ratio: float = 0.5
):
    def helper(list_):
        order = torch.randperm(len(list_), generator=generator)

        split_1 = round(len(list_) * with_affinity_ratio)
        rem_1 = len(list_) - split_1
        split_2 = split_1 + round(rem_1 * with_representation_ratio)

        train_validate = [list_[index] for index in order[:split_1]]
        no_affinity = [list_[index] for index in order[split_1:split_2]]
        no_repr = [list_[index] for index in order[split_2:]]

        return train_validate, no_affinity, no_repr

    return DTAFolding(*helper(drugs), *helper(targets))


def train_validate_splits(
        drug_target_list: list[tuple[int, int]],
        generator: torch.Generator,
        k: int = 5,
):
    order = torch.randperm(len(drug_target_list), generator=generator)
    chunks = torch.tensor_split(order, k)

    result = []
    for i in range(k):
        train = []
        validate = []
        for j, chunk in enumerate(chunks):
            current = (drug_target_list[int(index)] for index in chunk)
            if i == j:
                validate.extend(current)
            else:
                train.extend(current)
        result.append(Fold(train, validate))

    return result
