from typing import Callable

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from lib.dataset.affinity import AffinityData
from lib.util.dataset_util import MappingDataset


# def drug_autoencoder_dataloader(
#         drug_map: Callable[[int], torch.Tensor],
#         drug_list: list[int],
#         batch_size: int,
#         generator: torch.Generator
# ):
#     return DataLoader(
#         dataset=MappingDataset(lambda item: duplicate(drug_map(item))),
#         sampler=SubsetRandomSampler(drug_list, generator),
#         batch_size=batch_size,
#         drop_last=False
#     )
#
#
# def target_autoencoder_dataloader(
#         target_map: Callable[[int], torch.Tensor],
#         target_list: list[int],
#         batch_size: int,
#         generator: torch.Generator
# ):
#     return DataLoader(
#         dataset=MappingDataset(lambda item: duplicate(target_map(item))),
#         sampler=SubsetRandomSampler(target_list, generator),
#         batch_size=batch_size,
#         drop_last=False
#     )

def simple_predictor_dataloader(
        drug_map: Callable[[int], torch.Tensor],
        target_map: Callable[[int], torch.Tensor],
        affinities: AffinityData,
        drug_target_list: list[tuple[int, int]],
        batch_size: int,
        generator: torch.Generator,
        device,
        dtype
):
    def helper(item):
        drug, target = item
        return (
            (
                drug_map(drug).to(device, dtype=dtype),
                target_map(target).to(device, dtype=dtype)
            ),
            affinities[drug, target].to(device, dtype=dtype)
        )

    return DataLoader(
        dataset=MappingDataset(helper),
        sampler=SubsetRandomSampler(drug_target_list, generator),
        batch_size=batch_size,
        drop_last=False
    )


def combined_predictor_dataloader(
        drug_map: Callable[[int], torch.Tensor],
        target_map: Callable[[int], torch.Tensor],
        affinities: AffinityData,
        drug_target_list: list[tuple[int, int]],
        batch_size: int,
        generator: torch.Generator,
        device,
        dtype
):
    def helper(item):
        drug, target = item
        drug_repr = drug_map(drug).to(device, dtype=dtype)
        target_repr = target_map(target).to(device, dtype=dtype)

        return (
            (drug_repr, target_repr),
            (drug_repr, target_repr, affinities[drug, target].to(device, dtype=dtype))
        )

    return DataLoader(
        dataset=MappingDataset(helper),
        sampler=SubsetRandomSampler(drug_target_list, generator),
        batch_size=batch_size,
        drop_last=False
    )
