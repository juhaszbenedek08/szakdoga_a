from dataclasses import dataclass
from typing import Callable, Any, Iterable

import torch
from torch.utils.data import Dataset


def duplicate(item):
    return item, item


class Balancer:
    def __init__(
            self,
            mapping: Callable[[...], torch.Tensor],
            keys: Iterable,
            eps: float = 1e-6
    ):
        self.mapping = mapping

        tensor = torch.stack([mapping(key) for key in keys], dim=0).to(dtype=torch.float)

        std, mean = torch.std_mean(tensor, dim=0)

        self.mean = mean
        self.inv = 1 / torch.sqrt(std + eps)

    def __call__(self, item):
        return (self.mapping(item) - self.mean) * self.inv

    def to(self, *args, **kwargs):
        self.mean = self.mean.to(*args, **kwargs)
        self.inv = self.inv.to(*args, **kwargs)
        return self


@dataclass
class MappingDataset(Dataset):
    mapping: Callable[[...], Any]

    def __getitem__(self, item):
        return self.mapping(item)
