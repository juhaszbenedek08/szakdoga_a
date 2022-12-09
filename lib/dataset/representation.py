from typing import Iterable, Protocol

import numpy as np
import torch
from numpy import ndarray

from lib.util.random_util import generator


class RepresentationData(Protocol):
    name: str
    width: int
    keys: Iterable[int]

    def __getitem__(self, item: int) -> torch.Tensor:
        ...

    def __repr__(self):
        ...


class ClassicRepresentationData:
    def __init__(self, name: str, items: dict[int, ndarray]):
        self.name = name
        self._mapping = {key: i for i, key in enumerate(items.keys())}
        self._internal = torch.tensor(np.stack(tuple(items.values())))

        size = self._internal.size()
        if len(size) == 1:
            self.width = 1
        else:
            self.width = size[1]

    @property
    def keys(self) -> Iterable[int]:
        return self._mapping.keys()

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._internal[self._mapping[item]]

    def __repr__(self):
        return self.name


class ConstantRepresentationData:
    def __init__(
            self,
            name: str,
            width: int,
            keys: set[int]
    ):
        self.name = name
        self.width = width
        self._internal = torch.zeros(width)
        self.keys = keys

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._internal

    def __repr__(self):
        return self.name


class RandomRepresentationData:

    def __init__(
            self,
            name: str,
            width: int,
            keys: set[int],
            seed: int
    ):
        self.name = name
        self.width = width
        self.keys = keys
        self._mapping = {key: i for i, key in enumerate(keys)}
        self._internal = torch.bernoulli(
            torch.full(
                size=(len(keys), width),
                fill_value=0.5
            ),
            generator=generator(seed)
        )

    def __getitem__(self, item: int) -> torch.Tensor:
        return self._internal[self._mapping[item]]

    def __repr__(self):
        return self.name
