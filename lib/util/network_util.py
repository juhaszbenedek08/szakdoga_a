from more_itertools import pairwise
from typing import Callable, Any

import torch
from torch.nn import Module, Sequential, Linear, ModuleList


class XorNoise(Module):
    def __init__(self, width: int, p: float):
        super().__init__()
        self.p = p

        self.ones = torch.ones((width,))

    def forward(self, x):
        return torch.logical_xor(x, torch.bernoulli(self.ones, p=self.p)).to(dtype=x.dtype)


class ParallelConnector(Module):
    def __init__(self, *modules: Module):
        super().__init__()

        self.module_list = ModuleList(modules)

    def forward(self, xs):
        return tuple(module(x) for module, x in zip(self.module_list, xs))


class FullConnector(Module):
    def __init__(self, *modules: Module):
        super().__init__()

        self.module_list = ModuleList(modules)

    def forward(self, xs):
        return tuple(module(xs) for module in self.module_list)


class Rearranger(Module):
    def __init__(self, *mapping: list[int]):
        super().__init__()

        self.mapping = mapping

    def forward(self, xs):
        return tuple(tuple(xs[key] for key in keys) for keys in self.mapping)


class WeightedSum(Module):
    def __init__(self, *weights: int):
        super().__init__()

        self.weights = weights

    def forward(self, xs):
        return sum(x * weight for x, weight in zip(xs, self.weights))


class CustomModule(Module):
    def __init__(self, mapping: Callable[[...], Any]):
        super().__init__()

        self.mapping = mapping

    def forward(self, x):
        return self.mapping(x)


class Printer(Module):
    @staticmethod
    def forward(x):
        print(x)
        return x


class Concater(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(xs):
        return torch.concatenate(torch.atleast_2d(xs), dim=1)


class Offline(Module):

    def __init__(self, module: Module):
        super().__init__()
        self.module = module

    def forward(self, x):
        with torch.no_grad():
            return self.module(x)

    def train(self, mode=True):
        self.module.eval()

    # NOTE parameters() are not overridden


class Slicer(Module):
    def __init__(self, slice_: slice):
        super().__init__()

        self.slice_ = slice_

    def forward(self, x):
        return x[:, self.slice_]


class Single(Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        x, = x
        return x


class MLP(Module):
    def __init__(
            self,
            *layer_widths: int,
            in_between: Callable[[int], Module],
    ):
        super().__init__()

        def helper():
            iterable = pairwise(layer_widths)

            layer_1, layer_2 = next(iterable)
            yield Linear(layer_1, layer_2)

            for i, (layer_1, layer_2) in enumerate(iterable):
                yield in_between(i, layer_1)
                yield Linear(layer_1, layer_2)

        self.layers_ = Sequential(
            *helper()
        )

    def forward(self, x):
        return self.layers_(x)
