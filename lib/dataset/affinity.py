import torch


class AffinityData:
    def __init__(
            self,
            name: str,
            items: dict[tuple[int, int]]
    ):
        self.name = name
        self.true = {key for key, value in items.items() if value}
        self.false = {key for key, value in items.items() if not value}
        self._no = torch.tensor([0.0])
        self._yes = torch.tensor([1.0])

    def __getitem__(self, item) -> torch.Tensor:
        if item in self.true:
            return self._yes
        else:
            return self._no

    @property
    def keys(self):
        yield from self.true
        yield from self.false

    @property
    def items(self):
        for key in self.true:
            yield key, True
        for key in self.false:
            yield key, False

    def __repr__(self):
        return self.name
