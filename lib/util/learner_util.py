from dataclasses import dataclass, field
from typing import Callable, Protocol, Iterable, Any

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from lib.util.log_util import pretty_tqdm


class Optimizer(Protocol):
    def zero_grad(self) -> Any:
        ...

    def step(self) -> Any:
        ...

    def state_dict(self) -> Any:
        ...

    def load_state_dict(self, state_dict: Any) -> Any:
        ...


@dataclass
class Learner:
    train: Callable[[], float]
    validate: Callable[[], float]
    on_train: Callable[[float], Any]
    on_validate: Callable[[float], Any]


@dataclass
class EarlyStoppingLearner:
    learner: Learner
    window: int
    on_minimum: Callable[[], Any]

    def train(self):
        min_epoch = current_epoch = 0
        min_score = self.learner.validate()
        self.learner.on_validate(min_score)

        while current_epoch - min_epoch < self.window:
            train_score = self.learner.train()
            self.learner.on_train(train_score)
            validate_score = self.learner.validate()
            self.learner.on_validate(validate_score)
            if validate_score < min_score:
                min_epoch = current_epoch
                min_score = validate_score
                self.on_minimum()
            current_epoch += 1


# @dataclass
# class CircularLearner(Learner):
#     learners: list[Learner]
#     epochs: list[int]
#     aggregate_fn: Callable[[list[float]], float]
#
#     train: Callable[[], float] = field(init=False)
#     validate: Callable[[], float] = field(init=False)
#     on_train: Callable[[float], Any]
#     on_validate: Callable[[float], Any]
#
#     def __post_init__(self):
#         self.train = self._train
#         self.validate = self._validate
#
#     def _train(self):
#         scores = []
#         for learner, epoch in zip(self.learners, self.epochs):
#             for _ in range(epoch):
#                 score = learner.train()
#                 learner.on_train(score)
#             scores.append(score)
#         return self.aggregate_fn(*scores)
#
#     def _validate(self):
#         scores = []
#         for learner, epoch in zip(self.learners, self.epochs):
#             score = learner.validate()
#             learner.on_validate(score)
#             scores.append(score)
#         return self.aggregate_fn(*scores)


def simple_train(
        model: Module,
        dataloader: DataLoader,
        loss_fn: Module,
        optimizer: Optimizer,
        batch_size: int,
        device,
        dtype
) -> float:
    acc_loss = torch.tensor(0.0, device=device, dtype=dtype)

    model.train()
    for x, y in pretty_tqdm(dataloader, desc='Train', unit_scale=batch_size):
        optimizer.zero_grad()

        y_p = model(x)
        loss = loss_fn(y_p, y)
        loss.backward()

        optimizer.step()

        acc_loss += float(loss)

    avg_loss = acc_loss / len(dataloader) / batch_size
    return float(avg_loss)


def simple_score(
        model: Module,
        dataloader: DataLoader,
        loss_fn: Module,
        batch_size: int,
        device,
        dtype
) -> float:
    acc_loss = torch.tensor(0.0, device=device, dtype=dtype)

    model.eval()
    with torch.no_grad():
        for x, y in pretty_tqdm(dataloader, desc='Score', unit_scale=batch_size):
            y_p = model(x)
            loss = loss_fn(y_p, y)
            acc_loss += float(loss)
    avg_loss = acc_loss / len(dataloader) / batch_size

    return float(avg_loss)


def predict(
        model: Module,
        dataloader: Iterable,
        batch_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.no_grad():
        y_ps = []
        ys = []
        for x, y in pretty_tqdm(dataloader, desc='Predict', unit_scale=batch_size):
            ys.append(y)
            y_ps.append(model(x))
        return torch.concatenate(y_ps, dim=0), torch.concatenate(ys, dim=0)
