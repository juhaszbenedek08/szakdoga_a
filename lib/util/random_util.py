import torch


def generator(seed: int):
    result = torch.Generator()
    result.manual_seed(seed)
    return result


def aggregate(*seeds: int) -> int:
    return int(''.join(map(str, seeds)))


def shuffle_mapper(original: list[int], seed: int) -> dict[int, int]:
    return {
        key: original[index]
        for key, index
        in zip(original, torch.randperm(len(original), generator=generator(seed)))
    }


def generate_picks(items: list, generator: torch.Generator, batch_size: int = 10):
    while True:
        indices = torch.randint(0, len(items), (batch_size,), generator=generator)
        for index in indices:
            yield items[index]
