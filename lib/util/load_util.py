import functools
import gc
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import pandas as pd
import torch

_loaders = {}
_savers = {}


def _loader(name: str):
    def inner(func: Callable[[Path], Any]):
        _loaders[name] = func
        return func

    return inner


def _saver(name: str):
    def inner(func: Callable[[Path], Any]):
        _savers[name] = func
        return func

    return inner


##############################

@_loader('pickle')
def load_pickle(full_path: Path):
    with full_path.open('rb') as f:
        return pickle.load(f)


@_saver('pickle')
def save_pickle(full_path: Path, content: Any):
    with full_path.open('wb') as f:
        pickle.dump(content, f)


##############################

@_loader('json')
def load_json(full_path: Path):
    with full_path.open('r') as f:
        return json.load(f)


@_saver('json')
def save_json(full_path: Path, content: Any):
    with full_path.open('w') as f:
        json.dump(content, f, indent=4)


##############################

@_loader('pandas')
def load_pandas(full_path: Path):
    return pd.read_pickle(full_path)


@_saver('pandas')
def save_pandas(full_path: Path, content: Any):
    content.to_pickle(full_path)


###############################

@_loader('pt')
def load_torch(full_path: Path):
    with full_path.open('rb') as f:
        return torch.load(f)


@_saver('pt')
def save_torch(full_path: Path, content: Any):
    with full_path.open('wb') as f:
        torch.save(content, f)


##############################################################

@dataclass
class _Wrapped:
    value: Any


def load(path: Path, name: str):
    def outer(generate: Callable[[], Any]):
        @functools.wraps(generate)
        def inner():
            full_path = path / name
            extension = full_path.suffix[1:]

            if full_path.exists():
                result = _loaders[extension](full_path)
            else:
                path.mkdir(parents=True, exist_ok=True)
                result = generate()
                _savers[extension](full_path, result)
            return result

        return inner

    return outer


def free():
    gc.collect()
    torch.cuda.empty_cache()
