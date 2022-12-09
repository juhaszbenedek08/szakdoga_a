from functools import wraps
from pathlib import Path
from typing import Any, Callable

from more_itertools import chunked
from numpy import ndarray

from lib.dataset.representation import ClassicRepresentationData
from lib.util.load_util import load, load_pickle, save_pickle
from lib.util.log_util import log_func, logger, pretty_tqdm


def batch_generated(
        load_ids: Callable[[], dict[int, Any]],
        main_path: Path,
        temp_path: Path,
        name: str,
        batch_size: int = 100
):
    def outer(func: Callable[[dict[int, Any]], dict[int, Any]]):
        @log_func
        @load(main_path, name + '.pickle')
        @wraps(func)
        def inner() -> dict[int, Any]:
            ids = load_ids()

            full_path = temp_path / (name + '.partial')

            if full_path.exists():
                loaded_repr = load_pickle(full_path)
            else:
                temp_path.mkdir(parents=True, exist_ok=True)
                loaded_repr = {}
                save_pickle(full_path, loaded_repr)

            remaining_dict = {
                index: id_
                for index, id_
                in ids.items()
                if index not in loaded_repr
            }
            with pretty_tqdm(total=len(ids)) as pbar:
                pbar.update(len(ids) - len(remaining_dict))
                for batch in chunked(remaining_dict.items(), n=batch_size):
                    batch_ids = dict(batch)
                    batch_repr = func(batch_ids)
                    loaded_repr |= batch_repr
                    save_pickle(full_path, loaded_repr)
                    pbar.update(len(batch))

            loaded_repr = {key: value for key, value in loaded_repr.items() if value is not None}
            return loaded_repr

        return inner

    return outer


#######################################

def safe_generate(
        name: str,
        items: dict[int, Any],
        generator: Callable[[Any], ndarray],
        name_map: dict[[Any], str]
):
    result = {}
    with pretty_tqdm(total=len(items)) as pbar:
        for key, item in items.items():
            try:
                repr_ = generator(item)
                if repr_ is None:
                    raise Exception()
                result[key] = repr_
            except:
                logger.warn(f'No available {name} of {name_map[key] : >20}')
            pbar.update()
    return result


def romol_representation(
        path: Path,
        name: str,
        load_ids: Callable[[], dict[int, Any]],
        load_romol: Callable[[], dict[int, Any]],
):
    def outer(generator: Callable[[Any], ndarray]):
        @log_func
        @load(path, name + '.pickle')
        @wraps(generator)
        def inner() -> ClassicRepresentationData:
            mapping = safe_generate(
                name=name,
                items=load_romol(),
                generator=generator,
                name_map=load_ids()
            )
            return ClassicRepresentationData(name=name, items=mapping)

        return inner

    return outer
