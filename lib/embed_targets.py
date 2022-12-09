import pickle

import torch

from lib.load.embedder_list import embedders
from lib.util.anchor_util import KIBA, TARGETS
from lib.util.log_util import logger, pretty_tqdm

import bio_embeddings.embed as embed


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(path, content):
    with open(path, 'wb') as f:
        pickle.dump(content, f)


if __name__ == '__main__':
    KIBA_PATH = KIBA / 'kiba.pickle'

    names = load_pickle(KIBA_PATH).target_ids

    SEQ_PATH = TARGETS / 'target_sequence.pickle'
    ids = load_pickle(SEQ_PATH)

    for embedder in embedders:
        full_path = embedder.raw_path / f'{embedder.repr_name}.partial'

        if full_path.exists():
            loaded_repr = load_pickle(full_path)
        else:
            embedder.raw_path.mkdir(parents=True, exist_ok=True)
            loaded_repr = {}
            save_pickle(full_path, loaded_repr)

        remaining_dict = {
            index: id_
            for index, id_
            in ids.items()
            if index not in loaded_repr or loaded_repr[index] is None
        }

        model = getattr(embed, embedder.embedder_name)(device=torch.device('cpu'), half_model=True)
        # NOTE use cuda if possible, otherwise rewrite

        logger.info(f'Starting {embedder.repr_name}')

        with pretty_tqdm(total=len(ids)) as pbar:
            pbar.update(len(ids) - len(remaining_dict))
            for id_, sequence in remaining_dict.items():
                try:
                    repr_ = model.embed(sequence)
                    if repr_ is None:
                        raise Exception()
                    loaded_repr[id_] = repr_

                    save_pickle(full_path, loaded_repr)
                except:
                    logger.warn(f'No available {embedder.repr_name} of {names[id_] : >20}')
                pbar.update()
        logger.info(f'{embedder.repr_name} finished')
