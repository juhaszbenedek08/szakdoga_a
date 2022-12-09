import time
from math import log, ceil
from typing import Any, Callable

import numpy as np
import requests

from lib.dataset.representation import ClassicRepresentationData, RepresentationData
from lib.load.embedder_list import prottrans, esm, seqvec
from lib.load.minimal_dta import load_kiba
from lib.util.anchor_util import TARGETS, TEMP
from lib.util.load_util import load, load_pickle
from lib.util.log_util import log_func, logger
from lib.util.representation_util import batch_generated


def load_target_ids() -> dict[int, Any]:
    return load_kiba().target_ids


@batch_generated(load_target_ids, TARGETS, TEMP, 'target_sequence', batch_size=25)
def load_target_sequence(batch_dict):
    url = 'https://rest.uniprot.org/idmapping/run/'
    params = {'ids': ','.join(batch_dict.values()),
              'from': 'UniProtKB_AC-ID',
              'to': 'UniProtKB'
              }
    reply = requests.post(url, params)
    job_id = reply.json()['jobId']

    url = 'https://rest.uniprot.org/idmapping/status/' + job_id
    while True:
        reply = requests.get(url)
        if 'results' in reply.json():
            break
        else:
            time.sleep(5)

    url = 'https://rest.uniprot.org/idmapping/uniprotkb/results/' + job_id
    params = {'format': 'fasta'}

    sequence_dict = {}
    reply = requests.get(url, params)
    for item in reply.text.split('>')[1:]:
        tokens = item.split('|')
        uniprot_id = tokens[1]
        sequence = ''.join(tokens[2].split('\n')[1:])
        sequence_dict[uniprot_id] = sequence
    result = {}
    for index, target_id in batch_dict.items():
        try:
            result[index] = sequence_dict[target_id]
        except KeyError:
            logger.warning(f'No available target_sequence.pickle of {target_id : >20}')

    return result


#
# @log_func
# def ensure_fasta():
#     if FASTA_PATH.exists():
#         return
#     target_id_repr = load_target_ids()
#     target_sequence = load_target_sequence()
#
#     with open(FASTA_PATH, 'at') as f:
#         for index, target_id in target_id_repr.items():
#             sequence = target_sequence[index]
#             f.write(f'>{target_id}\n')
#             f.write(f'{sequence}\n')


###################################################

target_repr_loaders = []


def _target_repr_loader(func: Callable[[], RepresentationData]):
    target_repr_loaders.append(func)
    return func


# @_target_repr_loader
@log_func
@load(TARGETS, 'target_sequence_numeric.pickle')
def load_target_sequence_numeric():
    target_sequence = load_target_sequence()

    max_length = max(len(it) for it in target_sequence.values())

    letters = [
        'A', 'R', 'N', 'D', 'B',
        'C', 'E', 'Q', 'Z', 'G',
        'H', 'I', 'L', 'K', 'M',
        'F', 'P', 'S', 'T', 'W',
        'Y', 'V'
    ]

    min_bits = ceil(log(len(letters), 2))

    mapping = {}
    for i, value in enumerate(letters):
        list_form = [int(c) for c in format(i, 'b')]
        while len(list_form) < min_bits:
            list_form.insert(0, 0)
        mapping[value] = np.array(list_form)

    result = {}
    for index, sequence in target_sequence.items():
        arr = np.zeros(shape=max_length * min_bits, dtype=np.int8)
        for i, c in enumerate(sequence):
            arr[i:i + min_bits] = mapping[c]
        result[index] = arr

    return ClassicRepresentationData('target_sequence_numeric', result)


@_target_repr_loader
@log_func
@load(esm.repr_path, f'{esm.repr_name}.pickle')
def load_target_esm():
    return ClassicRepresentationData(
        esm.repr_name,
        {
            key: value[np.mgrid[0:100:10, 0:100:10]].flatten()
            for key, value
            in load_pickle(esm.raw_path / f'{esm.repr_name}.partial').items()
            if value is not None
        }
    )


@_target_repr_loader
@log_func
@load(seqvec.repr_path, f'{seqvec.repr_name}.pickle')
def load_target_seqvec():
    return ClassicRepresentationData(
        seqvec.repr_name,
        {
            key: value[0][np.mgrid[0:100:10, 0:100:10]].flatten()
            for key, value
            in load_pickle(seqvec.raw_path / f'{seqvec.repr_name}.partial').items()
            if value is not None
        }
    )


@_target_repr_loader
@log_func
@load(prottrans.repr_path, f'{prottrans.repr_name}.pickle')
def load_target_prottrans():
    return ClassicRepresentationData(
        prottrans.repr_name,
        {
            key: value[np.mgrid[0:100:10, 0:100:10]].flatten()
            for key, value
            in load_pickle(prottrans.raw_path / f'{prottrans.repr_name}.partial').items()
            if value is not None
        }
    )
