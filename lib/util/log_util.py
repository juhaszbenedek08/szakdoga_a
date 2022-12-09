import logging
import pprint
from functools import wraps, partial
from typing import Callable
import sys

import tqdm
from tqdm.contrib import DummyTqdmFile

orig_out_err = sys.stdout, sys.stderr
sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)

logger = logging.getLogger('thesis')
logger.setLevel(level=logging.INFO)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(
    logging.Formatter(
        fmt='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%H:%M:%S'
    )
)
logger.addHandler(_sh)


def log_func(func: Callable):
    @wraps(func)
    def inner(*args, **kwargs):
        logger.info('Entering: ' + inner.__qualname__ + '...')
        result = func(*args, **kwargs)
        logger.info('Exiting: ' + inner.__qualname__)
        return result

    return inner


_pp = pprint.PrettyPrinter(
    sort_dicts=False,
    width=120,
    indent=4,
    stream=sys.stdout
)


def pretty_format(arg):
    return _pp.pformat(arg)


pretty_tqdm = partial(tqdm.tqdm, file=orig_out_err[0], dynamic_ncols=True)
