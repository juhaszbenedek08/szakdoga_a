import hashlib
import json
from datetime import datetime
from typing import Any, Protocol, Union

from torch.utils.tensorboard import SummaryWriter

from lib.analyze.prediction import Prediction
from lib.util.anchor_util import MODELS
from lib.util.load_util import load_torch, save_torch, save_json, load_json, load_pickle, save_pickle
from lib.util.log_util import logger, pretty_format


class Persistent(Protocol):
    def state_dict(self) -> Any:
        ...

    def load_state_dict(self, state_dict: Any) -> Any:
        ...


class PersistenceWrapper:
    def __init__(self, value: Any):
        self.value = value

    def state_dict(self):
        return self.value

    def load_state_dict(self, state_dict: Any):
        self.value = state_dict

    def __repr__(self):
        return repr(self.value)


NO_KEY = None


class ExperimentManager:

    def __init__(self):
        self._keys = {}
        self._values = {}
        self.hash = None
        self.dir = None
        self.writer = None
        self.history_path = None

    def choose(self, **options: Any):
        result = ExperimentManager()
        result._keys = self._keys.copy()
        result._values = self._values.copy()

        for name, item in options.items():
            if isinstance(item, tuple):
                key, value = item
            else:
                key, value = repr(item), item

            if key is not NO_KEY:
                result._keys[name] = key
            result._values[name] = value

        return result

    def __getattr__(self, item):
        return self._values[item]

    def initialize(self):
        self.hash = hashlib.md5(
            json.dumps(self._keys, sort_keys=True)
            .encode(encoding='UTF-8', errors='strict')
        ).hexdigest()

        self.dir = MODELS / f'model_{self.hash}'
        self.dir.mkdir(parents=True, exist_ok=True)

        info_path = self.dir / 'info.json'
        if not info_path.exists():
            info = self._keys.copy()
            info['hash'] = hash(self)
            info['creation datetime'] = datetime.now().isoformat()
            save_json(info_path, info)

        self.history_path = self.dir / 'history.json'
        if not self.history_path.exists():
            save_json(self.history_path, [])

    def save_state(self):
        values = {}
        for name, value in self._values.items():
            if hasattr(value, 'state_dict') and hasattr(value, 'load_state_dict'):
                values[name] = value.state_dict()
        save_torch(
            self.dir / 'state.pt',
            values
        )

    def open_writer(self):
        self.writer = SummaryWriter(self.dir)

    def load_state(self):
        path = self.dir / 'state.pt'
        if path.exists():
            loaded = load_torch(path)
            for name, value in loaded.items():
                assert name in self._values
                self._values[name].load_state_dict(value)

    def log_history(
            self,
            tags: list[str],
            epoch: int,
            value: Union[str, float, int]
    ):
        if not isinstance(value, str):
            content = load_json(self.history_path)
            content.append({'tags': tags, 'epoch': epoch, 'value': value})
            save_json(self.history_path, content)

            logger.info(f'{f"[{epoch}]" : <5} {pretty_format(tags) : <30} {value: <.4}')

            self.writer.add_scalar('/'.join(tags), value, epoch)

        else:
            logger.info(f'{f"[{epoch}]" : <5} {pretty_format(tags) : <30} {value}')

    # def search_history(
    #         self,
    #         tags: list[str],
    #         epoch: int):
    #     path = self.dir / 'history.json'
    #     content = load_json(path)
    #     if content is not None:
    #         for item in content:
    #             if item['tags'] == tags and item['epoch'] == epoch:
    #                 return item['value']
    #
    #     return None

    def add_prediction(self, prediction: Prediction):
        path = self.dir / 'predictions.pickle'
        if path.exists():
            predictions = load_pickle(path)
        else:
            predictions = []
        predictions.append(prediction)
        save_pickle(path, predictions)

    def get_prediction(
            self,
            drug_split_name: str,
            target_split_name: str,
            mode: str
    ):
        path = self.dir / 'predictions.pickle'
        if path.exists():
            predictions = load_pickle(path)
            for prediction in predictions:
                if prediction.drug_split_name == drug_split_name \
                        and prediction.target_split_name == target_split_name \
                        and prediction.mode == mode:
                    return prediction
        return None

    def __del__(self):
        if self.writer is not None:
            self.writer.close()
