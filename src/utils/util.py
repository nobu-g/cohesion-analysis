import json
from pathlib import Path
from collections import OrderedDict
from collections.abc import Callable

import torch
from kyoto_reader import BasePhrase


def read_json(fname):
    if isinstance(fname, str):
        fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, ensure_ascii=False, indent=2, sort_keys=False)


class OrderedDefaultDict(OrderedDict):
    # Source: https://stackoverflow.com/questions/6190331/how-to-implement-an-ordered-default-dict
    def __init__(self, default_factory=None, *a, **kw):
        if (default_factory is not None and
           not isinstance(default_factory, Callable)):
            raise TypeError('first argument must be callable or None')
        OrderedDict.__init__(self, *a, **kw)
        self.default_factory = default_factory

    def __getitem__(self, key):
        try:
            return OrderedDict.__getitem__(self, key)
        except KeyError:
            return self.__missing__(key)

    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = value = self.default_factory()
        return value

    def __reduce__(self):
        if self.default_factory is None:
            args = tuple()
        else:
            args = self.default_factory,
        return type(self), args, None, None, self.items()

    def copy(self):
        return self.__copy__()

    def __copy__(self):
        return type(self)(self.default_factory, self)

    def __deepcopy__(self, memo):
        import copy
        return type(self)(self.default_factory, copy.deepcopy(self.items()))

    def __repr__(self):
        return f'OrderedDefaultDict({self.default_factory}, {OrderedDict.__repr__(self)})'


def prepare_device(n_gpu_use: int, logger):
    """
    setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        logger.warning("Warning: There\'s no GPU available on this machine,"
                       "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.warning("Warning: The number of GPU\'s configured to use is {}, but only {} are available "
                       "on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def is_pas_target(bp: BasePhrase, verbal: bool, nominal: bool) -> bool:
    if verbal and '用言' in bp.tag.features:
        return True
    if nominal and '非用言格解析' in bp.tag.features:
        return True
    return False


def is_bridging_target(bp: BasePhrase) -> bool:
    return '体言' in bp.tag.features and '非用言格解析' not in bp.tag.features


def is_coreference_target(bp: BasePhrase) -> bool:
    return '体言' in bp.tag.features
