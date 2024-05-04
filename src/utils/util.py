from collections.abc import Collection
from datetime import datetime, timedelta, timezone
from typing import Union

import numpy as np
from dataclasses_json import DataClassJsonMixin, LetterCase, config
from omegaconf import Container, OmegaConf
from rhoknp import BasePhrase, Phrase

IGNORE_INDEX = -100


class CamelCaseDataClassJsonMixin(DataClassJsonMixin):
    dataclass_json_config = config(letter_case=LetterCase.CAMEL)["dataclasses_json"]  # type: ignore


def current_datetime_string(fmt: str) -> str:
    now = datetime.now(timezone(timedelta(hours=+9), name="JST"))
    return now.strftime(fmt)


def get_core_expression(unit: Union[Phrase, BasePhrase]) -> str:
    """A core expression without ancillary words."""
    morphemes = unit.morphemes
    sidx = 0
    for i, morpheme in enumerate(morphemes):
        if morpheme.pos not in ("助詞", "特殊", "判定詞"):
            sidx += i
            break
    eidx = len(morphemes)
    for i, morpheme in enumerate(reversed(morphemes)):
        if morpheme.pos not in ("助詞", "特殊", "判定詞"):
            eidx -= i
            break
    ret = "".join(m.text for m in morphemes[sidx:eidx])
    if not ret:
        ret = unit.text
    return ret


def softmax(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / (e_x.sum(axis=axis, keepdims=True) + 1e-8)


def sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.exp(-np.abs(x))
    return np.where(x >= 0, 1 / (1 + z), z / (1 + z))


def oc_resolve(cfg: Container, keys: Collection[str]) -> None:
    for key in keys:
        value = getattr(cfg, key)
        if OmegaConf.is_config(value):
            OmegaConf.resolve(value)
        else:
            OmegaConf.is_interpolation(cfg, key)
            setattr(cfg, key, value)
