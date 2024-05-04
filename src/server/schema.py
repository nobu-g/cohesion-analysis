from typing import Optional

from humps import camelize
from pydantic import BaseModel, ConfigDict


class Relation(BaseModel):
    model_config = ConfigDict(alias_generator=camelize, populate_by_name=True)

    rel_type: str
    target: Optional[int]
    exo_target: str
    is_relaxed: bool
    confidence: Optional[float]
    pas_type: Optional[str]  # overt, dep, intra, inter, exo
    within_sub_doc: bool


class Phrase(BaseModel):
    model_config = ConfigDict(alias_generator=camelize, populate_by_name=True)

    sidx: int  # sentence index
    dtid: int  # document-wide tag id
    surf: str  # surface form
    core: str  # core form
    targets: list[str]  # target relation types
    relations: list[Relation]  # relations
    features: list[str]  # related features: 体言, 用言:動, 用言:形, 用言:判, 非用言格解析


class OutputData(BaseModel):
    model_config = ConfigDict(alias_generator=camelize, populate_by_name=True)

    phrases: list[Phrase]
    rel_types: list[str]
    model: str
    knp: str
