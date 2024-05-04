import copy
import logging
from dataclasses import dataclass
from typing import Optional, Union

from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from rhoknp import BasePhrase, Document
from rhoknp.cohesion import Argument, EndophoraArgument, ExophoraArgument, ExophoraReferent
from tokenizers import Encoding

from utils.sub_document import extract_target_sentences

logger = logging.getLogger(__file__)


@dataclass
class CohesionBasePhrase:
    """A wrapper class of BasePhrase for cohesion analysis"""

    head_morpheme_global_index: int
    morpheme_global_indices: list[int]
    morphemes: list[str]
    is_target: bool  # 本タスクの解析対象基本句かどうか
    referent_candidates: list["CohesionBasePhrase"]
    # case -> argument_tags / "=" -> mention_tags
    rel2tags: Optional[dict[str, list[str]]] = None


class KyotoExample:
    def __init__(self) -> None:
        self.example_id: int = -1
        self.doc_id: str = ""
        self.phrases: dict[Task, list[CohesionBasePhrase]] = {}
        self.analysis_target_morpheme_indices: list[int] = []
        self.encoding: Optional[Encoding] = None

    def load(
        self,
        document: Document,
        tasks: list[Task],
        task_to_extractor: dict[Task, BaseExtractor],
        task_to_rels: dict[Task, list[str]],
    ):
        self.doc_id = document.doc_id
        for task in tasks:
            extractor: BaseExtractor = task_to_extractor[task]
            self.phrases[task] = wrap_base_phrases(document.base_phrases, extractor, task_to_rels[task])

        analysis_target_morpheme_indices = []
        for sentence in extract_target_sentences(document):
            analysis_target_morpheme_indices += [m.global_index for m in sentence.morphemes]
        self.analysis_target_morpheme_indices = analysis_target_morpheme_indices


def wrap_base_phrases(
    base_phrases: list[BasePhrase],
    extractor: BaseExtractor,
    rel_types: list[str],
) -> list[CohesionBasePhrase]:
    cohesion_base_phrases = [
        CohesionBasePhrase(
            base_phrase.head.global_index,
            [morpheme.global_index for morpheme in base_phrase.morphemes],
            [morpheme.text for morpheme in base_phrase.morphemes],
            is_target=extractor.is_target(base_phrase),
            referent_candidates=[],
        )
        for base_phrase in base_phrases
    ]
    for base_phrase, cohesion_base_phrase in zip(base_phrases, cohesion_base_phrases):
        if cohesion_base_phrase.is_target:
            all_rels = extractor.extract_rels(base_phrase)
            rel_type_to_tags: dict[str, list[str]]
            if isinstance(extractor, (PasExtractor, BridgingExtractor)):
                assert isinstance(all_rels, dict)
                rel_type_to_tags = {rel_type: _get_argument_tags(all_rels[rel_type]) for rel_type in rel_types}
            elif isinstance(extractor, CoreferenceExtractor):
                assert rel_types == ["="]
                assert isinstance(all_rels, list)
                rel_type_to_tags = {"=": _get_referent_tags(all_rels)}
            else:
                raise AssertionError
            cohesion_base_phrase.rel2tags = rel_type_to_tags
        referent_candidates = extractor.get_candidates(base_phrase, base_phrase.document.base_phrases)
        cohesion_base_phrase.referent_candidates = [
            cohesion_base_phrases[cand.global_index] for cand in referent_candidates
        ]
    return cohesion_base_phrases


def _get_argument_tags(arguments: list[Argument]) -> list[str]:
    """Get argument tags.

    Note:
        endophora argument: string of base phrase global index
        exophora argument: exophora referent
        no argument: [NULL]
    """
    argument_tags: list[str] = []
    for argument in arguments:
        if isinstance(argument, EndophoraArgument):
            argument_tag = str(argument.base_phrase.global_index)
        else:
            assert isinstance(argument, ExophoraArgument)
            exophora_referent = copy.copy(argument.exophora_referent)
            exophora_referent.index = None  # 不特定:人１ -> 不特定:人
            argument_tag = f"[{exophora_referent.text}]"  # 不特定:人 -> [不特定:人]
        argument_tags.append(argument_tag)
    return argument_tags or ["[NULL]"]


def _get_referent_tags(referents: list[Union[BasePhrase, ExophoraReferent]]) -> list[str]:
    """Get referent tags.

    Note:
        endophora referent: string of base phrase global index
        exophora referent: exophora referent text wrapped by []
        no referent: [NA]
    """
    mention_tags: list[str] = []
    for referent in referents:
        if isinstance(referent, BasePhrase):
            mention_tag = str(referent.global_index)
        else:
            assert isinstance(referent, ExophoraReferent)
            referent.index = None  # 不特定:人１ -> 不特定:人
            mention_tag = f"[{referent.text}]"  # 著者 -> [著者]
        mention_tags.append(mention_tag)
    return mention_tags or ["[NA]"]
