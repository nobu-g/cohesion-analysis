import io
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO, Union

from cohesion_tools.task import Task
from rhoknp import Document

from datamodule.dataset import CohesionDataset
from datamodule.example import KyotoExample
from utils.sub_document import extract_target_sentences, to_orig_doc_id
from utils.util import CamelCaseDataClassJsonMixin

logger = logging.getLogger(__file__)


@dataclass
class RelProb(CamelCaseDataClassJsonMixin):
    rel: str
    probs: list[float]
    special_probs: list[float]


@dataclass
class Phrase(CamelCaseDataClassJsonMixin):
    rel_probs: list[RelProb]


@dataclass
class DocumentProb(CamelCaseDataClassJsonMixin):
    doc_id: str
    special_tokens: list[str]
    phrases: list[Phrase]


class ProbabilityJsonWriter:
    """A class to write the system prediction probabilities in a JSON format.

    Args:
        dataset: 解析対象のデータセット
    """

    def __init__(self, dataset: CohesionDataset) -> None:
        self.examples: list[KyotoExample] = dataset.examples
        self.cases: list[str] = dataset.cases
        self.tasks: list[Task] = dataset.tasks
        self.rel_types: list[str] = dataset.rel_types
        self.documents: list[Document] = dataset.documents
        self.special_tokens: list[str] = dataset.special_tokens
        self.num_special_tokens = dataset.num_special_tokens

    def write(
        self,
        probabilities: dict[int, list[list[list[float]]]],
        destination: Union[Path, TextIO, None] = None,
        skip_untagged: bool = True,
    ) -> list[DocumentProb]:
        """Write final predictions to files.

        Args:
            probabilities (dict[int, list[list[list[float]]]]): example_idをkeyとする基本句単位モデル出力
            destination (Union[Path, TextIO, None]): 解析済み文書の出力先 (default: None)
            skip_untagged (bool): 解析に失敗した文書を出力しないかどうか (default: True)
        """

        if isinstance(destination, Path):
            if destination.is_file():
                raise ValueError(f"File already exists: {destination}")
            logger.info(f"Writing predictions to: {destination}")
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            logger.warning("invalid output destination")

        did2probability: dict[str, list] = {self.examples[eid].doc_id: prob for eid, prob in probabilities.items()}
        did2orig_probability: dict[str, list[list[list[float]]]] = defaultdict(list)
        did2num_pre_pads: dict[str, list[int]] = defaultdict(list)
        for document in self.documents:
            doc_id = document.doc_id
            if doc_id in did2probability:
                probability: list[list[list[float]]] = did2probability[doc_id]  # (phrase, rel, 0 or phrase+special)
            else:
                if skip_untagged:
                    continue
                probability = [[[]] * len(self.rel_types)] * len(document.base_phrases)

            orig_doc_id = to_orig_doc_id(doc_id)
            target_sentences = extract_target_sentences(document)
            num_target_base_phrases = sum(len(s.base_phrases) for s in target_sentences)
            did2orig_probability[orig_doc_id] += probability[-num_target_base_phrases:]
            num_accum_base_phrases = len(did2orig_probability[orig_doc_id])
            did2num_pre_pads[orig_doc_id] += [
                num_accum_base_phrases - num_target_base_phrases,
            ] * num_target_base_phrases

        doc_probs = []
        for doc_id, probs in did2orig_probability.items():
            if destination is None:
                continue
            num_orig_phrases = len(probs)
            num_pre_pads = did2num_pre_pads[doc_id]
            assert len(num_pre_pads) == num_orig_phrases
            phrases = []
            for prob, num_pre_pad in zip(probs, num_pre_pads):
                rel_probs = []
                for rel, ps in zip(self.rel_types, prob):
                    if not ps:
                        continue
                    num_post_pad = num_orig_phrases + self.num_special_tokens - num_pre_pad - len(ps)
                    rel_probs.append(
                        RelProb(
                            rel=rel,
                            probs=[-1.0] * num_pre_pad + ps[: -self.num_special_tokens] + [-1.0] * num_post_pad,
                            special_probs=ps[-self.num_special_tokens :],
                        ),
                    )
                phrases.append(Phrase(rel_probs=rel_probs))
            doc_prob = DocumentProb(
                doc_id=doc_id,
                special_tokens=self.special_tokens,
                phrases=phrases,
            )
            doc_probs.append(doc_prob)
            json_string = doc_prob.to_json(ensure_ascii=False, indent=2, sort_keys=False)
            if isinstance(destination, Path):
                destination.joinpath(f"{doc_id}.json").write_text(json_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(json_string)
        return doc_probs
