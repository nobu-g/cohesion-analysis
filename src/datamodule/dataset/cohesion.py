import hashlib
import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from cohesion_tools.extractors import BridgingExtractor, CoreferenceExtractor, PasExtractor
from cohesion_tools.extractors.base import BaseExtractor
from cohesion_tools.task import Task
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from omegaconf import ListConfig
from rhoknp import Document
from rhoknp.cohesion import ExophoraReferent, ExophoraReferentType
from tokenizers import Encoding
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy

from datamodule.dataset.base import BaseDataset
from datamodule.example import KyotoExample
from datamodule.example.kyoto import CohesionBasePhrase
from utils.util import IGNORE_INDEX, sigmoid, softmax

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputFeatures:
    """
    A dataclass which represents a raw model input.
    The attributes of this class correspond to arguments of forward method of each model.
    """

    example_id: int
    input_ids: list[int]
    attention_mask: list[bool]
    token_type_ids: list[int]
    source_mask: list[bool]  # loss を計算する対象の基本句かどうか（文書分割によって文脈としてのみ使用される場合は False）
    target_mask: list[list[list[bool]]]  # source と関係を持つ候補かどうか（後ろと共参照はしないなど）
    source_label: list[list[int]]  # 解析対象基本句かどうか
    target_label: list[list[list[float]]]  # source と関係を持つかどうか


class CohesionDataset(BaseDataset):
    def __init__(
        self,
        knp_path: Union[str, Path],
        tasks: ListConfig,
        cases: ListConfig,
        bar_rels: ListConfig,
        max_seq_length: int,
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        exophora_referents: ListConfig,
        special_tokens: ListConfig,
        training: bool,
    ) -> None:
        self.knp_path = Path(knp_path)
        self.exophora_referents = [ExophoraReferent(s) for s in exophora_referents]
        self.special_tokens: list[str] = list(special_tokens)
        super().__init__(
            self.knp_path,
            document_split_stride=document_split_stride,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            training=training,
        )
        self.tasks: list[Task] = [Task(task) for task in tasks]
        self.cases: list[str] = list(cases)
        self.bar_rels: list[str] = list(bar_rels)
        self.special_to_index: dict[str, int] = {
            token: max_seq_length - len(self.special_tokens) + i for i, token in enumerate(self.special_tokens)
        }
        exophora_referent_types: list[ExophoraReferentType] = [er.type for er in self.exophora_referents]
        self.task_to_extractor: dict[Task, BaseExtractor] = {
            Task.PAS_ANALYSIS: PasExtractor(
                list(self.cases),
                exophora_referent_types,
                verbal_predicate=True,
                nominal_predicate=True,
            ),
            Task.BRIDGING_REFERENCE_RESOLUTION: BridgingExtractor(self.bar_rels, exophora_referent_types),
            Task.COREFERENCE_RESOLUTION: CoreferenceExtractor(exophora_referent_types),
        }
        self.task_to_rels: dict[Task, list[str]] = {
            Task.PAS_ANALYSIS: self.cases,
            Task.BRIDGING_REFERENCE_RESOLUTION: self.bar_rels,
            Task.COREFERENCE_RESOLUTION: ["="],
        }

        self.examples: list[KyotoExample] = self._load_examples(self.documents, str(knp_path))

        self.special_encoding: Encoding = self.tokenizer(
            self.special_tokens,
            is_split_into_words=True,
            padding=PaddingStrategy.DO_NOT_PAD,
            truncation=False,
            add_special_tokens=False,
        ).encodings[0]

    @property
    def special_indices(self) -> list[int]:
        return list(self.special_to_index.values())

    @property
    def num_special_tokens(self) -> int:
        return len(self.special_tokens)

    @property
    def rel_types(self) -> list[str]:
        return [rel_type for task in self.tasks for rel_type in self.task_to_rels[task]]

    def _load_examples(self, documents: list[Document], documents_path: str) -> list[KyotoExample]:
        examples = []
        load_cache: bool = "COHESION_DISABLE_CACHE" not in os.environ and "COHESION_OVERWRITE_CACHE" not in os.environ
        save_cache: bool = "COHESION_DISABLE_CACHE" not in os.environ
        cohesion_cache_dir: Path = Path(
            os.environ.get("COHESION_CACHE_DIR", f'/tmp/{os.environ["USER"]}/cohesion_cache'),
        )
        for document in tqdm(documents, desc="Loading examples"):
            # give enough options to identify examples
            hash_ = self._hash(documents_path, self.tasks, self.task_to_rels, self.task_to_extractor)
            example_cache_path = cohesion_cache_dir / hash_ / f"{document.doc_id}.pkl"
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open(mode="rb") as f:
                    try:
                        example = pickle.load(f)
                    except EOFError:
                        example = self._load_example_from_document(document)
            else:
                example = self._load_example_from_document(document)
                if save_cache:
                    self._save_cache(example, example_cache_path)  # type: ignore
            examples.append(example)
        examples = self._post_process_examples(examples)
        if len(examples) == 0:
            logger.error(
                "No examples to process. "
                f"Make sure there exist any documents in {self.knp_path} and they are not too long.",
            )
        return examples

    @rank_zero_only
    def _save_cache(self, example, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open(mode="wb") as f:
            pickle.dump(example, f)

    @staticmethod
    def _hash(*args) -> str:
        string = "".join(repr(a) for a in args)
        return hashlib.md5(string.encode()).hexdigest()

    def _post_process_examples(self, examples: list[KyotoExample]) -> list[KyotoExample]:
        idx = 0
        filtered = []
        for example in examples:
            phrases = next(iter(example.phrases.values()))
            encoding: Encoding = self.tokenizer(
                " ".join([morpheme for phrase in phrases for morpheme in phrase.morphemes]),
                is_split_into_words=False,
                padding=PaddingStrategy.MAX_LENGTH,
                truncation=False,
                max_length=self.max_seq_length - self.num_special_tokens,
            ).encodings[0]
            if len(encoding.ids) > self.max_seq_length - self.num_special_tokens:
                continue
            example.encoding = encoding
            example.example_id = idx
            filtered.append(example)
            idx += 1
        return filtered

    def _load_example_from_document(self, document: Document) -> KyotoExample:
        example = KyotoExample()
        example.load(
            document,
            tasks=self.tasks,
            task_to_extractor=self.task_to_extractor,
            task_to_rels=self.task_to_rels,
        )
        return example

    def dump_relation_prediction(
        self,
        relation_logits: np.ndarray,  # (rel, seq, seq), subword level
        example: KyotoExample,
    ) -> np.ndarray:  # (phrase, rel, phrase+special)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        predictions: list[np.ndarray] = []
        task_and_rels = [(task, rel) for task in self.tasks for rel in self.task_to_rels[task]]
        assert len(relation_logits) == len(task_and_rels) == len(self.rel_types)
        for (task, _), logits in zip(task_and_rels, relation_logits):
            predictions.append(self._token_to_phrase_level(logits, example.phrases[task], example.encoding))
        return np.array(predictions).transpose(1, 0, 2)  # (phrase, rel, phrase+special)

    def _token_to_phrase_level(
        self,
        token_level_logits_matrix: np.ndarray,  # (seq, seq)
        phrases: list[CohesionBasePhrase],
        encoding: Encoding,
    ) -> np.ndarray:  # (phrase, phrase+special)
        phrase_level_scores_matrix: list[np.ndarray] = []
        for phrase in phrases:
            token_index_span: tuple[int, int] = encoding.word_to_tokens(phrase.head_morpheme_global_index)
            # Use the head subword as the representative of the source word.
            # Cast to built-in list because list operation is faster than numpy array operation.
            token_level_logits: list[float] = token_level_logits_matrix[token_index_span[0]].tolist()  # (seq)
            phrase_level_logits: list[float] = []
            for target_phrase in phrases:
                # tgt 側は複数のサブワードから構成されるため平均を取る
                token_index_span = encoding.word_to_tokens(target_phrase.head_morpheme_global_index)
                sliced_token_level_logits: list[float] = token_level_logits[slice(*token_index_span)]
                phrase_level_logits.append(sum(sliced_token_level_logits) / len(sliced_token_level_logits))
            phrase_level_logits += [token_level_logits[idx] for idx in self.special_indices]
            assert len(phrase_level_logits) == len(phrases) + len(self.special_to_index)
            phrase_level_scores_matrix.append(softmax(np.array(phrase_level_logits)))
        return np.array(phrase_level_scores_matrix)

    def dump_source_mask_prediction(
        self,
        token_level_source_mask_logits: np.ndarray,  # (task, seq)
        example: KyotoExample,
    ) -> np.ndarray:  # (phrase, task)
        """1 example 中に存在する基本句それぞれに対してシステム予測のリストを返す．"""
        assert example.encoding is not None, "encoding isn't set"
        phrase_task_scores: list[list[float]] = []
        token_level_source_mask_scores = sigmoid(token_level_source_mask_logits)
        assert len(token_level_source_mask_scores) == len(self.tasks)
        for task, token_level_scores in zip(self.tasks, token_level_source_mask_scores.tolist()):
            phrase_level_scores: list[float] = []
            for phrase in example.phrases[task]:
                token_index_span: tuple[int, int] = example.encoding.word_to_tokens(phrase.head_morpheme_global_index)
                sliced_token_level_scores: list[float] = token_level_scores[slice(*token_index_span)]
                phrase_level_scores.append(sum(sliced_token_level_scores) / len(sliced_token_level_scores))
            phrase_task_scores.append(phrase_level_scores)
        return np.array(phrase_task_scores).transpose()

    def _convert_example_to_feature(self, example: KyotoExample) -> InputFeatures:
        """Loads a data file into a list of `InputBatch`s."""

        scores_set: list[list[list[float]]] = []  # (rel, src, tgt)
        candidates_set: list[list[list[int]]] = []  # (rel, src, tgt)
        for task in self.tasks:
            for rel in self.task_to_rels[task]:
                scores, candidates = self._convert_annotation_to_feature(example.phrases[task], rel, example.encoding)
                scores_set.append(scores)
                candidates_set.append(candidates)

        assert example.encoding is not None, "encoding isn't set"

        source_mask = [False] * self.max_seq_length
        for global_index in example.analysis_target_morpheme_indices:
            for token_index in range(*example.encoding.word_to_tokens(global_index)):
                source_mask[token_index] = True

        is_analysis_targets: list[list[int]] = []  # (task, src)
        for task in self.tasks:
            is_targets: list[int] = [IGNORE_INDEX] * self.max_seq_length
            for phrase in example.phrases[task]:
                token_index_span: tuple[int, int] = example.encoding.word_to_tokens(phrase.head_morpheme_global_index)
                for token_index in range(*token_index_span):
                    is_targets[token_index] = int(phrase.is_target)
            is_analysis_targets.append(is_targets)

        merged_encoding: Encoding = Encoding.merge([example.encoding, self.special_encoding])
        return InputFeatures(
            example_id=example.example_id,
            input_ids=merged_encoding.ids,
            attention_mask=merged_encoding.attention_mask,
            token_type_ids=merged_encoding.type_ids,
            source_mask=source_mask,
            target_mask=[
                [[(x in cands) for x in range(self.max_seq_length)] for cands in candidates]
                for candidates in candidates_set
            ],  # False -> mask, True -> keep
            source_label=is_analysis_targets,
            target_label=scores_set,
        )

    def _convert_annotation_to_feature(
        self,
        phrases: list[CohesionBasePhrase],
        rel_type: str,
        encoding: Encoding,
    ) -> tuple[list[list[float]], list[list[int]]]:
        scores_set: list[list[float]] = [[0.0] * self.max_seq_length for _ in range(self.max_seq_length)]
        candidates_set: list[list[int]] = [[] for _ in range(self.max_seq_length)]

        for phrase in phrases:
            scores: list[float] = [0.0] * self.max_seq_length
            # phrase.rel2tags が None の場合は推論時，もしくは学習対象外の基本句．
            # その場合は scores が全てゼロになるため loss が計算されない．
            if phrase.rel2tags is not None:
                # 学習・解析対象基本句
                for arg_string in phrase.rel2tags[rel_type]:
                    if arg_string in self.special_to_index:
                        token_index = self.special_to_index[arg_string]
                        scores[token_index] = 1.0
                    else:
                        token_index_span: tuple[int, int] = encoding.word_to_tokens(
                            phrases[int(arg_string)].head_morpheme_global_index,
                        )
                        # 1.0 for all subwords that compose the target word
                        for token_index in range(*token_index_span):
                            scores[token_index] = 1.0

            token_level_candidates: list[int] = []
            for candidate in phrase.referent_candidates:
                token_index_span = encoding.word_to_tokens(candidate.head_morpheme_global_index)
                token_level_candidates += range(*token_index_span)
            token_level_candidates += self.special_indices

            token_index_span = encoding.word_to_tokens(phrase.head_morpheme_global_index)
            # use the head subword as the representative of the source word
            scores_set[token_index_span[0]] = scores
            candidates_set[token_index_span[0]] = token_level_candidates

        return scores_set, candidates_set  # token level

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> InputFeatures:
        return self._convert_example_to_feature(self.examples[idx])
