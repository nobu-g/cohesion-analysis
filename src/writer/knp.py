import logging
from typing import Optional

from cohesion_tools.task import Task
from rhoknp import BasePhrase, Document
from rhoknp.cohesion import ExophoraReferent, RelTag

from datamodule.dataset import CohesionDataset

logger = logging.getLogger(__file__)


class PredictionKNPWriter:
    """A class to write the system output in a KNP format.

    Args:
        dataset: 解析対象のデータセット
        flip_writer_reader_according_to_type_id: 話者ラベルが付与されていた場合にラベルに従って著者・読者を入れ替えるかどうか
    """

    def __init__(self, dataset: CohesionDataset, flip_writer_reader_according_to_type_id: bool) -> None:
        self.rel_types: list[str] = dataset.rel_types
        self.exophora_referents: list[ExophoraReferent] = dataset.exophora_referents
        self.special_tokens: list[str] = dataset.special_tokens
        self.flip_writer_reader_according_to_type_id: bool = flip_writer_reader_according_to_type_id
        self.tasks: list[Task] = dataset.tasks
        self.task_to_rels: dict[Task, list[str]] = dataset.task_to_rels

    def add_rel_tags(
        self,
        document: Document,
        phrase_selection_prediction: list[list[int]],  # (phrase, rel)
        sid_to_type_id: dict[str, int],
        is_analysis_target: list[list[bool]],  # (phrase, task)
    ) -> None:
        assert len(document.base_phrases) == len(phrase_selection_prediction) == len(is_analysis_target)
        for base_phrase, selected_phrases, is_targets in zip(
            document.base_phrases, phrase_selection_prediction, is_analysis_target
        ):
            base_phrase.rel_tags.clear()
            rel_type_to_selected_phrase = dict(zip(self.rel_types, selected_phrases))
            for task, is_target in zip(self.tasks, is_targets):
                base_phrase.features[task.value + "対象"] = is_target
                if is_target is False:
                    continue
                for rel_type in self.task_to_rels[task]:
                    rel_tag = self._to_rel_tag(
                        rel_type,
                        rel_type_to_selected_phrase[rel_type],
                        document.base_phrases,
                        flip_reader_writer=(
                            self.flip_writer_reader_according_to_type_id is True
                            and sid_to_type_id.get(base_phrase.sentence.sid) == 1
                        ),
                    )
                    if rel_tag is not None:
                        base_phrase.rel_tags.append(rel_tag)

    def _to_rel_tag(
        self,
        rel_type: str,
        predicted_base_phrase_global_index: int,
        base_phrases: list[BasePhrase],
        flip_reader_writer: bool,
    ) -> Optional[RelTag]:
        exophora_referents_set = {str(e) for e in self.exophora_referents}
        if 0 <= predicted_base_phrase_global_index < len(base_phrases):
            # endophora
            prediction_base_phrase: BasePhrase = base_phrases[predicted_base_phrase_global_index]
            return RelTag(
                type=rel_type,
                target=prediction_base_phrase.head.text,
                sid=prediction_base_phrase.sentence.sid,
                base_phrase_index=prediction_base_phrase.index,
                mode=None,
            )
        elif 0 <= predicted_base_phrase_global_index - len(base_phrases) < len(self.special_tokens):
            # exophora
            special_token = self.special_tokens[predicted_base_phrase_global_index - len(base_phrases)]
            stripped_special_token = special_token.removeprefix("[").removesuffix("]")
            if flip_reader_writer is True:
                flip_map = {"著者": "読者", "読者": "著者"}
                stripped_special_token = flip_map.get(stripped_special_token, stripped_special_token)
            if stripped_special_token in exophora_referents_set:  # exclude [NULL] and [NA]
                return RelTag(
                    type=rel_type,
                    target=stripped_special_token,
                    sid=None,
                    base_phrase_index=None,
                    mode=None,
                )
        else:
            raise ValueError(f"invalid predicted base phrase index: {predicted_base_phrase_global_index}")

        return None
