from collections import defaultdict
from statistics import mean
from typing import Any, Optional

import numpy as np
import torch
from cohesion_tools.evaluators.cohesion import CohesionEvaluator, CohesionScore
from cohesion_tools.evaluators.utils import F1Metric
from cohesion_tools.extractors import PasExtractor
from cohesion_tools.task import Task
from rhoknp import Document, Sentence
from torchmetrics import Metric

from datamodule.dataset.cohesion import CohesionDataset
from datamodule.example import KyotoExample
from utils.sub_document import extract_target_sentences, to_orig_doc_id
from writer.knp import PredictionKNPWriter


class CohesionMetric(Metric):
    full_state_update: bool = False
    STATE_NAMES = (
        "example_ids",
        "relation_logits",
        "source_mask_logits",
    )

    def __init__(self, analysis_target_threshold: float = 0.5) -> None:
        super().__init__()
        self.analysis_target_threshold: float = analysis_target_threshold
        self.dataset: Optional[CohesionDataset] = None
        # Metric state variables can either be torch.Tensor or an empty list which can be used to store torch.Tensors`.
        # i.e. Expected metric state to be either a Tensor or a list of Tensor
        for state_name in self.STATE_NAMES:
            self.add_state(state_name, default=[])
        self.example_ids: torch.Tensor  # (N)
        self.relation_logits: torch.Tensor  # (N, rel, seq, seq)
        self.source_mask_logits: torch.Tensor  # (N, task, seq)

    def set_attributes(self, **kwargs: dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update(self, kwargs: dict[str, torch.Tensor]) -> None:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            value = kwargs[state_name]
            # https://github.com/pytorch/pytorch/issues/90245
            if isinstance(value, torch.BoolTensor):
                value = value.long()
            state.append(value)

    def compute(self) -> dict[str, float]:
        for state_name in self.STATE_NAMES:
            state = getattr(self, state_name)
            if not isinstance(state, torch.Tensor):
                setattr(self, state_name, torch.cat(state, dim=0))

        predicted_documents, gold_documents = self._build_documents()

        metrics: dict[str, float] = {}
        assert self.dataset is not None, "dataset is not set"
        metrics.update(
            self._compute_analysis_target_metrics(
                dataset=self.dataset,
                predicted_documents=predicted_documents,
                gold_documents=gold_documents,
            )
        )
        metrics.update(
            self._compute_cohesion_metrics(
                dataset=self.dataset,
                predicted_documents=predicted_documents,
                gold_documents=gold_documents,
            )
        )
        return metrics

    def _build_documents(self) -> tuple[list[Document], list[Document]]:
        orig_did_to_sentences: dict[str, list[Sentence]] = defaultdict(list)
        orig_did_to_gold_sentences: dict[str, list[Sentence]] = defaultdict(list)
        assert self.dataset is not None, "dataset is not set"
        knp_writer = PredictionKNPWriter(self.dataset)
        assert len(self.example_ids) == len(self.relation_logits) == len(self.source_mask_logits)
        for example_id, relation_logits, source_mask_logits in zip(
            self.example_ids, self.relation_logits, self.source_mask_logits
        ):
            gold_example: KyotoExample = self.dataset.examples[example_id.item()]
            gold_document: Document = self.dataset.doc_id2document[gold_example.doc_id]
            # (phrase, rel, phrase+special)
            relation_prediction: np.ndarray = self.dataset.dump_relation_prediction(
                relation_logits.cpu().numpy(), gold_example
            )
            # (phrase, rel)
            phrase_selection_prediction: np.ndarray = np.argmax(relation_prediction, axis=2)
            # (phrase, task)
            source_mask_prediction: np.ndarray = self.dataset.dump_source_mask_prediction(
                source_mask_logits.cpu().numpy(), gold_example
            )
            is_analysis_target: np.ndarray = source_mask_prediction >= self.analysis_target_threshold  # (phrase, task)

            predicted_document = gold_document.reparse()
            predicted_document.doc_id = gold_example.doc_id
            knp_writer.add_rel_tags(
                predicted_document, phrase_selection_prediction.tolist(), is_analysis_target.tolist()
            )
            orig_doc_id = to_orig_doc_id(gold_example.doc_id)
            for sentence in extract_target_sentences(predicted_document):
                orig_did_to_sentences[orig_doc_id].append(sentence)
            for sentence in extract_target_sentences(gold_document):
                orig_did_to_gold_sentences[orig_doc_id].append(sentence)
        return (
            [Document.from_sentences(sentences) for sentences in orig_did_to_sentences.values()],
            [Document.from_sentences(sentences) for sentences in orig_did_to_gold_sentences.values()],
        )

    @staticmethod
    def _compute_analysis_target_metrics(
        dataset: CohesionDataset, predicted_documents: list[Document], gold_documents: list[Document]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        task_to_analysis_target_metric: dict[Task, F1Metric] = {task: F1Metric() for task in dataset.tasks}
        for predicted_document, gold_document in zip(predicted_documents, gold_documents):
            assert predicted_document.doc_id == gold_document.doc_id
            assert len(predicted_document.base_phrases) == len(gold_document.base_phrases)
            for predicted_base_phrase, gold_base_phrase in zip(
                predicted_document.base_phrases, gold_document.base_phrases
            ):
                for task in dataset.tasks:
                    metric = task_to_analysis_target_metric[task]
                    extractor = dataset.task_to_extractor[task]
                    predicted_is_target = predicted_base_phrase.features.get(task.value + "対象", False)
                    assert isinstance(predicted_is_target, bool)
                    gold_is_target: bool = extractor.is_target(gold_base_phrase)
                    if predicted_is_target is True:
                        if gold_is_target is True:
                            metric.tp += 1
                        metric.tp_fp += 1
                    if gold_is_target is True:
                        if predicted_is_target is True:
                            pass
                        metric.tp_fn += 1
        for task, metric in task_to_analysis_target_metric.items():
            for metric_name in ("f1", "precision", "recall"):
                metrics[f"{task.value}_target_{metric_name}"] = getattr(metric, metric_name)
        for metric_name in ("f1", "precision", "recall"):
            metrics[f"cohesion_target_{metric_name}"] = mean(
                getattr(metric, metric_name) for metric in task_to_analysis_target_metric.values()
            )
        return metrics

    @staticmethod
    def _compute_cohesion_metrics(
        dataset: CohesionDataset,
        predicted_documents: list[Document],
        gold_documents: list[Document],
    ):
        metrics: dict[str, float] = {}
        pas_extractor = dataset.task_to_extractor[Task.PAS_ANALYSIS]
        assert isinstance(pas_extractor, PasExtractor)
        evaluator = CohesionEvaluator(
            tasks=dataset.tasks,
            exophora_referent_types=[e.type for e in dataset.exophora_referents],
            pas_cases=dataset.cases,
            bridging_rel_types=dataset.bar_rels,
        )
        score: CohesionScore = evaluator.run(gold_documents=gold_documents, predicted_documents=predicted_documents)

        for task_str, analysis_type_to_metric in score.to_dict().items():
            for analysis_type, metric in analysis_type_to_metric.items():
                key = task_str
                if analysis_type != "all":
                    key += f"_{analysis_type}"
                metrics[key + "_f1"] = metric.f1
        metrics["cohesion_analysis_f1"] = mean(
            metrics[key] for key in ("pas_f1", "bridging_f1", "coreference_f1") if key in metrics
        )
        return metrics

    def set_properties(self, kwargs: dict[str, Any]) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
