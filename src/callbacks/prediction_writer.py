import io
import logging
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Optional, TextIO, Union

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import BasePredictionWriter
from rhoknp import Document, Sentence
from torch.utils.data import Dataset
from typing_extensions import override

from datamodule.dataset import CohesionDataset
from utils.sub_document import extract_target_sentences, to_orig_doc_id
from writer.json import ProbabilityJsonWriter
from writer.knp import PredictionKNPWriter

logger = logging.getLogger(__name__)


class CohesionWriter(BasePredictionWriter):
    def __init__(
        self,
        knp_destination: Union[Path, TextIO, None] = None,
        json_destination: Union[Path, TextIO, None] = None,
        analysis_target_threshold: float = 0.5,
    ) -> None:
        super().__init__(write_interval="epoch")

        self.knp_destination: Union[Path, TextIO, None] = knp_destination
        self.json_destination: Union[Path, TextIO, None] = json_destination
        for dest in (self.knp_destination, self.json_destination):
            if dest is None:
                continue
            assert isinstance(dest, (Path, TextIO)), f"destination must be either Path or TextIO, but got {type(dest)}"
            if isinstance(dest, Path):
                dest.mkdir(parents=True, exist_ok=True)
        self.analysis_target_threshold: float = analysis_target_threshold

    def write_on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass

    @override
    def write_on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Sequence[Any],
        batch_indices: Optional[Sequence[Any]] = None,
    ) -> None:
        dataset: Dataset = trainer.predict_dataloaders.dataset  # type: ignore[union-attr]
        assert isinstance(dataset, CohesionDataset)
        orig_did_to_sentences: dict[str, list[Sentence]] = defaultdict(list)
        knp_writer = PredictionKNPWriter(dataset)
        json_writer = ProbabilityJsonWriter(dataset)
        eid_to_relation_prediction: dict[int, list[list[list[float]]]] = {}
        for pred in predictions:
            batch_example_ids = pred["example_ids"]  # (b)
            batch_relation_logits = pred["relation_logits"]  # (b, rel, seq, seq)
            batch_source_mask_logits = pred["source_mask_logits"]  # (b, task, seq)
            assert len(batch_relation_logits) == len(batch_example_ids) == len(batch_source_mask_logits)
            for example_id, relation_logits, source_mask_logits in zip(
                batch_example_ids, batch_relation_logits, batch_source_mask_logits
            ):
                example = dataset.examples[example_id.item()]
                document: Document = dataset.doc_id2document[example.doc_id]
                # (phrase, rel, phrase+special)
                relation_prediction: np.ndarray = dataset.dump_relation_prediction(
                    relation_logits.cpu().numpy(), example
                )
                # (phrase, rel)
                phrase_selection_prediction: np.ndarray = np.argmax(relation_prediction, axis=2)
                # (phrase, task)
                source_mask_prediction: np.ndarray = dataset.dump_source_mask_prediction(
                    source_mask_logits.cpu().numpy(), example
                )
                # (phrase, task)
                is_analysis_target: np.ndarray = source_mask_prediction >= self.analysis_target_threshold
                predicted_document = document.reparse()
                predicted_document.doc_id = example.doc_id
                knp_writer.add_rel_tags(
                    predicted_document, phrase_selection_prediction.tolist(), is_analysis_target.tolist()
                )
                eid_to_relation_prediction[example_id.item()] = relation_prediction.tolist()
                orig_doc_id = to_orig_doc_id(example.doc_id)
                for sentence in extract_target_sentences(predicted_document):
                    orig_did_to_sentences[orig_doc_id].append(sentence)

        for sentences in orig_did_to_sentences.values():
            document = Document.from_sentences(sentences)
            self.write_document(document)

        _ = json_writer.write(eid_to_relation_prediction, destination=self.json_destination)

    def write_document(self, document: Document) -> None:
        if isinstance(self.knp_destination, Path):
            self.knp_destination.mkdir(parents=True, exist_ok=True)
            self.knp_destination.joinpath(f"{document.doc_id}.knp").write_text(document.to_knp())
        elif isinstance(self.knp_destination, io.TextIOBase):
            self.knp_destination.write(document.to_knp())
