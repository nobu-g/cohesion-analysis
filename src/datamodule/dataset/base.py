import logging
from pathlib import Path
from typing import Union

from rhoknp import Document, Sentence
from rhoknp.utils.reader import chunk_by_document
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from utils.sub_document import SequenceSplitter, SpanCandidate, to_sub_doc_id

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
        self,
        source: Union[Path, str, list[Document]],
        document_split_stride: int,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
        training: bool,
    ) -> None:
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_seq_length: int = max_seq_length
        self.training: bool = training
        self.orig_documents: list[Document]
        if isinstance(source, (Path, str)):
            source = Path(source)
            self.orig_documents = self._load_documents(source)
        else:
            self.orig_documents = source
        self.doc_id2document: dict[str, Document] = {}
        for orig_document in self.orig_documents:
            self.doc_id2document.update(
                {
                    document.doc_id: document
                    for document in self._split_document(
                        orig_document,
                        self.max_seq_length - len(self.tokenizer.additional_special_tokens) - 2,  # -2: [CLS] and [SEP]
                        stride=document_split_stride,
                    )
                },
            )

    @property
    def documents(self) -> list[Document]:
        return list(self.doc_id2document.values())

    @staticmethod
    def _load_documents(document_path: Path, ext: str = "knp") -> list[Document]:
        """Search KNP format files in the given path and load them into document object.
        If the path is a directory, search files in the directory. Each file is assumed to have one document.
        If the path is a file, load contents in the file. The file is assumed to have multiple documents.

        Args:
            document_path: Path to the directory or file.
            ext: Extension of the KNP format file. Default is "knp".

        Returns:
            List of document objects.
        """
        documents = []
        if document_path.is_dir():
            for path in sorted(document_path.glob(f"*.{ext}")):
                documents.append(Document.from_knp(path.read_text()))
        else:
            with document_path.open() as f:
                for knp_text in chunk_by_document(f):
                    documents.append(Document.from_knp(knp_text))
        return documents

    def _split_document(self, document: Document, max_token_length: int, stride: int) -> list[Document]:
        sentence_tokens = [self._get_tokenized_len(sentence) for sentence in document.sentences]
        if sum(sentence_tokens) <= max_token_length:
            return [document]

        splitter = SequenceSplitter(sentence_tokens, max_token_length, stride)
        sub_documents: list[Document] = []
        sub_idx = 0
        for span in splitter.split_into_spans():
            assert isinstance(span, SpanCandidate)
            sentences = document.sentences[span.start : span.end]
            sub_document = Document.from_sentences(sentences)
            sub_doc_id = to_sub_doc_id(document.doc_id, sub_idx, stride=span.stride)
            sub_document.doc_id = sub_doc_id
            for sentence, sub_sentence in zip(sentences, sub_document.sentences):
                sub_sentence.comment = sentence.comment
            sub_documents.append(sub_document)
            sub_idx += 1
        return sub_documents

    def _get_tokenized_len(self, source: Union[Document, Sentence]) -> int:
        return len(self.tokenizer.tokenize(" ".join(m.text for m in source.morphemes)))
