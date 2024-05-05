import json
import os
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import ListConfig
from rhoknp import Document
from transformers import AutoTokenizer, PreTrainedTokenizerBase

here = Path(__file__).parent
sys.path.append(str(here.parent / "src"))

from datamodule.dataset import CohesionDataset  # noqa: E402

INF = 2**10
DATA_DIR = here / "data"

os.environ["COHESION_DISABLE_CACHE"] = "1"


@pytest.fixture()
def fixture_input_tensor():
    # コイン ト ##ス を 行う [NULL] [NA]
    # (b, seq, case, seq) = (1, 7, 4, 7)
    input_tensor = torch.tensor(
        [
            [
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -0.2],  # coref
                ],  # コイン
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-0.5, -INF, -INF, -INF, -INF, -INF, -0.1],  # coref
                ],  # ト
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # ##ス
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # を
                [
                    [-0.2, -0.5, -INF, -INF, -INF, 0.50, -INF],  # ガ
                    [-0.5, -0.4, -INF, -INF, -INF, -0.9, -INF],  # ヲ
                    [-0.3, -0.3, -INF, -INF, -INF, -0.1, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # 行う
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # [NULL]
                [
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ガ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ヲ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # ニ
                    [-INF, -INF, -INF, -INF, -INF, -INF, -INF],  # coref
                ],  # [NA]
            ],  # コイントスを行う
        ],
    )
    return input_tensor


@pytest.fixture()
def fixture_documents_pred():
    return [Document.from_knp(path.read_text()) for path in sorted(DATA_DIR.joinpath("pred").glob("*.knp"))]


@pytest.fixture()
def fixture_documents_gold():
    return [Document.from_knp(path.read_text()) for path in sorted(DATA_DIR.joinpath("gold").glob("*.knp"))]


@pytest.fixture()
def fixture_scores():
    with DATA_DIR.joinpath("expected/score/0.json").open() as f:
        yield json.load(f)


@pytest.fixture()
def exophora_referents() -> list[str]:
    return ["著者", "読者", "不特定:人", "不特定:物"]


@pytest.fixture()
def pas_cases() -> list[str]:
    return ["ガ", "ヲ", "ニ", "ガ２"]


@pytest.fixture()
def bar_rels() -> list[str]:
    return ["ノ"]


@pytest.fixture()
def special_tokens(exophora_referents: list[str]) -> list[str]:
    return [*[f"[{er}]" for er in exophora_referents], "[NULL]", "[NA]"]


@pytest.fixture()
def tokenizer(special_tokens: list[str]) -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained("ku-nlp/deberta-v2-tiny-japanese", additional_special_tokens=special_tokens)


@pytest.fixture()
def fixture_train_dataset(
    pas_cases: list[str],
    bar_rels: list[str],
    special_tokens: list[str],
    exophora_referents: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> CohesionDataset:
    dataset = CohesionDataset(
        DATA_DIR / "knp",
        tasks=ListConfig(["pas", "bridging", "coreference"]),
        cases=ListConfig(pas_cases),
        bar_rels=ListConfig(bar_rels),
        max_seq_length=128,
        document_split_stride=1,
        special_tokens=ListConfig(special_tokens),
        exophora_referents=ListConfig(exophora_referents),
        training=True,
        flip_reader_writer=False,
        tokenizer=tokenizer,
    )
    return dataset


@pytest.fixture()
def fixture_eval_dataset(
    pas_cases: list[str],
    bar_rels: list[str],
    special_tokens: list[str],
    exophora_referents: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> CohesionDataset:
    dataset = CohesionDataset(
        DATA_DIR / "knp",
        tasks=ListConfig(["pas", "bridging", "coreference"]),
        cases=ListConfig(pas_cases),
        bar_rels=ListConfig(bar_rels),
        max_seq_length=128,
        document_split_stride=1,
        special_tokens=ListConfig(special_tokens),
        exophora_referents=ListConfig(exophora_referents),
        training=False,
        flip_reader_writer=False,
        tokenizer=tokenizer,
    )
    return dataset


@pytest.fixture()
def fixture_example1():
    with DATA_DIR.joinpath("expected/example/1.json").open() as f:
        yield json.load(f)
