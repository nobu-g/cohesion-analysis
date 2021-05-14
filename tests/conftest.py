import json
import os
import sys
from pathlib import Path

import pytest
import torch
from kyoto_reader import ALL_CASES, ALL_COREFS
from kyoto_reader import KyotoReader

here = Path(__file__).parent
sys.path.append(str(here.parent / 'src'))

from data_loader.dataset import PASDataset

INF = 2 ** 10


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
        ]
    )
    yield input_tensor


@pytest.fixture()
def fixture_documents_pred():
    reader = KyotoReader(here / 'data' / 'pred',
                         target_cases=ALL_CASES,
                         target_corefs=ALL_COREFS)
    yield reader.process_all_documents()


@pytest.fixture()
def fixture_documents_gold():
    reader = KyotoReader(here / 'data' / 'gold',
                         target_cases=ALL_CASES,
                         target_corefs=ALL_COREFS)
    yield reader.process_all_documents()


@pytest.fixture()
def fixture_scores():
    with here.joinpath('data/expected/score/0.json').open() as f:
        yield json.load(f)


@pytest.fixture()
def fixture_train_dataset() -> PASDataset:
    dataset = PASDataset(
        here / 'data/knp',
        cases=['ガ', 'ヲ', 'ニ', 'ガ２'],
        exophors=['著者', '読者', '不特定:人', '不特定:物'],
        coreference=True,
        bridging=True,
        max_seq_length=128,
        bert_path=os.environ['BERT_PATH'],
        training=True,
        kc=False,
        train_targets=['overt', 'dep', 'zero'],
        pas_targets=['pred', 'noun'],
        gold_path=here / 'data/knp',
    )
    return dataset


@pytest.fixture()
def fixture_eval_dataset() -> PASDataset:
    dataset = PASDataset(
        here / 'data/reparsed',
        cases=['ガ', 'ヲ', 'ニ', 'ガ２'],
        exophors=['著者', '読者', '不特定:人', '不特定:物'],
        coreference=True,
        bridging=True,
        max_seq_length=128,
        bert_path=os.environ['BERT_PATH'],
        training=True,
        kc=False,
        train_targets=['overt', 'dep', 'zero'],
        pas_targets=['pred', 'noun'],
        gold_path=here / 'data/knp',
    )
    return dataset


@pytest.fixture()
def fixture_example1():
    with here.joinpath('data/expected/example/1.json').open() as f:
        yield json.load(f)
