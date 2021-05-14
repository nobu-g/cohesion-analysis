from typing import List

from kyoto_reader import Document

from scorer import Scorer, Measure


def test_scorer(fixture_documents_pred: List[Document], fixture_documents_gold: List[Document], fixture_scores: dict):
    cases = ['ガ', 'ヲ']
    scorer = Scorer(fixture_documents_pred, fixture_documents_gold,
                    target_cases=cases,
                    target_exophors=['著者', '読者', '不特定:人', '不特定:物'],
                    coreference=True,
                    bridging=True,
                    pas_target='all')

    result = scorer.run().to_dict()
    for case in cases:
        case_result = result[case]
        for anal in Scorer.DEPTYPE2ANALYSIS.values():
            expected: dict = fixture_scores[case][anal]
            actual: Measure = case_result[anal]
            assert expected['denom_pred'] == actual.denom_pred
            assert expected['denom_gold'] == actual.denom_gold
            assert expected['correct'] == actual.correct
