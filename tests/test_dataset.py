from typing import Any

from cohesion_tools.task import Task

from datamodule.dataset import CohesionDataset


def test_train_dataset(fixture_train_dataset: CohesionDataset, fixture_example1: dict[str, Any]) -> None:
    example = fixture_train_dataset.examples[1]
    mrphs_exp = fixture_example1["mrphs"]
    encoding = example.encoding
    assert encoding is not None
    pas_phrases = example.phrases[Task.PAS_ANALYSIS]
    bridging_phrases = example.phrases[Task.BRIDGING_REFERENCE_RESOLUTION]
    num_morphemes = sum(len(phrase.morpheme_global_indices) for phrase in pas_phrases)

    assert [t for t in encoding.tokens if t != "[PAD]"] == ["[CLS]"] + [
        t for mrph in mrphs_exp for t in mrph["tokens"]
    ] + ["[SEP]"]
    assert num_morphemes == len(fixture_example1["mrphs"])
    for pas_phrase, bar_phrase in zip(pas_phrases, bridging_phrases):
        arguments: dict[str, list[str]] = {}
        if pas_phrase.is_target:
            assert pas_phrase.rel2tags is not None
            arguments = pas_phrase.rel2tags
        for case in fixture_train_dataset.cases:
            arg_strings: list[str] = []
            for arg_string in arguments.get(case, []):
                if arg_string in fixture_train_dataset.special_to_index:
                    arg_strings.append(arg_string)
                else:
                    arg_strings.append(str(pas_phrases[int(arg_string)].head_morpheme_global_index))
            assert set(arg_strings) == set(mrphs_exp[pas_phrase.head_morpheme_global_index]["arguments"][case])
        for morpheme_global_index, morpheme_surf in zip(pas_phrase.morpheme_global_indices, pas_phrase.morphemes):
            mrph_exp = fixture_example1["mrphs"][morpheme_global_index]
            assert morpheme_surf == mrph_exp["surf"]
            assert [encoding.tokens[i] for i in range(*encoding.word_to_tokens(morpheme_global_index))] == mrph_exp[
                "tokens"
            ]
            candidates = set()
            if pas_phrase.is_target and morpheme_global_index == pas_phrase.head_morpheme_global_index:
                candidates |= set(candidate.head_morpheme_global_index for candidate in pas_phrase.referent_candidates)
            if bar_phrase.is_target and morpheme_global_index == bar_phrase.head_morpheme_global_index:
                candidates |= set(candidate.head_morpheme_global_index for candidate in bar_phrase.referent_candidates)
            assert candidates == set(mrph_exp["arg_candidates"])


def test_train_dataset_coreference(fixture_train_dataset: CohesionDataset, fixture_example1: dict[str, Any]) -> None:
    example = fixture_train_dataset.examples[1]
    mrphs_exp = fixture_example1["mrphs"]
    phrases = example.phrases[Task.COREFERENCE_RESOLUTION]

    for phrase in phrases:
        arguments: list[str] = []
        if phrase.is_target:
            assert phrase.rel2tags is not None
            arguments = phrase.rel2tags["="]
        arg_strings = [arg.rstrip("%CNO") for arg in arguments]
        arg_strings = [
            (s if s in fixture_train_dataset.special_to_index else str(phrases[int(s)].head_morpheme_global_index))
            for s in arg_strings
        ]
        assert set(arg_strings) == set(mrphs_exp[phrase.head_morpheme_global_index]["arguments"]["="])

        for morpheme_global_index in phrase.morpheme_global_indices:
            mrph_exp = mrphs_exp[morpheme_global_index]
            candidates = set()
            if phrase.is_target and morpheme_global_index == phrase.head_morpheme_global_index:
                candidates |= set(candidate.head_morpheme_global_index for candidate in phrase.referent_candidates)
            assert candidates == set(mrph_exp["ment_candidates"])


# def test_eval_dataset_pas_analysis(fixture_eval_dataset: CohesionDataset, fixture_example1: dict):
#     example = fixture_eval_dataset.examples[1]
#     mrphs_exp = fixture_example1['mrphs']
#     encoding = example.encoding
#     assert encoding is not None
#     annotation = example.annotations[Task.PAS_ANALYSIS]
#     assert isinstance(annotation, PasAnnotation)
#     phrases = example.phrases[Task.PAS_ANALYSIS]
#     mrphs = example.mrphs[Task.PAS_ANALYSIS]
#
#     assert [t for t in encoding.tokens if t != '[PAD]'] == ['[CLS]'] + [
#         t for mrph in mrphs_exp for t in mrph['tokens']
#     ] + ['[SEP]']
#     assert len(mrphs) == len(fixture_example1['mrphs'])
#     for phrase in phrases:
#         arguments: dict[str, list[str]] = annotation.arguments_set[phrase.dtid]
#         for case in fixture_eval_dataset.cases:
#             if phrase.is_target:
#                 assert arguments[case] == ['[NULL]']
#             else:
#                 assert arguments[case] == []
#
#
# def test_eval_dataset_bridging(fixture_eval_dataset: CohesionDataset, fixture_example1: dict):
#     example = fixture_eval_dataset.examples[1]
#     mrphs_exp = fixture_example1['mrphs']
#     encoding = example.encoding
#     assert encoding is not None
#     annotation = example.annotations[Task.BRIDGING]
#     assert isinstance(annotation, BridgingAnnotation)
#     phrases = example.phrases[Task.BRIDGING]
#     mrphs = example.mrphs[Task.BRIDGING]
#
#     assert [t for t in encoding.tokens if t != '[PAD]'] == ['[CLS]'] + [
#         t for mrph in mrphs_exp for t in mrph['tokens']
#     ] + ['[SEP]']
#     assert len(mrphs) == len(fixture_example1['mrphs'])
#     for phrase in phrases:
#         arguments = annotation.arguments_set[phrase.dtid]
#         if phrase.is_target:
#             assert arguments == ['[NULL]']
#         else:
#             assert arguments == []
#
#
# def test_eval_dataset_coreference(fixture_eval_dataset: CohesionDataset, fixture_example1: dict):
#     example = fixture_eval_dataset.examples[1]
#     mrphs_exp = fixture_example1['mrphs']
#     encoding = example.encoding
#     assert encoding is not None
#     annotation = example.annotations[Task.COREFERENCE]
#     isinstance(annotation, CoreferenceAnnotation)
#     phrases = example.phrases[Task.COREFERENCE]
#     mrphs = example.mrphs[Task.COREFERENCE]
#
#     assert [t for t in encoding.tokens if t != '[PAD]'] == ['[CLS]'] + [
#         t for mrph in mrphs_exp for t in mrph['tokens']
#     ] + ['[SEP]']
#     assert len(mrphs) == len(fixture_example1['mrphs'])
#     for phrase in phrases:
#         arguments = annotation.arguments_set[phrase.dtid]
#         if phrase.is_target:
#             assert arguments == ['[NA]']
#         else:
#             assert arguments == []
