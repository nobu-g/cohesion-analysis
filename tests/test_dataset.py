
from data_loader.dataset import PASDataset


def test_train_dataset(fixture_train_dataset: PASDataset, fixture_example1: dict):
    example = fixture_train_dataset.examples[1]
    mrphs = fixture_example1['mrphs']
    assert example.tokens == ['[CLS]'] + [t for mrph in mrphs for t in mrph['tokens']] + ['[SEP]']
    assert len(example.words) == len(fixture_example1['mrphs'])
    for i in range(len(example.words)):
        word = example.words[i]
        tok_index = example.orig_to_tok_index[i]
        arguments = example.arguments_set[i]
        arg_candidates = example.arg_candidates_set[i]
        ment_candidates = example.ment_candidates_set[i]
        # dtid = example.dtids[i]
        # ddep = example.ddeps[i]

        mrph = fixture_example1['mrphs'][i]
        assert mrph['surf'] == word
        assert mrph['tokens'][0] == example.tokens[tok_index]  # head token is the representative token of a mrph
        for rel in fixture_train_dataset.relations:
            arg_strings = [arg[:-2] if arg[-2:] in ('%C', '%N', '%O') else arg for arg in arguments[rel]]
            assert set(arg_strings) == set(mrph['arguments'][rel])

        assert set(arg_candidates) == set(mrph['arg_candidates'])
        assert set(ment_candidates) == set(mrph['ment_candidates'])


def test_eval_dataset(fixture_eval_dataset: PASDataset, fixture_example1: dict):
    example = fixture_eval_dataset.examples[1]
    mrphs = fixture_example1['mrphs']
    assert example.tokens == ['[CLS]'] + [t for mrph in mrphs for t in mrph['tokens']] + ['[SEP]']
    assert len(example.words) == len(fixture_example1['mrphs'])
    for i in range(len(example.words)):
        word = example.words[i]
        tok_index = example.orig_to_tok_index[i]
        arguments = example.arguments_set[i]
        arg_candidates = example.arg_candidates_set[i]
        ment_candidates = example.ment_candidates_set[i]

        mrph = fixture_example1['mrphs'][i]
        assert mrph['surf'] == word
        assert mrph['tokens'][0] == example.tokens[tok_index]  # head token is the representative token of a mrph
        for rel in fixture_eval_dataset.relations:
            if mrph['arguments'][rel]:
                if rel != '=':
                    assert arguments[rel] == ['NULL']
                else:
                    assert arguments[rel] == ['NA']
            else:
                assert arguments[rel] == []

        assert set(arg_candidates) == set(mrph['arg_candidates'])
        assert set(ment_candidates) == set(mrph['ment_candidates'])
