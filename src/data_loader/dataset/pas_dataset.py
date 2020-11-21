import os
import logging
import hashlib
from pathlib import Path
import _pickle as cPickle
from collections import defaultdict
from typing import List, Dict, Optional, NamedTuple, Union

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import BertTokenizer
from kyoto_reader import KyotoReader, Document, ALL_EXOPHORS

from .read_example import PasExample
from utils.constants import TASK_ID


class InputFeatures(NamedTuple):
    input_ids: List[int]
    input_mask: List[bool]
    segment_ids: List[int]
    arguments_set: List[List[List[int]]]
    overt_mask: List[List[List[int]]]
    ng_token_mask: List[List[List[bool]]]
    deps: List[List[int]]


class PASDataset(Dataset):
    def __init__(self,
                 path: Union[str, Path],
                 cases: List[str],
                 exophors: List[str],
                 coreference: bool,
                 bridging: bool,
                 dataset_config: dict,
                 training: bool,
                 kc: bool,
                 train_targets: List[str],
                 pas_targets: List[str],
                 logger=None,
                 kc_joined_path: Optional[str] = None,
                 ) -> None:
        if isinstance(path, str):
            path = Path(path)
        self.reader = KyotoReader(path,
                                  target_cases=dataset_config['target_cases'],
                                  target_corefs=dataset_config['target_corefs'],
                                  extract_nes=False)
        self.target_cases: List[str] = [c for c in cases if c in self.reader.target_cases and c != 'ノ']
        self.target_exophors: List[str] = [e for e in exophors if e in ALL_EXOPHORS]
        self.coreference: bool = coreference
        self.bridging: bool = bridging
        self.kc: bool = kc
        self.train_overt: bool = 'overt' in train_targets
        self.train_case: bool = 'case' in train_targets
        self.train_zero: bool = 'zero' in train_targets
        self.pas_targets: List[str] = pas_targets
        self.logger = logger if logger else logging.getLogger(__file__)
        special_tokens = self.target_exophors + ['NULL'] + (['NA'] if coreference else [])
        self.special_to_index: Dict[str, int] = {token: dataset_config['max_seq_length'] - i - 1 for i, token
                                                 in enumerate(reversed(special_tokens))}
        self.tokenizer = BertTokenizer.from_pretrained(dataset_config['bert_path'], do_lower_case=False,
                                                       tokenize_chinese_chars=False)
        self.expanded_vocab_size: int = self.tokenizer.vocab_size + len(special_tokens)
        self.max_seq_length = dataset_config['max_seq_length']
        documents = list(self.reader.process_all_documents())
        self.documents: Optional[List[Document]] = documents if not training else None

        if self.kc and not training:
            assert kc_joined_path is not None
            reader = KyotoReader(Path(kc_joined_path),
                                 target_cases=dataset_config['target_cases'],
                                 target_corefs=dataset_config['target_corefs'],
                                 extract_nes=False)
            self.joined_documents = list(reader.process_all_documents())

        self.examples: List[PasExample] = []
        load_cache: bool = ('BPA_DISABLE_CACHE' not in os.environ and 'BPA_OVERWRITE_CACHE' not in os.environ)
        save_cache: bool = ('BPA_DISABLE_CACHE' not in os.environ)
        bpa_cache_dir: Path = Path(os.environ.get('BPA_CACHE_DIR', f'/data/{os.environ["USER"]}/bpa_cache'))
        for document in tqdm(documents, desc='processing documents'):
            hash_ = self._hash(document, self.target_cases, self.target_exophors, coreference, bridging, kc,
                               pas_targets, train_targets, dataset_config, dataset_config['bert_path'])
            example_cache_path = bpa_cache_dir / hash_ / f'{document.doc_id}.pkl'
            if example_cache_path.exists() and load_cache:
                with example_cache_path.open('rb') as f:
                    example = cPickle.load(f)
            else:
                example = PasExample()
                example.load(document,
                             cases=self.target_cases,
                             exophors=self.target_exophors,
                             coreference=coreference,
                             bridging=bridging,
                             kc=kc,
                             pas_targets=pas_targets,
                             tokenizer=self.tokenizer)
                if save_cache:
                    example_cache_path.parent.mkdir(exist_ok=True, parents=True)
                    with example_cache_path.open('wb') as f:
                        cPickle.dump(example, f)

            # ignore too long document
            if len(example.tokens) > self.max_seq_length - len(self.special_to_index):
                continue

            self.examples.append(example)

    @staticmethod
    def _hash(document, *args) -> str:
        attrs = ('cases', 'corefs', 'relax_cases', 'extract_nes', 'use_pas_tag')
        assert set(attrs) <= set(vars(document).keys())
        vars_document = {k: v for k, v in vars(document).items() if k in attrs}
        string = repr(sorted(vars_document)) + ''.join(repr(a) for a in args)
        return hashlib.md5(string.encode()).hexdigest()

    def _convert_example_to_feature(self,
                                    example: PasExample,
                                    ) -> InputFeatures:
        """Loads a data file into a list of `InputBatch`s."""

        vocab_size = self.tokenizer.vocab_size
        max_seq_length = self.max_seq_length
        num_special_tokens = len(self.special_to_index)
        num_relations = len(self.target_cases) + int(self.bridging) + int(self.coreference)

        tokens = example.tokens
        tok_to_orig_index = example.tok_to_orig_index
        orig_to_tok_index = example.orig_to_tok_index

        segment_ids: List[int] = []
        arguments_set: List[List[List[int]]] = []
        candidates_set: List[List[List[int]]] = []
        overts_set: List[List[List[int]]] = []
        deps: List[List[int]] = []

        # subword loop
        for token, orig_index in zip(tokens, tok_to_orig_index):
            segment_ids.append(0)

            # subsequent subword or [CLS] token or [SEP] token
            if token.startswith("##") or orig_index is None:
                arguments_set.append([[] for _ in range(num_relations)])
                overts_set.append([[] for _ in range(num_relations)])
                candidates_set.append([[] for _ in range(num_relations)])
                deps.append([0] * max_seq_length)
                continue

            arguments: List[List[int]] = [[] for _ in range(num_relations)]
            overts: List[List[int]] = [[] for _ in range(num_relations)]
            for i, (case, arg_strings) in enumerate(example.arguments_set[orig_index].items()):
                if not arg_strings:
                    continue
                for arg_string in arg_strings:
                    if case == '=':
                        # coreference (arg_string: 著者, 23, NA, ...)
                        if arg_string in self.special_to_index:
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            arguments[i].append(orig_to_tok_index[int(arg_string)])
                    else:
                        # pas (arg_string: 著者, 8%C, 15%O, NULL, ...)
                        if arg_string in self.special_to_index:
                            if self.train_zero is False:
                                continue
                            arguments[i].append(self.special_to_index[arg_string])
                        else:
                            arg_index, flag = int(arg_string[:-2]), arg_string[-1]
                            if flag == 'C':
                                overts[i].append(orig_to_tok_index[arg_index])
                            if (flag == 'C' and self.train_overt is False) or \
                               (flag == 'N' and self.train_case is False) or \
                               (flag == 'O' and self.train_zero is False):
                                continue
                            arguments[i].append(orig_to_tok_index[arg_index])

            arguments_set.append(arguments)
            overts_set.append(overts)

            ddep = example.ddeps[orig_index]
            deps.append([(0 if idx is None or ddep != example.dtids[idx] else 1) for idx in tok_to_orig_index])
            deps[-1] += [0] * (max_seq_length - len(tok_to_orig_index))

            # arguments_set が空のもの (助詞など) には candidates を設定しない
            candidates: List[List[int]] = []
            for case, arg_strings in example.arguments_set[orig_index].items():
                if arg_strings:
                    if case != '=':
                        cands = [orig_to_tok_index[dmid] for dmid in example.arg_candidates_set[orig_index]]
                        specials = self.target_exophors + ['NULL']
                    else:
                        cands = [orig_to_tok_index[dmid] for dmid in example.ment_candidates_set[orig_index]]
                        specials = self.target_exophors + ['NA']
                    cands += [self.special_to_index[special] for special in specials]
                else:
                    cands = []
                candidates.append(cands)
            candidates_set.append(candidates)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [True] * len(input_ids)

        # Zero-pad up to the sequence length (except for special tokens).
        while len(input_ids) < max_seq_length - num_special_tokens:
            input_ids.append(0)
            input_mask.append(False)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_relations)])
            overts_set.append([[] for _ in range(num_relations)])
            candidates_set.append([[] for _ in range(num_relations)])
            deps.append([0] * max_seq_length)

        # add special tokens
        for i in range(num_special_tokens):
            input_ids.append(vocab_size + i)
            input_mask.append(True)
            segment_ids.append(0)
            arguments_set.append([[] for _ in range(num_relations)])
            overts_set.append([[] for _ in range(num_relations)])
            candidates_set.append([[] for _ in range(num_relations)])
            deps.append([0] * max_seq_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(arguments_set) == max_seq_length
        assert len(overts_set) == max_seq_length
        assert len(candidates_set) == max_seq_length
        assert len(deps) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            arguments_set=[[[int(x in args) for x in range(max_seq_length)] for args in arguments]
                           for arguments in arguments_set],
            overt_mask=[[[(x in overt) for x in range(max_seq_length)] for overt in overts]
                           for overts in overts_set],
            ng_token_mask=[[[(x in cands) for x in range(max_seq_length)] for cands in candidates]
                           for candidates in candidates_set],  # False -> mask, True -> keep
            deps=deps,
        )

        return feature

    def stat(self) -> dict:
        n_mentions = n_preds_pa = n_preds_bar = 0
        cr = defaultdict(int)
        n_args_bar = defaultdict(int)
        n_args_pa = defaultdict(lambda: defaultdict(int))

        for arguments in (x for example in self.examples for x in example.arguments_set):
            for case, args in arguments.items():
                if not args:
                    continue
                arg: str = args[0]
                if case == '=':
                    if arg == 'NA':
                        cr['na'] += 1
                        continue
                    n_mentions += 1
                    if arg in self.target_exophors:
                        cr['exo'] += 1
                    else:
                        cr['ana'] += 1
                else:
                    n_args = n_args_bar if case == 'ノ' else n_args_pa[case]
                    if arg == 'NULL':
                        n_args['null'] += 1
                        continue
                    n_args['all'] += 1
                    if arg in self.target_exophors:
                        n_args['exo'] += 1
                    elif '%C' in arg:
                        n_args['overt'] += 1
                    elif '%N' in arg:
                        n_args['dep'] += 1
                    elif '%O' in arg:
                        n_args['zero'] += 1

            arguments_: List[List[str]] = list(arguments.values())
            if self.coreference:
                arguments_ = arguments_[:-1]
            if self.bridging:
                if arguments_[-1]:
                    n_preds_bar += 1
                arguments_ = arguments_[:-1]
            if any(arguments_):
                n_preds_pa += 1

        n_args_pa_all = defaultdict(int)
        for case, ans in n_args_pa.items():
            for anal, num in ans.items():
                n_args_pa_all[anal] += num
        n_args_pa['all'] = n_args_pa_all

        cr['mentions'] = n_mentions

        return {'examples': len(self.examples),
                'pas': {'preds': n_preds_pa, 'args': n_args_pa},
                'bridging': {'preds': n_preds_bar, 'args': n_args_bar},
                'coreference': cr,
                'tokens': sum(len(example.tokens) for example in self.examples),
                }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx) -> tuple:
        feature = self._convert_example_to_feature(self.examples[idx])
        input_ids = np.array(feature.input_ids)          # (seq)
        attention_mask = np.array(feature.input_mask)    # (seq)
        segment_ids = np.array(feature.segment_ids)      # (seq)
        arguments_ids = np.array(feature.arguments_set)  # (seq, case, seq)
        overt_mask = np.array(feature.overt_mask)        # (seq, case, seq)
        ng_token_mask = np.array(feature.ng_token_mask)  # (seq, case, seq)
        deps = np.array(feature.deps)                    # (seq, seq)
        task = np.array(TASK_ID['pa'])                   # ()
        return input_ids, attention_mask, segment_ids, ng_token_mask, arguments_ids, deps, task, overt_mask
