import logging
from typing import List, Dict, Tuple, Optional
from collections import OrderedDict

from transformers import BertTokenizer
from kyoto_reader import Document, BasePhrase, BaseArgument, Argument, SpecialArgument, UNCERTAIN

from utils.util import is_pas_target, is_bridging_target, is_coreference_target

logger = logging.getLogger(__file__)


class PasExample:
    """A single training/test example for pas analysis."""

    def __init__(self) -> None:
        self.words: List[str] = []
        self.tokens: List[str] = []
        self.orig_to_tok_index: List[int] = []
        self.tok_to_orig_index: List[Optional[int]] = []
        self.arguments_set: List[Dict[str, List[str]]] = []
        self.arg_candidates_set: List[List[int]] = []
        self.ment_candidates_set: List[List[int]] = []
        self.dtids: List[int] = []  # dmid -> dtid
        self.ddeps: List[int] = []  # dmid -> dmid which has dep
        self.doc_id: str = ''

    def load(self,
             document: Document,
             cases: List[str],
             exophors: List[str],
             coreference: bool,
             bridging: bool,
             relations: List[str],
             kc: bool,
             pas_targets: List[str],
             tokenizer: BertTokenizer,
             ) -> None:
        self.doc_id = document.doc_id
        process_all = (kc is False) or (document.doc_id.split('-')[-1] == '00')
        last_sent = document.sentences[-1] if len(document) > 0 else None
        relax_exophors = {}
        for exophor in exophors:
            relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in '１２３４５６７８９':
                    relax_exophors[exophor + n] = exophor
        dmid2arguments: Dict[int, Dict[str, List[BaseArgument]]] = {pred.dmid: document.get_arguments(pred)
                                                                    for pred in document.get_predicates()}
        head_dmids = []
        for sentence in document:
            process: bool = process_all or (sentence is last_sent)
            head_dmids += [bp.dmid for bp in sentence.bps]
            for bp in sentence.bps:
                for mrph in bp.mrph_list():
                    self.words.append(mrph.midasi)
                    self.dtids.append(bp.dtid)
                    self.ddeps.append(bp.parent.dtid if bp.parent is not None else -1)
                    arguments = OrderedDict((rel, []) for rel in relations)
                    arg_candidates = ment_candidates = []
                    if document.mrph2dmid[mrph] == bp.dmid and process is True:
                        if is_pas_target(bp, verbal=('pred' in pas_targets), nominal=('noun' in pas_targets)):
                            arg_candidates = [x for x in head_dmids if x != bp.dmid]
                            for case in cases:
                                dmid2args = {dmid: arguments[case] for dmid, arguments in dmid2arguments.items()}
                                arguments[case] = self._get_args(bp.dmid, dmid2args, relax_exophors, arg_candidates)

                        if bridging and is_bridging_target(bp):
                            arg_candidates = [x for x in head_dmids if x != bp.dmid]
                            dmid2args = {dmid: arguments['ノ'] for dmid, arguments in dmid2arguments.items()}
                            arguments['ノ'] = self._get_args(bp.dmid, dmid2args, relax_exophors, arg_candidates)

                        if coreference and is_coreference_target(bp):
                            ment_candidates = [x for x in head_dmids if x < bp.dmid]  # do not solve cataphora
                            arguments['='] = self._get_mentions(bp, document, relax_exophors, ment_candidates)

                    self.arguments_set.append(arguments)
                    self.arg_candidates_set.append(arg_candidates)
                    self.ment_candidates_set.append(ment_candidates)

        self.tokens, self.tok_to_orig_index, self.orig_to_tok_index = self._get_tokenized_tokens(self.words, tokenizer)

    def _get_args(self,
                  dmid: int,
                  dmid2args: Dict[int, List[BaseArgument]],
                  relax_exophors: Dict[str, str],
                  candidates: List[int],
                  ) -> List[str]:
        """述語の dmid と その項 dmid2args から、項の文字列を得る
        返り値が空リストの場合、この項について loss は計算されない
        overt: {dmid}%C
        case: {dmid}%N
        zero: {dmid}%O
        exophor: {exophor}
        no arg: NULL
        """
        # filter out non-target exophors
        args = []
        for arg in dmid2args.get(dmid, []):
            if isinstance(arg, SpecialArgument):
                if arg.exophor in relax_exophors:
                    arg.exophor = relax_exophors[arg.exophor]
                    args.append(arg)
                elif arg.exophor == UNCERTAIN:
                    return []  # don't train uncertain argument
            else:
                args.append(arg)
        if not args:
            return ['NULL']
        arg_strings: List[str] = []
        for arg in args:
            if isinstance(arg, Argument):
                if arg.dmid not in candidates:
                    logger.debug(f'argument: {arg} in {self.doc_id} is not in candidates and ignored')
                    continue
                string = str(arg.dmid)
                if arg.dep_type == 'overt':
                    string += '%C'
                elif arg.dep_type == 'dep':
                    string += '%N'
                else:
                    assert arg.dep_type in ('intra', 'inter')
                    string += '%O'
            # exophor
            else:
                string = str(arg)
            arg_strings.append(string)
        return arg_strings

    def _get_mentions(self,
                      bp: BasePhrase,
                      document: Document,
                      relax_exophors: Dict[str, str],
                      candidates: List[int],
                      ) -> List[str]:
        if bp.dtid in document.mentions:
            ment_strings: List[str] = []
            src_mention = document.mentions[bp.dtid]
            tgt_mentions = document.get_siblings(src_mention, relax=False)
            exophors = [document.entities[eid].exophor for eid in src_mention.eids
                        if document.entities[eid].is_special]
            for mention in tgt_mentions:
                if mention.dmid not in candidates:
                    logger.debug(f'mention: {mention} in {self.doc_id} is not in candidates and ignored')
                    continue
                ment_strings.append(str(mention.dmid))
            for exophor in exophors:
                if exophor in relax_exophors:
                    ment_strings.append(relax_exophors[exophor])  # 不特定:人１ -> 不特定:人
            if ment_strings:
                return ment_strings
            else:
                return ['NA']  # force cataphor to point [NA]
        else:
            return ['NA']

    @staticmethod
    def _get_tokenized_tokens(words: List[str],
                              tokenizer: BertTokenizer,
                              ) -> Tuple[List[str], List[Optional[int]], List[int]]:
        all_tokens = []
        tok_to_orig_index: List[Optional[int]] = []
        orig_to_tok_index: List[int] = []

        all_tokens.append('[CLS]')
        tok_to_orig_index.append(None)  # There's no original token corresponding to [CLS] token

        for i, word in enumerate(words):
            orig_to_tok_index.append(len(all_tokens))  # assign head subword
            sub_tokens = tokenizer.tokenize(word)
            for sub_token in sub_tokens:
                all_tokens.append(sub_token)
                tok_to_orig_index.append(i)

        all_tokens.append('[SEP]')
        tok_to_orig_index.append(None)  # There's no original token corresponding to [SEP] token

        return all_tokens, tok_to_orig_index, orig_to_tok_index

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        string = ''
        for i, (word, args) in enumerate(zip(self.words, self.arguments_set)):
            pad = ' ' * (5 - len(word)) * 2
            string += f'{i:02} {word}{pad}({" ".join(f"{case}:{arg}" for case, arg in args.items())})\n'
        return string
