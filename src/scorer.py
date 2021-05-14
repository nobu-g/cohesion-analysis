import argparse
import io
import logging
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Set, Union, Optional, TextIO

import pandas as pd
from jinja2 import Template, Environment, FileSystemLoader
from kyoto_reader import KyotoReader, Document, Argument, SpecialArgument, BaseArgument, Predicate, Mention, BasePhrase
from pyknp import BList

from utils.constants import CASE2YOMI
from utils.util import is_pas_target, is_bridging_target, is_coreference_target

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    """A class to evaluate system output.

    To evaluate system output with this class, you have to prepare gold data and system prediction data as instances of
    :class:`kyoto_reader.Document`

    Args:
        documents_pred (List[Document]): システム予測文書集合
        documents_gold (List[Document]): 正解文書集合
        target_cases (List[str]): 評価の対象とする格 (kyoto_reader.ALL_CASES を参照)
        target_exophors (List[str]): 評価の対象とする外界照応の照応先 (kyoto_reader.ALL_EXOPHORS を参照)
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)
        pas_target (str): 述語項構造解析において述語として扱う対象 ('pred': 用言, 'noun': 体言, 'all': 両方, '': 述語なし (default: pred))

    Attributes:
        cases (List[str]): 評価の対象となる格
        doc_ids: (List[str]): 評価の対象となる文書の文書ID集合
        did2document_pred (Dict[str, Document]): 文書IDからシステム予測文書を引くための辞書
        did2document_gold (Dict[str, Document]): 文書IDから正解文書を引くための辞書
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        pas_target (str): 述語項構造解析において述語として扱う対象
        comp_result (Dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
        sub_scorers (List[SubScorer]): 文書ごとの評価を行うオブジェクトのリスト
        relax_exophors (Dict[str, str]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
    """
    DEPTYPE2ANALYSIS = OrderedDict([('overt', 'overt'),
                                    ('dep', 'dep'),
                                    ('intra', 'zero_intra'),
                                    ('inter', 'zero_inter'),
                                    ('exo', 'zero_exophora')])

    def __init__(self,
                 documents_pred: List[Document],
                 documents_gold: List[Document],
                 target_cases: List[str],
                 target_exophors: List[str],
                 bridging: bool = False,
                 coreference: bool = False,
                 pas_target: str = 'pred'):
        # long document may have been ignored
        assert set(doc.doc_id for doc in documents_pred) <= set(doc.doc_id for doc in documents_gold)
        self.cases: List[str] = target_cases if pas_target != '' else []
        self.doc_ids: List[str] = [doc.doc_id for doc in documents_pred]
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in documents_gold}
        self.bridging: bool = bridging
        self.coreference: bool = coreference
        self.pas_target: str = pas_target

        self.comp_result: Dict[tuple, str] = {}
        self.sub_scorers: List[SubScorer] = []
        self.relax_exophors: Dict[str, str] = {}
        for exophor in target_exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in ('１', '２', '３', '４', '５', '６', '７', '８', '９', '１０', '１１'):
                    self.relax_exophors[exophor + n] = exophor

    def run(self) -> 'ScoreResult':
        """読み込んだ正解文書集合とシステム予測文書集合に対して評価を行う

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result = {}
        self.sub_scorers = []
        all_result = None
        for doc_id in self.doc_ids:
            sub_scorer = SubScorer(self.did2document_pred[doc_id], self.did2document_gold[doc_id],
                                   cases=self.cases,
                                   bridging=self.bridging,
                                   coreference=self.coreference,
                                   relax_exophors=self.relax_exophors,
                                   pas_target=self.pas_target)
            if all_result is None:
                all_result = sub_scorer.run()
            else:
                all_result += sub_scorer.run()
            self.sub_scorers.append(sub_scorer)
            self.comp_result.update({(doc_id, *key): val for key, val in sub_scorer.comp_result.items()})
        return all_result

    def write_html(self, output_file: Union[str, Path]) -> None:
        """正解データとシステム予測の比較をHTML形式で書き出し

        Args:
            output_file (Union[str, Path]): 出力先ファイル
        """
        data: List[tuple] = []
        for sub_scorer in self.sub_scorers:
            gold_tree = ''
            for sid in sub_scorer.document_gold.sid2sentence.keys():
                with io.StringIO() as string:
                    self._draw_tree(sid,
                                    sub_scorer.predicates_gold,
                                    sub_scorer.mentions_gold,
                                    sub_scorer.bridgings_gold,
                                    sub_scorer.document_gold,
                                    fh=string)
                    gold_tree += string.getvalue()

            pred_tree = ''
            for sid in sub_scorer.document_pred.sid2sentence.keys():
                with io.StringIO() as string:
                    self._draw_tree(sid,
                                    sub_scorer.predicates_pred,
                                    sub_scorer.mentions_pred,
                                    sub_scorer.bridgings_pred,
                                    sub_scorer.document_pred,
                                    fh=string)
                    pred_tree += string.getvalue()
            data.append((sub_scorer.document_gold.sentences, gold_tree, pred_tree))

        env = Environment(loader=FileSystemLoader(str(Path(__file__).parent)))
        template: Template = env.get_template('template.html')

        with Path(output_file).open('wt') as f:
            f.write(template.render({'data': data}))

    def _draw_tree(self,
                   sid: str,
                   predicates: List[BasePhrase],
                   mentions: List[BasePhrase],
                   anaphors: List[BasePhrase],
                   document: Document,
                   fh: Optional[TextIO] = None,
                   html: bool = True
                   ) -> None:
        """Write the predicate-argument structures, coreference relations, and bridging anaphora relations of the
        specified sentence in tree format.

        Args:
            sid (str): 出力対象の文ID
            predicates (List[BasePhrase]): documentに含まれる全ての述語
            mentions (List[BasePhrase]): documentに含まれる全てのメンション
            anaphors (List[BasePhrase]): documentに含まれる全ての橋渡し照応詞
            document (Document): 出力対象の文が含まれる文書
            fh (Optional[TextIO]): 出力ストリーム
            html (bool): HTML形式で出力するかどうか
        """
        result2color = {anal: 'blue' for anal in Scorer.DEPTYPE2ANALYSIS.values()}
        result2color.update({'overt': 'green', 'wrong': 'red', None: 'gray'})
        result2color_coref = {'correct': 'blue', 'wrong': 'red', None: 'gray'}
        blist: BList = document.sid2sentence[sid].blist
        with io.StringIO() as string:
            blist.draw_tag_tree(fh=string, show_pos=False)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        assert len(tree_strings) == len(blist.tag_list())
        all_targets = [m.core for m in document.mentions.values()]
        tid2predicate: Dict[int, BasePhrase] = {predicate.tid: predicate for predicate in predicates
                                                if predicate.sid == sid}
        tid2mention: Dict[int, BasePhrase] = {mention.tid: mention for mention in mentions if mention.sid == sid}
        tid2bridging: Dict[int, BasePhrase] = {anaphor.tid: anaphor for anaphor in anaphors if anaphor.sid == sid}
        for tid in range(len(tree_strings)):
            tree_strings[tid] += '  '
            if tid in tid2predicate:
                predicate = tid2predicate[tid]
                arguments = document.get_arguments(predicate)
                for case in self.cases:
                    args = arguments[case]
                    if case == 'ガ':
                        args += arguments['判ガ']
                    targets = set()
                    for arg in args:
                        target = str(arg)
                        if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                            target += str(arg.dtid)
                        targets.add(target)
                    result = self.comp_result.get((document.doc_id, predicate.dtid, case), None)
                    if html:
                        tree_strings[tid] += f'<font color="{result2color[result]}">{case}:{",".join(targets)}</font> '
                    else:
                        tree_strings[tid] += f'{case}:{",".join(targets)} '

            if self.bridging and tid in tid2bridging:
                anaphor = tid2bridging[tid]
                arguments = document.get_arguments(anaphor)
                args = arguments['ノ'] + arguments['ノ？']
                targets = set()
                for arg in args:
                    target = str(arg)
                    if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                        target += str(arg.dtid)
                    targets.add(target)
                result = self.comp_result.get((document.doc_id, anaphor.dtid, 'ノ'), None)
                if html:
                    tree_strings[tid] += f'<font color="{result2color[result]}">ノ:{",".join(targets)}</font> '
                else:
                    tree_strings[tid] += f'ノ:{",".join(targets)} '

            if self.coreference and tid in tid2mention:
                targets = set()
                src_dtid = tid2mention[tid].dtid
                if src_dtid in document.mentions:
                    src_mention = document.mentions[src_dtid]
                    tgt_mentions_relaxed = SubScorer.filter_mentions(
                        document.get_siblings(src_mention, relax=True), src_mention)
                    for tgt_mention in tgt_mentions_relaxed:
                        target: str = tgt_mention.core
                        if all_targets.count(target) > 1:
                            target += str(tgt_mention.dtid)
                        targets.add(target)
                    for eid in src_mention.eids:
                        entity = document.entities[eid]
                        if entity.exophor in self.relax_exophors:
                            targets.add(entity.exophor)
                result = self.comp_result.get((document.doc_id, src_dtid, '='), None)
                if html:
                    tree_strings[tid] += f'<font color="{result2color_coref[result]}">＝:{",".join(targets)}</font>'
                else:
                    tree_strings[tid] += '＝:' + ','.join(targets)

        print('\n'.join(tree_strings), file=fh)


class SubScorer:
    """Scorer for single document pair.

    Args:
        document_pred (Document): システム予測文書
        document_gold (Document): 正解文書
        cases (List[str]): 評価の対象とする格
        bridging (bool): 橋渡し照応の評価を行うかどうか (default: False)
        coreference (bool): 共参照の評価を行うかどうか (default: False)
        relax_exophors (Dict[str, str]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
        pas_target (str): 述語項構造解析において述語として扱う対象

    Attributes:
        doc_id (str): 対象の文書ID
        document_pred (Document): システム予測文書
        document_gold (Document): 正解文書
        cases (List[str]): 評価の対象となる格
        pas (bool): 述語項構造の評価を行うかどうか
        bridging (bool): 橋渡し照応の評価を行うかどうか
        coreference (bool): 共参照の評価を行うかどうか
        comp_result (Dict[tuple, str]): 正解と予測を比較した結果を格納するための辞書
        relax_exophors (Dict[str, str]): 「不特定:人１」などを「不特定:人」として評価するためのマップ
        predicates_pred: (List[BasePhrase]): システム予測文書に含まれる述語
        bridgings_pred: (List[BasePhrase]): システム予測文書に含まれる橋渡し照応詞
        mentions_pred: (List[BasePhrase]): システム予測文書に含まれるメンション
        predicates_gold: (List[BasePhrase]): 正解文書に含まれる述語
        bridgings_gold: (List[BasePhrase]): 正解文書に含まれる橋渡し照応詞
        mentions_gold: (List[BasePhrase]): 正解文書に含まれるメンション
    """

    def __init__(self,
                 document_pred: Document,
                 document_gold: Document,
                 cases: List[str],
                 bridging: bool,
                 coreference: bool,
                 relax_exophors: Dict[str, str],
                 pas_target: str):
        assert document_pred.doc_id == document_gold.doc_id
        self.doc_id: str = document_gold.doc_id
        self.document_pred: Document = document_pred
        self.document_gold: Document = document_gold
        self.cases: List[str] = cases
        self.pas: bool = pas_target != ''
        self.bridging: bool = bridging
        self.coreference: bool = coreference
        self.comp_result: Dict[tuple, str] = {}
        self.relax_exophors: Dict[str, str] = relax_exophors

        self.predicates_pred: List[BasePhrase] = []
        self.bridgings_pred: List[BasePhrase] = []
        self.mentions_pred: List[BasePhrase] = []
        for bp in document_pred.bp_list():
            if is_pas_target(bp, verbal=(pas_target in ('pred', 'all')), nominal=(pas_target in ('noun', 'all'))):
                self.predicates_pred.append(bp)
            if self.bridging and is_bridging_target(bp):
                self.bridgings_pred.append(bp)
            if self.coreference and is_coreference_target(bp):
                self.mentions_pred.append(bp)
        self.predicates_gold: List[BasePhrase] = []
        self.bridgings_gold: List[BasePhrase] = []
        self.mentions_gold: List[BasePhrase] = []
        for bp in document_gold.bp_list():
            if is_pas_target(bp, verbal=(pas_target in ('pred', 'all')), nominal=(pas_target in ('noun', 'all'))):
                self.predicates_gold.append(bp)
            if self.bridging and is_bridging_target(bp):
                self.bridgings_gold.append(bp)
            if self.coreference and is_coreference_target(bp):
                self.mentions_gold.append(bp)

    def run(self) -> 'ScoreResult':
        """Perform evaluation for the given gold document and system prediction document.

        Returns:
            ScoreResult: 評価結果のスコア
        """
        self.comp_result = {}
        measures_pas = self._evaluate_pas() if self.pas else None
        measures_bridging = self._evaluate_bridging() if self.bridging else None
        measure_coref = self._evaluate_coref() if self.coreference else None
        return ScoreResult(measures_pas, measures_bridging, measure_coref)

    def _evaluate_pas(self) -> pd.DataFrame:
        """calculate predicate-argument structure analysis scores"""
        # measures: Dict[str, Dict[str, Measure]] = OrderedDict(
        #     (case, OrderedDict((anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values()))
        #     for case in self.cases)
        measures = pd.DataFrame([[Measure() for _ in Scorer.DEPTYPE2ANALYSIS.values()] for _ in self.cases],
                                index=self.cases, columns=Scorer.DEPTYPE2ANALYSIS.values())
        dtid2predicate_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.predicates_pred}
        dtid2predicate_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.predicates_gold}

        for dtid in range(len(self.document_pred.bp_list())):
            if dtid in dtid2predicate_pred:
                predicate_pred = dtid2predicate_pred[dtid]
                arguments_pred = self.document_pred.get_arguments(predicate_pred, relax=False)
            else:
                arguments_pred = None

            if dtid in dtid2predicate_gold:
                predicate_gold = dtid2predicate_gold[dtid]
                arguments_gold = self.document_gold.get_arguments(predicate_gold, relax=False)
                arguments_gold_relaxed = self.document_gold.get_arguments(predicate_gold, relax=True)
            else:
                predicate_gold = arguments_gold = arguments_gold_relaxed = None

            for case in self.cases:
                args_pred: List[BaseArgument] = arguments_pred[case] if arguments_pred is not None else []
                assert len(args_pred) in (0, 1)  # Our analyzer predicts one argument for one predicate
                if predicate_gold is not None:
                    args_gold = self._filter_args(arguments_gold[case], predicate_gold)
                    args_gold_relaxed = self._filter_args(
                        arguments_gold_relaxed[case] + (arguments_gold_relaxed['判ガ'] if case == 'ガ' else []),
                        predicate_gold)
                else:
                    args_gold = args_gold_relaxed = []

                key = (dtid, case)

                # calculate precision
                if args_pred:
                    arg = args_pred[0]
                    if arg in args_gold_relaxed:
                        # use dep_type of gold argument if possible
                        arg_gold = args_gold_relaxed[args_gold_relaxed.index(arg)]
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg_gold.dep_type]
                        self.comp_result[key] = analysis
                        measures.at[case, analysis].correct += 1
                    else:
                        # system出力のdep_typeはgoldのものと違うので不整合が起きるかもしれない
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg.dep_type]
                        self.comp_result[key] = 'wrong'  # precision が下がる
                    measures.at[case, analysis].denom_pred += 1

                # calculate recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用
                # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用
                if args_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                    arg_gold = None
                    for arg in args_gold_relaxed:
                        if arg in args_pred:
                            arg_gold = arg  # 予測されている項を優先して正解の項に採用
                            break
                    if arg_gold is not None:
                        analysis = Scorer.DEPTYPE2ANALYSIS[arg_gold.dep_type]
                        assert self.comp_result[key] == analysis
                    else:
                        analysis = Scorer.DEPTYPE2ANALYSIS[args_gold[0].dep_type]
                        if args_pred:
                            assert self.comp_result[key] == 'wrong'
                        else:
                            self.comp_result[key] = 'wrong'  # recall が下がる
                    measures.at[case, analysis].denom_gold += 1
        return measures

    def _filter_args(self,
                     args: List[BaseArgument],
                     predicate: Predicate,
                     ) -> List[BaseArgument]:
        filtered_args = []
        for arg in args:
            if isinstance(arg, SpecialArgument):
                if arg.exophor not in self.relax_exophors:  # filter out non-target exophors
                    continue
                arg.exophor = self.relax_exophors[arg.exophor]  # 「不特定:人１」なども「不特定:人」として扱う
            else:
                assert isinstance(arg, Argument)
                # filter out self-anaphora and cataphoras
                if predicate.dtid == arg.dtid or (predicate.dtid < arg.dtid and arg.sid != predicate.sid):
                    continue
            filtered_args.append(arg)
        return filtered_args

    def _evaluate_bridging(self) -> pd.Series:
        """calculate bridging anaphora resolution scores"""
        measures: Dict[str, Measure] = OrderedDict((anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values())
        dtid2anaphor_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.bridgings_pred}
        dtid2anaphor_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.bridgings_gold}

        for dtid in range(len(self.document_pred.bp_list())):
            if dtid in dtid2anaphor_pred:
                anaphor_pred = dtid2anaphor_pred[dtid]
                antecedents_pred: List[BaseArgument] = \
                    self._filter_args(self.document_pred.get_arguments(anaphor_pred, relax=False)['ノ'], anaphor_pred)
            else:
                antecedents_pred = []
            assert len(antecedents_pred) in (0, 1)  # in bert_pas_analysis, predict one argument for one predicate

            if dtid in dtid2anaphor_gold:
                anaphor_gold: Predicate = dtid2anaphor_gold[dtid]
                antecedents_gold: List[BaseArgument] = \
                    self._filter_args(self.document_gold.get_arguments(anaphor_gold, relax=False)['ノ'], anaphor_gold)
                arguments: Dict[str, List[BaseArgument]] = self.document_gold.get_arguments(anaphor_gold, relax=True)
                antecedents_gold_relaxed: List[BaseArgument] = \
                    self._filter_args(arguments['ノ'] + arguments['ノ？'], anaphor_gold)
            else:
                antecedents_gold = antecedents_gold_relaxed = []

            key = (dtid, 'ノ')

            # calculate precision
            if antecedents_pred:
                antecedent_pred = antecedents_pred[0]
                if antecedent_pred in antecedents_gold_relaxed:
                    # use dep_type of gold antecedent if possible
                    antecedent_gold = antecedents_gold_relaxed[antecedents_gold_relaxed.index(antecedent_pred)]
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold.dep_type]
                    if analysis == 'overt':
                        analysis = 'dep'
                    self.comp_result[key] = analysis
                    measures[analysis].correct += 1
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_pred.dep_type]
                    if analysis == 'overt':
                        analysis = 'dep'
                    self.comp_result[key] = 'wrong'
                measures[analysis].denom_pred += 1

            # calculate recall
            if antecedents_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                antecedent_gold = None
                for ant in antecedents_gold_relaxed:
                    if ant in antecedents_pred:
                        antecedent_gold = ant  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if antecedent_gold is not None:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold.dep_type]
                    if analysis == 'overt':
                        analysis = 'dep'
                    assert self.comp_result[key] == analysis
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedents_gold[0].dep_type]
                    if analysis == 'overt':
                        analysis = 'dep'
                    if antecedents_pred:
                        assert self.comp_result[key] == 'wrong'
                    else:
                        self.comp_result[key] = 'wrong'
                measures[analysis].denom_gold += 1
        return pd.Series(measures)

    def _evaluate_coref(self) -> pd.Series:
        """calculate coreference resolution scores"""
        measure = Measure()
        dtid2mention_pred: Dict[int, Mention] = {bp.dtid: self.document_pred.mentions[bp.dtid]
                                                 for bp in self.mentions_pred
                                                 if bp.dtid in self.document_pred.mentions}
        dtid2mention_gold: Dict[int, Mention] = {bp.dtid: self.document_gold.mentions[bp.dtid]
                                                 for bp in self.mentions_gold
                                                 if bp.dtid in self.document_gold.mentions}
        for dtid in range(len(self.document_pred.bp_list())):
            if dtid in dtid2mention_pred:
                src_mention_pred = dtid2mention_pred[dtid]
                tgt_mentions_pred = \
                    self.filter_mentions(self.document_pred.get_siblings(src_mention_pred), src_mention_pred)
                exophors_pred = {e.exophor for e in map(self.document_pred.entities.get, src_mention_pred.eids)
                                 if e.is_special}
            else:
                tgt_mentions_pred = exophors_pred = set()

            if dtid in dtid2mention_gold:
                src_mention_gold = dtid2mention_gold[dtid]
                tgt_mentions_gold = self.filter_mentions(self.document_gold.get_siblings(src_mention_gold, relax=False),
                                                         src_mention_gold)
                tgt_mentions_gold_relaxed = self.filter_mentions(
                    self.document_gold.get_siblings(src_mention_gold, relax=True), src_mention_gold)
                exophors_gold = {self.relax_exophors[e.exophor] for e
                                 in map(self.document_gold.entities.get, src_mention_gold.eids)
                                 if e.is_special and e.exophor in self.relax_exophors}
                exophors_gold_relaxed = {self.relax_exophors[e.exophor] for e
                                         in map(self.document_gold.entities.get, src_mention_gold.all_eids)
                                         if e.is_special and e.exophor in self.relax_exophors}
            else:
                tgt_mentions_gold = tgt_mentions_gold_relaxed = exophors_gold = exophors_gold_relaxed = set()

            key = (dtid, '=')

            # calculate precision
            if tgt_mentions_pred or exophors_pred:
                if (tgt_mentions_pred & tgt_mentions_gold_relaxed) or (exophors_pred & exophors_gold_relaxed):
                    self.comp_result[key] = 'correct'
                    measure.correct += 1
                else:
                    self.comp_result[key] = 'wrong'
                measure.denom_pred += 1

            # calculate recall
            if tgt_mentions_gold or exophors_gold or (self.comp_result.get(key, None) == 'correct'):
                if (tgt_mentions_pred & tgt_mentions_gold_relaxed) or (exophors_pred & exophors_gold_relaxed):
                    assert self.comp_result[key] == 'correct'
                else:
                    self.comp_result[key] = 'wrong'
                measure.denom_gold += 1
        return pd.Series([measure], index=['all'])

    @staticmethod
    def filter_mentions(tgt_mentions: Set[Mention], src_mention: Mention) -> Set[Mention]:
        """filter out cataphors"""
        return {tgt_mention for tgt_mention in tgt_mentions if tgt_mention.dtid < src_mention.dtid}


@dataclass(frozen=True)
class ScoreResult:
    """A data class for storing the numerical result of an evaluation"""
    measures_pas: Optional[pd.DataFrame]
    measures_bridging: Optional[pd.Series]
    measure_coref: Optional[pd.Series]

    def to_dict(self) -> Dict[str, Dict[str, 'Measure']]:
        """convert data to dictionary"""
        df_all = pd.DataFrame(index=['all_case'])
        if self.pas:
            df_pas: pd.DataFrame = self.measures_pas.copy()
            df_pas['zero'] = df_pas['zero_intra'] + df_pas['zero_inter'] + df_pas['zero_exophora']
            df_pas['dep_zero'] = df_pas['zero'] + df_pas['dep']
            df_pas['all'] = df_pas['dep_zero'] + df_pas['overt']
            df_all = pd.concat([df_pas, df_all])
            df_all.loc['all_case'] = df_pas.sum(axis=0)

        if self.bridging:
            df_bar = self.measures_bridging.copy()
            df_bar['zero'] = df_bar['zero_intra'] + df_bar['zero_inter'] + df_bar['zero_exophora']
            df_bar['dep_zero'] = df_bar['zero'] + df_bar['dep']
            assert df_bar['overt'] == Measure()  # No overt in BAR
            df_bar['all'] = df_bar['dep_zero']
            df_all.at['all_case', 'bridging'] = df_bar['all']

        if self.coreference:
            df_all.at['all_case', 'coreference'] = self.measure_coref['all']

        return {k1: {k2: v2 for k2, v2 in v1.items() if pd.notnull(v2)}
                for k1, v1 in df_all.to_dict(orient='index').items()}

    def export_txt(self,
                   destination: Union[str, Path, TextIO]
                   ) -> None:
        """Export the evaluation results in a text format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
        """
        lines = []
        for key, ms in self.to_dict().items():
            lines.append(f'{key}格' if self.pas and key in self.measures_pas.index else key)
            for analysis, measure in ms.items():
                lines.append(f'  {analysis}')
                lines.append(f'    precision: {measure.precision:.4f} ({measure.correct}/{measure.denom_pred})')
                lines.append(f'    recall   : {measure.recall:.4f} ({measure.correct}/{measure.denom_gold})')
                lines.append(f'    F        : {measure.f1:.4f}')
        text = '\n'.join(lines) + '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def export_csv(self,
                   destination: Union[str, Path, TextIO],
                   sep: str = ','
                   ) -> None:
        """Export the evaluation results in a csv format.

        Args:
            destination (Union[str, Path, TextIO]): 書き出す先
            sep (str): 区切り文字 (default: ',')
        """
        text = ''
        result_dict = self.to_dict()
        text += 'case' + sep
        text += sep.join(result_dict['all_case'].keys()) + '\n'
        for case, measures in result_dict.items():
            text += CASE2YOMI.get(case, case) + sep
            text += sep.join(f'{measure.f1:.6}' for measure in measures.values())
            text += '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    @property
    def pas(self):
        """Whether self includes the score of predicate-argument structure analysis."""
        return self.measures_pas is not None

    @property
    def bridging(self):
        """Whether self includes the score of bridging anaphora resolution."""
        return self.measures_bridging is not None

    @property
    def coreference(self):
        """Whether self includes the score of coreference resolution."""
        return self.measure_coref is not None

    def __add__(self, other: 'ScoreResult') -> 'ScoreResult':
        measures_pas = self.measures_pas + other.measures_pas if self.pas else None
        measures_bridging = self.measures_bridging + other.measures_bridging if self.bridging else None
        measure_coref = self.measure_coref + other.measure_coref if self.coreference else None
        return ScoreResult(measures_pas, measures_bridging, measure_coref)


@dataclass
class Measure:
    """A data class to calculate and represent F-measure"""
    denom_pred: int = 0
    denom_gold: int = 0
    correct: int = 0

    def __add__(self, other: 'Measure'):
        return Measure(self.denom_pred + other.denom_pred,
                       self.denom_gold + other.denom_gold,
                       self.correct + other.correct)

    def __eq__(self, other: 'Measure'):
        return self.denom_pred == other.denom_pred and \
               self.denom_gold == other.denom_gold and \
               self.correct == other.correct

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            return .0
        return self.correct / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            return .0
        return self.correct / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            return .0
        return 2 * self.correct / (self.denom_pred + self.denom_gold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-dir', default=None, type=str,
                        help='path to directory where system output KWDLC files exist (default: None)')
    parser.add_argument('--gold-dir', default=None, type=str,
                        help='path to directory where gold KWDLC files exist (default: None)')
    parser.add_argument('--coreference', '--coref', '--cr', action='store_true', default=False,
                        help='perform coreference resolution')
    parser.add_argument('--bridging', '--brg', '--bar', action='store_true', default=False,
                        help='perform bridging anaphora resolution')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２',
                        help='case strings separated by ","')
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人,不特定:物',
                        help='exophor strings separated by ","')
    parser.add_argument('--read-prediction-from-pas-tag', action='store_true', default=False,
                        help='use <述語項構造:> tag instead of <rel > tag in prediction files')
    parser.add_argument('--pas-target', choices=['', 'pred', 'noun', 'all'], default='pred',
                        help='PAS analysis evaluation target (pred: verbal predicates, noun: nominal predicates)')
    parser.add_argument('--result-html', default=None, type=str,
                        help='path to html file which prediction result is exported (default: None)')
    parser.add_argument('--result-csv', default=None, type=str,
                        help='path to csv file which prediction result is exported (default: None)')
    args = parser.parse_args()

    reader_gold = KyotoReader(Path(args.gold_dir), extract_nes=False, use_pas_tag=False)
    reader_pred = KyotoReader(
        Path(args.prediction_dir),
        extract_nes=False,
        use_pas_tag=args.read_prediction_from_pas_tag,
    )
    documents_pred = reader_pred.process_all_documents()
    documents_gold = reader_gold.process_all_documents()

    assert set(args.case_string.split(',')) <= set(CASE2YOMI.keys())
    msg = '"ノ" found in case string. If you want to perform bridging anaphora resolution, specify "--bridging" ' \
          'option instead'
    assert 'ノ' not in args.case_string.split(','), msg
    scorer = Scorer(documents_pred, documents_gold,
                    target_cases=args.case_string.split(','),
                    target_exophors=args.exophors.split(','),
                    coreference=args.coreference,
                    bridging=args.bridging,
                    pas_target=args.pas_target)
    result = scorer.run()
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    if args.result_csv:
        result.export_csv(args.result_csv)
    result.export_txt(sys.stdout)


if __name__ == '__main__':
    main()
