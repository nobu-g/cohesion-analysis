import io
import sys
import logging
import argparse
import textwrap
from pathlib import Path
from typing import List, Dict, Set, Union, Optional, TextIO
from collections import OrderedDict

from pyknp import BList
from kyoto_reader import KyotoReader, Document, Argument, SpecialArgument, BaseArgument, Predicate, Mention

from utils.util import OrderedDefaultDict
from utils.constants import CASE2YOMI

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class Scorer:
    DEPTYPE2ANALYSIS = OrderedDict([('overt', 'overt'),
                                    ('dep', 'case_analysis'),
                                    ('intra', 'zero_intra_sentential'),
                                    ('inter', 'zero_inter_sentential'),
                                    ('exo', 'zero_exophora')])

    def __init__(self,
                 documents_pred: List[Document],
                 documents_gold: List[Document],
                 target_cases: List[str],
                 target_exophors: List[str],
                 coreference: bool = False,
                 bridging: bool = False,
                 pas_target: str = 'pred'):
        # long document may have been ignored
        assert set(doc.doc_id for doc in documents_pred) <= set(doc.doc_id for doc in documents_gold)
        self.cases: List[str] = target_cases
        self.bridging: bool = bridging
        self.doc_ids: List[str] = [doc.doc_id for doc in documents_pred]
        self.did2document_pred: Dict[str, Document] = {doc.doc_id: doc for doc in documents_pred}
        self.did2document_gold: Dict[str, Document] = {doc.doc_id: doc for doc in documents_gold}
        self.coreference = coreference
        self.comp_result: Dict[tuple, str] = {}
        self.measures: Dict[str, Dict[str, Measure]] = OrderedDict(
            (case, OrderedDict((anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values()))
            for case in self.cases)
        self.measure_coref: Measure = Measure()
        self.measures_bridging: Dict[str, Measure] = OrderedDict(
            (anal, Measure()) for anal in Scorer.DEPTYPE2ANALYSIS.values())
        self.relax_exophors: Dict[str, str] = {}
        for exophor in target_exophors:
            self.relax_exophors[exophor] = exophor
            if exophor in ('不特定:人', '不特定:物', '不特定:状況'):
                for n in ('１', '２', '３', '４', '５', '６', '７', '８', '９', '１０', '１１'):
                    self.relax_exophors[exophor + n] = exophor

        self.did2predicates_pred: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        self.did2predicates_gold: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        self.did2bridgings_pred: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        self.did2bridgings_gold: Dict[str, List[Predicate]] = OrderedDefaultDict(list)
        self.did2mentions_pred: Dict[str, List[Mention]] = OrderedDefaultDict(list)
        self.did2mentions_gold: Dict[str, List[Mention]] = OrderedDefaultDict(list)
        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            for predicate_pred in document_pred.get_predicates():
                features = predicate_pred.tag.features
                if (pas_target in ('pred', 'all') and '用言' in features) or \
                        (pas_target in ('noun', 'all') and '非用言格解析' in features and '体言' in features):
                    self.did2predicates_pred[doc_id].append(predicate_pred)
                if self.bridging and '体言' in features and '非用言格解析' not in features:
                    self.did2bridgings_pred[doc_id].append(predicate_pred)
            for predicate_gold in document_gold.get_predicates():
                features = predicate_gold.tag.features
                if (pas_target in ('pred', 'all') and '用言' in features) or \
                        (pas_target in ('noun', 'all') and '非用言格解析' in features and '体言' in features):
                    self.did2predicates_gold[doc_id].append(predicate_gold)
                if self.bridging and '体言' in features and '非用言格解析' not in features:
                    self.did2bridgings_gold[doc_id].append(predicate_gold)

            for mention_pred in document_pred.mentions.values():
                if '体言' in mention_pred.tag.features:
                    self.did2mentions_pred[doc_id].append(mention_pred)
            for mention_gold in document_gold.mentions.values():
                if '体言' in mention_gold.tag.features:
                    self.did2mentions_gold[doc_id].append(mention_gold)

        for doc_id in self.doc_ids:
            document_pred = self.did2document_pred[doc_id]
            document_gold = self.did2document_gold[doc_id]
            self._evaluate_pas(doc_id, document_pred, document_gold)
            if self.bridging:
                self._evaluate_bridging(doc_id, document_pred, document_gold)
            if self.coreference:
                self._evaluate_coref(doc_id, document_pred, document_gold)

    def _evaluate_pas(self, doc_id: str, document_pred: Document, document_gold: Document):
        """calculate PAS analysis scores"""
        dtid2predicate_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2predicates_pred[doc_id]}
        dtid2predicate_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2predicates_gold[doc_id]}

        for dtid in range(len(document_pred.tag_list())):
            if dtid in dtid2predicate_pred:
                predicate_pred = dtid2predicate_pred[dtid]
                arguments_pred = document_pred.get_arguments(predicate_pred, relax=False)
            else:
                arguments_pred = None

            if dtid in dtid2predicate_gold:
                predicate_gold = dtid2predicate_gold[dtid]
                arguments_gold = document_gold.get_arguments(predicate_gold, relax=False)
                arguments_gold_relaxed = document_gold.get_arguments(predicate_gold, relax=True)
            else:
                predicate_gold = arguments_gold = arguments_gold_relaxed = None

            for case in self.cases:
                args_pred: List[BaseArgument] = arguments_pred[case] if arguments_pred is not None else []
                assert len(args_pred) in (0, 1)  # this project predicts one argument for one predicate
                if predicate_gold is not None:
                    args_gold = self._filter_args(arguments_gold[case], predicate_gold)
                    args_gold_relaxed = self._filter_args(
                        arguments_gold_relaxed[case] + (arguments_gold_relaxed['判ガ'] if case == 'ガ' else []),
                        predicate_gold)
                else:
                    args_gold = args_gold_relaxed = []

                key = (doc_id, dtid, case)

                # calculate precision
                if args_pred:
                    arg = args_pred[0]
                    analysis = Scorer.DEPTYPE2ANALYSIS[arg.dep_type]
                    if arg in args_gold_relaxed:
                        self.comp_result[key] = analysis
                        self.measures[case][analysis].correct += 1
                    else:
                        self.comp_result[key] = 'wrong'  # precision が下がる
                    self.measures[case][analysis].denom_pred += 1

                # calculate recall
                # 正解が複数ある場合、そのうち一つが当てられていればそれを正解に採用．
                # いずれも当てられていなければ、relax されていない項から一つを選び正解に採用．
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
                    self.measures[case][analysis].denom_gold += 1

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

    def _evaluate_bridging(self, doc_id: str, document_pred: Document, document_gold: Document):
        dtid2anaphor_pred: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2bridgings_pred[doc_id]}
        dtid2anaphor_gold: Dict[int, Predicate] = {pred.dtid: pred for pred in self.did2bridgings_gold[doc_id]}

        for dtid in range(len(document_pred.tag_list())):
            if dtid in dtid2anaphor_pred:
                anaphor_pred: Predicate = dtid2anaphor_pred[dtid]
                antecedents_pred: List[BaseArgument] = \
                    self._filter_args(document_pred.get_arguments(anaphor_pred, relax=False)['ノ'], anaphor_pred)
            else:
                antecedents_pred = []
            assert len(antecedents_pred) in (0, 1)  # this project predicts one argument for one predicate

            if dtid in dtid2anaphor_gold:
                anaphor_gold: Predicate = dtid2anaphor_gold[dtid]
                antecedents_gold: List[BaseArgument] = \
                    self._filter_args(document_gold.get_arguments(anaphor_gold, relax=False)['ノ'], anaphor_gold)
                antecedents_gold_relaxed: List[BaseArgument] = \
                    self._filter_args(document_gold.get_arguments(anaphor_gold, relax=True)['ノ'] +
                                      document_gold.get_arguments(anaphor_gold, relax=True)['ノ？'], anaphor_gold)
            else:
                antecedents_gold = antecedents_gold_relaxed = []

            key = (doc_id, dtid, 'ノ')

            # calculate precision
            if antecedents_pred:
                antecedent_pred = antecedents_pred[0]
                analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_pred.dep_type]
                if antecedent_pred in antecedents_gold_relaxed:
                    self.comp_result[key] = analysis
                    self.measures_bridging[analysis].correct += 1
                else:
                    self.comp_result[key] = 'wrong'
                self.measures_bridging[analysis].denom_pred += 1

            # calculate recall
            if antecedents_gold or (self.comp_result.get(key, None) in Scorer.DEPTYPE2ANALYSIS.values()):
                antecedent_gold = None
                for ant in antecedents_gold_relaxed:
                    if ant in antecedents_pred:
                        antecedent_gold = ant  # 予測されている先行詞を優先して正解の先行詞に採用
                        break
                if antecedent_gold is not None:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedent_gold.dep_type]
                    assert self.comp_result[key] == analysis
                else:
                    analysis = Scorer.DEPTYPE2ANALYSIS[antecedents_gold[0].dep_type]
                    if antecedents_pred:
                        assert self.comp_result[key] == 'wrong'
                    else:
                        self.comp_result[key] = 'wrong'
                self.measures_bridging[analysis].denom_gold += 1

    def _evaluate_coref(self, doc_id: str, document_pred: Document, document_gold: Document):
        dtid2mention_pred: Dict[int, Mention] = {ment.dtid: ment for ment in self.did2mentions_pred[doc_id]}
        dtid2mention_gold: Dict[int, Mention] = {ment.dtid: ment for ment in self.did2mentions_gold[doc_id]}

        for dtid in range(len(document_pred.tag_list())):
            if dtid in dtid2mention_pred:
                src_mention_pred = dtid2mention_pred[dtid]
                tgt_mentions_pred = \
                    self._filter_mentions(document_pred.get_siblings(src_mention_pred), src_mention_pred)
                exophors_pred = [e.exophor for e in map(document_pred.entities.get, src_mention_pred.eids)
                                 if e.is_special]
            else:
                tgt_mentions_pred = exophors_pred = []

            if dtid in dtid2mention_gold:
                src_mention_gold = dtid2mention_gold[dtid]
                tgt_mentions_gold = \
                    self._filter_mentions(document_gold.get_siblings(src_mention_gold, relax=False), src_mention_gold)
                tgt_mentions_gold_relaxed = \
                    self._filter_mentions(document_gold.get_siblings(src_mention_gold, relax=True), src_mention_gold)
                exophors_gold = [e.exophor for e in map(document_gold.entities.get, src_mention_gold.eids)
                                 if e.is_special and e.exophor in self.relax_exophors.values()]
                exophors_gold_relaxed = [e.exophor for e in map(document_gold.entities.get, src_mention_gold.all_eids)
                                         if e.is_special and e.exophor in self.relax_exophors.values()]
            else:
                tgt_mentions_gold = tgt_mentions_gold_relaxed = exophors_gold = exophors_gold_relaxed = []

            key = (doc_id, dtid, '=')

            # calculate precision
            if tgt_mentions_pred or exophors_pred:
                if (set(tgt_mentions_pred) & set(tgt_mentions_gold_relaxed)) \
                        or (set(exophors_pred) & set(exophors_gold_relaxed)):
                    self.comp_result[key] = 'correct'
                    self.measure_coref.correct += 1
                else:
                    self.comp_result[key] = 'wrong'
                self.measure_coref.denom_pred += 1

            # calculate recall
            if tgt_mentions_gold or exophors_gold or (self.comp_result.get(key, None) == 'correct'):
                if (set(tgt_mentions_pred) & set(tgt_mentions_gold_relaxed)) \
                        or (set(exophors_pred) & set(exophors_gold_relaxed)):
                    assert self.comp_result[key] == 'correct'
                else:
                    self.comp_result[key] = 'wrong'
                self.measure_coref.denom_gold += 1

    @staticmethod
    def _filter_mentions(tgt_mentions: Set[Mention], src_mention: Mention) -> List[Mention]:
        return [tgt_mention for tgt_mention in tgt_mentions if tgt_mention.dtid < src_mention.dtid]

    def result_dict(self) -> Dict[str, Dict[str, 'Measure']]:
        result = OrderedDict()
        all_case_result = OrderedDefaultDict(lambda: Measure())
        for case, measures in self.measures.items():
            case_result = OrderedDefaultDict(lambda: Measure())
            case_result.update(measures)
            case_result['zero_all'] = case_result['zero_intra_sentential'] + \
                                      case_result['zero_inter_sentential'] + \
                                      case_result['zero_exophora']
            case_result['all'] = case_result['case_analysis'] + case_result['zero_all']
            case_result['all_w_overt'] = case_result['all'] + case_result['overt']
            for analysis, measure in case_result.items():
                all_case_result[analysis] += measure
            result[case] = case_result

        if self.coreference:
            all_case_result['coreference'] = self.measure_coref

        if self.bridging:
            case_result = OrderedDefaultDict(lambda: Measure())
            case_result.update(self.measures_bridging)
            case_result['zero_all'] = case_result['zero_intra_sentential'] + \
                                      case_result['zero_inter_sentential'] + \
                                      case_result['zero_exophora']
            case_result['all'] = case_result['case_analysis'] + case_result['zero_all']
            case_result['all_w_overt'] = case_result['all'] + case_result['overt']
            all_case_result['bridging'] = case_result['all']
            all_case_result['bridging_w_overt'] = case_result['all_w_overt']

        result['all_case'] = all_case_result
        return result

    def export_txt(self, destination: Union[str, Path, TextIO]):
        lines = []
        for case, measures in self.result_dict().items():
            if case in self.cases:
                lines.append(f'{case}格')
            else:
                lines.append(f'{case}')
            for analysis, measure in measures.items():
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

    def export_csv(self, destination: Union[str, Path, TextIO], sep: str = ','):
        text = ''
        result_dict = self.result_dict()
        text += 'case' + sep
        text += sep.join(result_dict['all_case'].keys()) + '\n'
        for case, measures in result_dict.items():
            text += CASE2YOMI.get(case, case) + sep
            text += sep.join(f'{measure.f1:.5}' for measure in measures.values())
            text += '\n'

        if isinstance(destination, str) or isinstance(destination, Path):
            with Path(destination).open('wt') as writer:
                writer.write(text)
        elif isinstance(destination, io.TextIOBase):
            destination.write(text)

    def write_html(self, output_file: Union[str, Path]):
        if isinstance(output_file, str):
            output_file = Path(output_file)
        with output_file.open('w') as writer:
            writer.write('<html lang="ja">\n')
            writer.write(self._html_header())
            writer.write('<body>\n')
            writer.write('<h2>Bert Result</h2>\n')
            writer.write('<pre>\n')
            for doc_id in self.doc_ids:
                document_pred = self.did2document_pred[doc_id]
                document_gold = self.did2document_gold[doc_id]
                writer.write('<h3 class="obi1">')
                for sid, sentence in document_gold.sid2sentence.items():
                    writer.write(sid + ' ')
                    writer.write(''.join(bnst.midasi for bnst in sentence.bnst_list()))
                    writer.write('<br>')
                writer.write('</h3>\n')
                writer.write('<table>')
                writer.write('<tr>\n<th>gold</th>\n')
                writer.write('<th>prediction</th>\n</tr>')

                writer.write('<tr>')
                # gold
                writer.write('<td><pre>\n')
                for sid in document_gold.sid2sentence.keys():
                    self._draw_tree(sid,
                                    self.did2predicates_gold[doc_id],
                                    self.did2mentions_gold[doc_id],
                                    self.did2bridgings_gold[doc_id],
                                    document_gold,
                                    fh=writer)
                    writer.write('\n')
                writer.write('</pre>')
                # prediction
                writer.write('<td><pre>\n')
                for sid in document_pred.sid2sentence.keys():
                    self._draw_tree(sid,
                                    self.did2predicates_pred[doc_id],
                                    self.did2mentions_pred[doc_id],
                                    self.did2bridgings_pred[doc_id],
                                    document_pred,
                                    fh=writer)
                    writer.write('\n')
                writer.write('</pre>\n</tr>\n')

                writer.write('</table>\n')
            writer.write('</pre>\n')
            writer.write('</body>\n')
            writer.write('</html>\n')

    @staticmethod
    def _html_header():
        return textwrap.dedent('''
        <head>
        <meta charset="utf-8" />
        <title>Bert Result</title>
        <style type="text/css">
        td {
            font-size: 11pt;
            border: 1px solid #606060;
            vertical-align: top;
            margin: 5pt;
        }
        .obi1 {
            background-color: pink;
            font-size: 12pt;
        }
        pre {
            font-family: "ＭＳ ゴシック", "Osaka-Mono", "Osaka-等幅", "さざなみゴシック", "Sazanami Gothic", DotumChe,
            GulimChe, BatangChe, MingLiU, NSimSun, Terminal;
            white-space: pre;
        }
        </style>
        </head>
        ''')

    def _draw_tree(self,
                   sid: str,
                   predicates: List[Predicate],
                   mentions: List[Mention],
                   anaphors: List[Predicate],
                   document: Document,
                   fh: Optional[TextIO] = None,
                   html: bool = True
                   ) -> None:
        """sid で指定された文の述語項構造・共参照関係をツリー形式で fh に書き出す

        Args:
            sid (str): 出力対象の文ID
            predicates (List[Predicate]): document に含まれる全ての述語
            mentions (List[Mention]): document に含まれる全ての mention
            anaphors (List[Predicate]): document に含まれる全ての橋渡し照応詞
            document (Document): sid が含まれる文書
            fh (Optional[TextIO]): 出力ストリーム
            html (bool): html 形式で出力するかどうか
        """
        blist: BList = document.sid2sentence[sid].blist
        with io.StringIO() as string:
            blist.draw_tag_tree(fh=string, show_pos=False)
            tree_strings = string.getvalue().rstrip('\n').split('\n')
        assert len(tree_strings) == len(blist.tag_list())
        all_midasis = [m.midasi for m in document.mentions.values()]
        tid2predicate = {predicate.tid: predicate for predicate in predicates if predicate.sid == sid}
        tid2anaphor = {anaphor.tid: anaphor for anaphor in anaphors if anaphor.sid == sid}
        for tid in range(len(tree_strings)):
            cases = []
            predicate = None
            if tid in tid2predicate:
                cases += self.cases
                predicate = tid2predicate[tid]
            if tid in tid2anaphor:
                cases += ['ノ']
                predicate = tid2anaphor[tid]
            if predicate is None:
                continue
            tree_strings[tid] += '  '
            arguments = document.get_arguments(predicate)
            for case in cases:
                args = arguments[case]
                if case == 'ガ':
                    args += arguments['判ガ']
                if case == 'ノ':
                    args += arguments['ノ？']
                color: str = 'gray'
                result = self.comp_result.get((document.doc_id, predicate.dtid, case), None)
                if result == 'overt':
                    color = 'green'
                elif result in Scorer.DEPTYPE2ANALYSIS.values():
                    color = 'blue'
                elif result == 'wrong':
                    for arg in args:
                        if isinstance(arg, Argument):
                            color = 'red'
                        elif arg.midasi in self.relax_exophors:
                            color = 'red'
                targets = set()
                for arg in args:
                    target = arg.midasi
                    if all_midasis.count(arg.midasi) > 1 and isinstance(arg, Argument):
                        target += str(arg.dtid)
                    targets.add(target)
                if html:
                    tree_strings[tid] += f'<font color="{color}">{",".join(targets)}:{case}</font> '
                else:
                    tree_strings[tid] += f'{",".join(targets)}:{case} '
        if self.coreference:
            for src_mention in filter(lambda m: m.sid == sid, mentions):
                tgt_mentions_relaxed = self._filter_mentions(
                    document.get_siblings(src_mention, relax=True), src_mention)
                targets = set()
                for tgt_mention in tgt_mentions_relaxed:
                    target: str = tgt_mention.midasi
                    if all_midasis.count(target) > 1:
                        target += str(tgt_mention.dtid)
                    targets.add(target)
                for eid in src_mention.eids:
                    entity = document.entities[eid]
                    if entity.exophor in self.relax_exophors.values():
                        targets.add(entity.exophor)
                if not targets:
                    continue
                result = self.comp_result.get((document.doc_id, src_mention.dtid, '='), None)
                result2color = {'correct': 'blue', 'wrong': 'red', None: 'gray'}
                tid = src_mention.tid
                tree_strings[tid] += '  ＝:'
                if html:
                    tree_strings[tid] += f'<span style="background-color:#e0e0e0;color:{result2color[result]}">' \
                                         + ','.join(targets) + '</span> '
                else:
                    tree_strings[tid] += ','.join(targets)

        print('\n'.join(tree_strings), file=fh)


class Measure:
    def __init__(self,
                 denom_pred: int = 0,
                 denom_gold: int = 0,
                 correct: int = 0):
        self.denom_pred = denom_pred
        self.denom_gold = denom_gold
        self.correct = correct

    def __add__(self, other: 'Measure'):
        return Measure(self.denom_pred + other.denom_pred,
                       self.denom_gold + other.denom_gold,
                       self.correct + other.correct)

    @property
    def precision(self) -> float:
        if self.denom_pred == 0:
            logger.warning('zero division at precision')
            return 0.0
        return self.correct / self.denom_pred

    @property
    def recall(self) -> float:
        if self.denom_gold == 0:
            logger.warning('zero division at recall')
            return 0.0
        return self.correct / self.denom_gold

    @property
    def f1(self) -> float:
        if self.denom_pred + self.denom_gold == 0:
            logger.warning('zero division at f1')
            return 0.0
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
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人',
                        help='exophor strings separated by ","')
    parser.add_argument('--coref-string', type=str, default='=,=構,=≒,=構≒',
                        help='coreference strings separated by ","')
    parser.add_argument('--read-prediction-from-pas-tag', action='store_true', default=False,
                        help='use <述語項構造:> tag instead of <rel > tag in prediction files')
    parser.add_argument('--pas-target', choices=['', 'pred', 'noun', 'all'], default='pred',
                        help='PAS analysis evaluation target (pred: verbal predicates, noun: nominal predicates)')
    parser.add_argument('--result-html', default=None, type=str,
                        help='path to html file which prediction result is exported (default: None)')
    parser.add_argument('--result-csv', default=None, type=str,
                        help='path to csv file which prediction result is exported (default: None)')
    args = parser.parse_args()

    reader_gold = KyotoReader(
        Path(args.gold_dir),
        target_cases='ガ,ヲ,ニ,ガ２,ノ,ノ？,判ガ'.split(','),
        target_corefs=args.coref_string.split(','),
        extract_nes=False,
        use_pas_tag=False,
    )
    reader_pred = KyotoReader(
        Path(args.prediction_dir),
        target_cases=reader_gold.target_cases,
        target_corefs=reader_gold.target_corefs,
        extract_nes=False,
        use_pas_tag=args.read_prediction_from_pas_tag,
    )
    documents_pred = list(reader_pred.process_all_documents())
    documents_gold = list(reader_gold.process_all_documents())

    assert set(args.case_string.split(',')) <= set(CASE2YOMI.keys())
    msg = '"ノ" found in case string. If you want to perform bridging anaphora resolution, specify "--bridging" option'
    assert 'ノ' not in args.case_string.split(','), msg
    scorer = Scorer(documents_pred, documents_gold,
                    target_cases=args.case_string.split(','),
                    target_exophors=args.exophors.split(','),
                    coreference=args.coreference,
                    bridging=args.bridging,
                    pas_target=args.pas_target)
    if args.result_html:
        scorer.write_html(Path(args.result_html))
    if args.result_csv:
        scorer.export_csv(args.result_csv)
    scorer.export_txt(sys.stdout)


if __name__ == '__main__':
    main()
