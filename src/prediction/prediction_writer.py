import io
import re
from collections import defaultdict
from logging import Logger, getLogger
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple, Union, TextIO

from kyoto_reader import KyotoReader, Document, Pas, BaseArgument, Argument, SpecialArgument, BasePhrase

from data_loader.dataset import PASDataset
from data_loader.dataset.read_example import PasExample


class PredictionKNPWriter:
    """A class to write the system output in a KNP format.

    Args:
        dataset (PASDataset): 解析対象のデータセット
        logger (Logger): ロガー (default: None)
        use_knp_overt (bool): overt について本システムの解析結果ではなく、KNP の格解析結果を使用する (default: True)
    """
    REL_PAT = re.compile(r'<rel type="([^\s]+?)"(?: mode="([^>]+?)")? target="(.*?)"(?: sid="(.*?)" id="(.+?)")?/>')
    TAG_PAT = re.compile(r'^\+ -?\d+\w ?')

    def __init__(self,
                 dataset: PASDataset,
                 logger: Logger = None,
                 use_knp_overt: bool = True,
                 ) -> None:
        self.examples: List[PasExample] = dataset.examples
        self.cases: List[str] = dataset.target_cases
        self.bridging: bool = dataset.bridging
        self.coreference: bool = dataset.coreference
        self.relations: List[str] = dataset.relations
        self.exophors: List[str] = dataset.target_exophors
        self.index_to_special: Dict[int, str] = {idx: token for token, idx in dataset.special_to_index.items()}
        self.documents: List[Document] = dataset.documents
        self.logger: Logger = logger or getLogger(__name__)
        self.use_knp_overt: bool = use_knp_overt
        self.kc: bool = dataset.kc
        self.reader: KyotoReader = dataset.reader

    def write(self,
              arguments_sets: List[List[List[int]]],
              destination: Union[Path, TextIO, None],
              skip_untagged: bool = True,
              add_pas_tag: bool = True,
              ) -> List[Document]:
        """Write final predictions to the file.

        Args:
            arguments_sets (List[List[List[int]]]): モデル出力
            destination (Union[Path, TextIO, None]): 解析済み文書の出力先
            skip_untagged (bool): 解析に失敗した文書を出力しないかどうか (default: True)
            add_pas_tag (bool): 解析結果に<述語項構造 >タグを付与するかどうか (default: True)
        Returns:
            List[Document]: 解析済み文書
        """

        if isinstance(destination, Path):
            self.logger.info(f'Writing predictions to: {destination}')
            destination.mkdir(exist_ok=True)
        elif not (destination is None or isinstance(destination, io.TextIOBase)):
            self.logger.warning('invalid output destination')

        did2examples = {ex.doc_id: ex for ex in self.examples}
        did2arguments_sets = {ex.doc_id: arguments_set for ex, arguments_set in zip(self.examples, arguments_sets)}

        did2knps: Dict[str, List[str]] = defaultdict(list)
        for document in self.documents:
            did = document.doc_id
            input_knp_lines = document.knp_string.strip().split('\n')
            if did in did2examples:
                output_knp_lines = self._rewrite_rel(input_knp_lines,
                                                     did2examples[did],
                                                     did2arguments_sets[did],
                                                     document)  # overtを抽出するためこれはreparse後に格解析したものがいい
            else:
                if skip_untagged:
                    continue
                assert all('<rel ' not in line for line in input_knp_lines)
                output_knp_lines = input_knp_lines

            knp_strings: List[str] = []
            buff = ''
            for knp_line in output_knp_lines:
                buff += knp_line + '\n'
                if knp_line.strip() == 'EOS':
                    knp_strings.append(buff)
                    buff = ''
            if self.kc:
                # merge documents
                orig_did, idx = did.split('-')
                if idx == '00':
                    did2knps[orig_did] += knp_strings
                else:
                    did2knps[orig_did].append(knp_strings[-1])
            else:
                did2knps[did] = knp_strings

        documents_pred: List[Document] = []  # kc については元通り結合された文書のリスト
        for did, knp_strings in did2knps.items():
            document_pred = Document(''.join(knp_strings),
                                     did,
                                     self.reader.target_cases,
                                     self.reader.target_corefs,
                                     self.reader.relax_cases,
                                     extract_nes=False,
                                     use_pas_tag=False)
            documents_pred.append(document_pred)
            if destination is None:
                continue
            output_knp_lines = document_pred.knp_string.strip().split('\n')
            if add_pas_tag:
                output_knp_lines = self._add_pas_tag(output_knp_lines, document_pred)
            output_string = '\n'.join(output_knp_lines) + '\n'
            if isinstance(destination, Path):
                output_basename = did + '.knp'
                with destination.joinpath(output_basename).open('w') as writer:
                    writer.write(output_string)
            elif isinstance(destination, io.TextIOBase):
                destination.write(output_string)

        return documents_pred

    def _rewrite_rel(self,
                     knp_lines: List[str],
                     example: PasExample,
                     arguments_set: List[List[int]],  # (max_seq_len, cases)
                     document: Document,  # <格解析>付き
                     ) -> List[str]:
        overts = self._extract_overt(document)

        output_knp_lines = []
        dtid = 0
        sent_idx = 0
        for line in knp_lines:
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                if line == 'EOS':
                    sent_idx += 1
                continue

            assert '<rel ' not in line
            match = self.TAG_PAT.match(line)
            if match is not None:
                rel_string = self._rel_string(document.bp_list()[dtid],
                                              example,
                                              arguments_set,
                                              document,
                                              overts[dtid])
                rel_idx = match.end()
                output_knp_lines.append(line[:rel_idx] + rel_string + line[rel_idx:])
            else:
                self.logger.warning(f'invalid format line: {line}')
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _extract_overt(self,
                       document: Document,
                       ) -> Dict[int, Dict[str, int]]:
        overts: Dict[int, Dict[str, int]] = defaultdict(dict)
        for sentence in document:
            for bp in sentence.bps:
                if bp.tag.pas is None:
                    continue
                for case, args in bp.tag.pas.arguments.items():
                    if case not in self.cases:  # ノ格は表層格から overt 判定できないので overts に追加しない
                        continue
                    for arg in filter(lambda a: a.flag == 'C', args):
                        overts[bp.dtid][case] = sentence.bps[arg.tid].dmid
        return overts

    def _rel_string(self,
                    bp: BasePhrase,
                    example: PasExample,
                    arguments_set: List[List[int]],  # (max_seq_len, cases)
                    document: Document,
                    overt_dict: Dict[str, int],
                    ) -> str:
        rels: List[RelTag] = []
        dmid2bp = {dmid: bp for bp in document.bp_list() for dmid in bp.dmids}
        assert len(example.arguments_set) == len(dmid2bp)
        for dmid in bp.dmids:
            token_index: int = example.orig_to_tok_index[dmid]
            arguments: List[int] = arguments_set[token_index]
            # 助詞などの非解析対象形態素については gold_args が空になっている
            # inference時、解析対象形態素は ['NULL'] となる
            is_targets: Dict[str, bool] = {rel: bool(args) for rel, args in example.arguments_set[dmid].items()}
            assert len(self.relations) == len(arguments)
            for relation, argument in zip(self.relations, arguments):
                if not is_targets[relation]:
                    continue
                if self.use_knp_overt and relation in overt_dict:
                    # overt
                    prediction_dmid = overt_dict[relation]
                elif argument in self.index_to_special:
                    # special
                    special_arg = self.index_to_special[argument]
                    if special_arg in self.exophors:  # exclude [NULL] and [NA]
                        rels.append(RelTag(relation, special_arg, None, None))
                    continue
                else:
                    # normal
                    prediction_dmid = example.tok_to_orig_index[argument]
                    if prediction_dmid is None:
                        # [SEP] or [CLS]
                        self.logger.warning("Choose [SEP] as an argument. Tentatively, change it to NULL.")
                        continue
                prediction_bp: BasePhrase = dmid2bp[prediction_dmid]
                rels.append(RelTag(relation, prediction_bp.core, prediction_bp.sid, prediction_bp.tid))

        return ''.join(rel.to_string() for rel in rels)

    def _add_pas_tag(self,
                     knp_lines: List[str],
                     document: Document,
                     ) -> List[str]:
        dtid2pas = {pas.dtid: pas for pas in document.pas_list()}
        dtid = 0
        output_knp_lines = []
        for line in knp_lines:
            if not line.startswith('+ '):
                output_knp_lines.append(line)
                continue
            if dtid in dtid2pas:
                pas_string = self._pas_string(dtid2pas[dtid], 'dummy:dummy', document)
                output_knp_lines.append(line + pas_string)
            else:
                output_knp_lines.append(line)

            dtid += 1

        return output_knp_lines

    def _pas_string(self,
                    pas: Pas,
                    cfid: str,
                    document: Document,
                    ) -> str:
        sid2index: Dict[str, int] = {sid: i for i, sid in enumerate(document.sid2sentence.keys())}
        dtype2caseflag = {'overt': 'C', 'dep': 'N', 'intra': 'O', 'inter': 'O', 'exo': 'E'}
        case_elements = []
        for case in self.cases + (['ノ'] * self.bridging):
            items = ['-'] * 6
            items[0] = case
            args = pas.arguments[case]
            if args:
                arg: BaseArgument = args[0]
                items[1] = dtype2caseflag[arg.dep_type]  # フラグ (C/N/O/D/E/U)
                items[2] = str(arg)  # 見出し
                if isinstance(arg, Argument):
                    items[3] = str(sid2index[pas.sid] - sid2index[arg.sid])  # N文前
                    items[4] = str(arg.tid)  # tag id
                    items[5] = str(document.get_entities(arg)[0].eid)  # Entity ID
                else:
                    assert isinstance(arg, SpecialArgument)
                    items[3] = str(-1)
                    items[4] = str(-1)
                    items[5] = str(arg.eid)  # Entity ID
            else:
                items[1] = 'U'
            case_elements.append('/'.join(items))
        return f"<述語項構造:{cfid}:{';'.join(case_elements)}>"


class RelTag(NamedTuple):
    type_: str
    target: str
    sid: Optional[str]
    tid: Optional[int]

    def to_string(self):
        string = f'<rel type="{self.type_}" target="{self.target}"'
        if self.sid is not None:
            string += f' sid="{self.sid}" id="{self.tid}"'
        string += '/>'
        return string
