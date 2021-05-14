import io
import sys
import argparse
from logging import getLogger
from pathlib import Path
from typing import List, Optional, TextIO

from kyoto_reader import Document, Argument, BaseArgument
from pyknp import BList

from prediction.prediction_writer import PredictionKNPWriter
from analyzer import Analyzer
from utils.parse_config import ConfigParser
from utils.util import is_pas_target, is_bridging_target, is_coreference_target

logger = getLogger(__name__)


def draw_tree(document: Document,
              sid: str,
              cases: List[str],
              bridging: bool = False,
              coreference: bool = False,
              fh: Optional[TextIO] = None,
              html: bool = False,
              ) -> None:
    """sid で指定された文の述語項構造・共参照関係をツリー形式で fh に書き出す

    Args:
        document (Document): sid が含まれる文書
        sid (str): 出力対象の文ID
        cases (List[str]): 表示対象の格
        bridging (bool): 橋渡し照応関係も表示するかどうか
        coreference (bool): 共参照関係も表示するかどうか
        fh (Optional[TextIO]): 出力ストリーム
        html (bool): html 形式で出力するかどうか

    """
    blist: BList = document.sid2sentence[sid].blist
    with io.StringIO() as string:
        blist.draw_tag_tree(fh=string, show_pos=False)
        tree_strings = string.getvalue().rstrip('\n').split('\n')
    assert len(tree_strings) == len(blist.tag_list())
    all_targets = [m.core for m in document.mentions.values()]
    tid2mention = {mention.tid: mention for mention in document.mentions.values() if mention.sid == sid}
    for bp in document[sid].bps:
        tree_strings[bp.tid] += '  '
        if is_pas_target(bp, verbal=True, nominal=True):
            arguments = document.get_arguments(bp)
            for case in cases:
                args: List[BaseArgument] = arguments.get(case, [])
                targets = set()
                for arg in args:
                    target = str(arg)
                    if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                        target += str(arg.dtid)
                    targets.add(target)
                if html:
                    color = 'black' if targets else 'gray'
                    tree_strings[bp.tid] += f'<font color="{color}">{case}:{",".join(targets)}</font> '
                else:
                    tree_strings[bp.tid] += f'{case}:{",".join(targets)} '
        if bridging and is_bridging_target(bp):
            args = document.get_arguments(bp).get('ノ', [])
            targets = set()
            for arg in args:
                target = str(arg)
                if all_targets.count(str(arg)) > 1 and isinstance(arg, Argument):
                    target += str(arg.dtid)
                targets.add(target)
            if html:
                color = 'black' if targets else 'gray'
                tree_strings[bp.tid] += f'<font color="{color}">ノ:{",".join(targets)}</font> '
            else:
                tree_strings[bp.tid] += f'ノ:{",".join(targets)} '
        if coreference and is_coreference_target(bp):
            if bp.tid in tid2mention:
                src_mention = tid2mention[bp.tid]
                tgt_mentions = [tgt for tgt in document.get_siblings(src_mention) if tgt.dtid < src_mention.dtid]
                targets = set()
                for tgt_mention in tgt_mentions:
                    target = tgt_mention.core
                    if all_targets.count(tgt_mention.core) > 1:
                        target += str(tgt_mention.dtid)
                    targets.add(target)
                for eid in src_mention.eids:
                    entity = document.entities[eid]
                    if entity.is_special:
                        targets.add(entity.exophor)
            else:
                targets = set()
            if html:
                color = 'black' if targets else 'gray'
                tree_strings[bp.tid] += f'<font color="{color}">＝:{",".join(targets)}</font>'
            else:
                tree_strings[bp.tid] += f'＝:{",".join(targets)}'
    print('\n'.join(tree_strings), file=fh)


def main(config, args):
    analyzer = Analyzer(config, remote_knp=args.remote_knp)

    if args.input is not None:
        source = args.input
    elif args.knp_dir is not None:
        source = Path(args.knp_dir)
    else:
        source = ''.join(sys.stdin.readlines())

    arguments_set, dataset = analyzer.analyze(source)

    prediction_writer = PredictionKNPWriter(dataset, logger)
    if args.export_dir is not None:
        destination = Path(args.export_dir)
    elif args.tab is True:
        destination = sys.stdout
    else:
        destination = None
    documents_pred: List[Document] = prediction_writer.write(arguments_set, destination,
                                                             skip_untagged=args.skip_untagged,
                                                             add_pas_tag=(not args.rel_only))
    if args.tab is False:
        for document_pred in documents_pred:
            for sid in document_pred.sid2sentence.keys():
                draw_tree(document_pred, sid, dataset.target_cases, dataset.bridging, dataset.coreference, sys.stdout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-r', '--resume', '-m', '--model', default=None, type=str,
                        help='path to trained checkpoint')
    parser.add_argument('--ens', '--ensemble', default=None, type=str,
                        help='path to directory where checkpoints to ensemble exist')
    parser.add_argument('-d', '--device', default='', type=str,
                        help='indices of GPUs to enable (default: all)')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--input', default=None, type=str,
                        help='sentences to analysis (if not specified, use stdin)')
    parser.add_argument('--knp-dir', default=None, type=str,
                        help='path to the directory where parsed documents are saved'
                             'in case parsed files exist here, KNP is skipped')
    parser.add_argument('--export-dir', default=None, type=str,
                        help='directory where analysis result is exported')
    parser.add_argument('-tab', '--tab', action='store_true', default=False,
                        help='whether to output details')
    parser.add_argument('--remote-knp', action='store_true', default=False,
                        help='Use KNP running on remote host. '
                             'Make sure you specify host address and port in analyzer/config.ini')
    parser.add_argument('--skip-untagged', action='store_true', default=False,
                        help='If set, do not export documents which failed to be analyzed')
    parser.add_argument('--rel-only', action='store_true', default=False,
                        help='If set, do not add <述語項構造> tag besides <rel> tag to system output')
    parsed_args = parser.parse_args()
    main(ConfigParser.from_args(parsed_args, run_id=''), parsed_args)
