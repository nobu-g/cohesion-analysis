import re
import os
import socket
from typing import Dict, Tuple, Union, Optional
import configparser
from pathlib import Path
from datetime import datetime
import shutil
from logging import Logger

import jaconv
from pyknp import Juman, KNP
from transformers import BertConfig
from textformatting import ssplit

import model.model as module_arch
import data_loader.dataset as module_dataset
import data_loader.data_loaders as module_loader
from data_loader.dataset import PASDataset
from utils.parse_config import ConfigParser
from test import Inference


class Analyzer:
    """Perform PAS analysis given a sentence."""

    def __init__(self, config: ConfigParser, logger: Logger, remote_knp: bool = False):
        cfg = configparser.ConfigParser()
        here = Path(__file__).parent
        cfg.read(here / 'config.ini')
        if 'default' not in cfg:
            logger.warning('Analyzer config not found. Instead, use default values.')
            cfg['default'] = {}
        section = cfg['default']
        self.juman = section.get('juman_command', shutil.which('jumanpp'))
        self.knp = section.get('knp_command', shutil.which('knp'))
        self.knp_host = section.get('knp_host')
        self.knp_port = section.getint('knp_port')
        self.juman_option = section.get('juman_option', '-s 1')
        self.knp_dpnd_option = section.get('knp_dpnd_option', '-tab -disable-segmentation-modification -dpnd-fast')
        self.knp_case_option = section.get('knp_case_option', '-tab -disable-segmentation-modification -case2')
        self.pos_map, self.pos_map_inv = self._read_pos_list(section.get('pos_list', here / 'pos.list'))
        self.config = config
        self.logger = logger
        self.remote_knp = remote_knp
        msg = 'You enabled remote_knp option, but no knp_host or knp_port are specified in the default section of ' \
              'src/analyzer/config.ini'
        assert not remote_knp or (self.knp_host and self.knp_port), msg

        os.environ['BPA_DISABLE_CACHE'] = '1'

        dataset_args = self.config['test_kwdlc_dataset']['args']
        bert_config = BertConfig.from_pretrained(dataset_args['dataset_config']['bert_path'])
        coreference = dataset_args['coreference']
        exophors = dataset_args['exophors']
        expanded_vocab_size = bert_config.vocab_size + len(exophors) + 1 + int(coreference)

        # build model architecture
        model = self.config.init_obj('arch', module_arch, vocab_size=expanded_vocab_size)
        self.logger.info(model)

        self.inference = Inference(self.config, model, logger=self.logger)

    def analyze(self, source: Union[Path, str], knp_dir: Optional[str] = None) -> Tuple[list, PASDataset]:
        if isinstance(source, Path):
            self.logger.info(f'read knp files from {source}')
            save_dir = source
        else:
            save_dir = Path(knp_dir) if knp_dir is not None else Path('log') / datetime.now().strftime(r'%m%d_%H%M%S')
            save_dir.mkdir(exist_ok=True, parents=True)
            sents = [self.sanitize_string(sent) for sent in ssplit(source)]
            self.logger.info('input: ' + ''.join(sents))
            knp_out = ''
            for i, sent in enumerate(sents):
                knp_out_ = self._apply_knp(sent)
                knp_out_ = knp_out_.replace('S-ID:1', f'S-ID:{i + 1}')
                knp_out += knp_out_
            with save_dir.joinpath(f'doc.knp').open(mode='wt') as f:
                f.write(knp_out)

        return self._analysis(save_dir)

    def analyze_from_knp(self, knp_out: str, knp_dir: Optional[str] = None) -> Tuple[list, PASDataset]:
        save_dir = Path(knp_dir) if knp_dir is not None else Path('log') / datetime.now().strftime(r'%m%d_%H%M%S')
        save_dir.mkdir(exist_ok=True)
        with save_dir.joinpath('doc.knp').open(mode='wt') as f:
            f.write(knp_out)
        return self._analysis(save_dir)

    def _analysis(self, path: Path) -> Tuple[list, PASDataset]:
        self.config['test_kwdlc_dataset']['args']['path'] = str(path)
        dataset = self.config.init_obj(f'test_kwdlc_dataset', module_dataset)
        data_loader = self.config.init_obj(f'test_data_loader', module_loader, dataset)

        _, *predictions = self.inference(data_loader)

        prediction = predictions[0]  # (N, seq, case)

        return prediction.tolist(), dataset

    def _apply_jumanpp(self, inp: str) -> Tuple[str, str]:
        jumanpp = Juman(command=self.juman, option=self.juman_option)
        jumanpp_result = jumanpp.analysis(inp)
        jumanpp_out = jumanpp_result.spec() + 'EOS\n'
        jumanpp_conll_out = self._jumanpp2conll_one_sentence(jumanpp_out) + 'EOS\n'
        return jumanpp_out, jumanpp_conll_out

    def _apply_knp(self, sent: str) -> str:
        self.logger.info(f'parse sentence: {sent}')
        knp = KNP(command=self.knp, jumancommand=self.juman, option=self.knp_dpnd_option)
        knp_result = knp.parse(sent)

        if self.remote_knp is True:
            _, jumanpp_conll_out = self._apply_jumanpp(sent)
            clientsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.logger.info(f'connect to {self.knp_host}:{self.knp_port}')
            clientsock.connect((self.knp_host, self.knp_port))
            clientsock.sendall(jumanpp_conll_out.encode('utf-8'))

            buf = []
            while True:
                data = clientsock.recv(8192)
                data_utf8 = data.decode('utf-8')
                buf.append(data_utf8)
                if data_utf8.endswith('EOS\n'):
                    break
            clientsock.close()
            conllu_out = ''.join(buf)
            self.logger.info(f'received {len(conllu_out)} chars from remote KNP')

            # modify KNP result by conllu result of remote KNP
            head_ids, dpnd_types = self._read_conllu_from_buf(conllu_out)
            self._modify_knp(knp_result, head_ids, dpnd_types)

        # add predicate-argument structures by KNP
        knp = KNP(command=self.knp, jumancommand=self.juman, option=self.knp_case_option)
        knp_result_new = knp.parse_juman_result(knp_result.spec())
        return knp_result_new.spec()

    def _jumanpp2conll_one_sentence(self, jumanpp_out: str):

        output_lines = []
        prev_id = 0
        for line in jumanpp_out.splitlines():
            result = []
            if line.startswith('EOS'):
                break
            items = line.strip().split('\t')
            if prev_id == items[1]:
                continue  # skip the same id
            else:
                result.append(str(items[1]))
                prev_id = items[1]
            result.append(items[5])  # midasi
            result.append(items[8])  # genkei
            conll_pos = self.get_pos(items[9], items[11])  # hinsi, bunrui
            result.append(conll_pos)
            result.append(conll_pos)
            result.append('_')
            if len(items) > 19:
                result.append(items[18])  # head
                result.append(items[19])  # dpnd_type
            else:
                result.append('0')  # head
                result.append('D')  # dpnd_type (dummy)
            result.append('_')
            result.append('_')
            output_lines.append('\t'.join(result) + '\n')
        return ''.join(output_lines)

    def get_pos(self, pos: str, subpos: str) -> str:
        if subpos == '*':
            key = pos
        elif pos == '未定義語':
            key = '未定義語-その他'
        else:
            key = f'{pos}-{subpos}'

        if key in self.pos_map:
            return self.pos_map[key]
        else:
            assert f'Found unknown POS: {pos}-{subpos}'

    @staticmethod
    def _read_pos_list(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        pos_map, pos_map_inv = {}, {}
        with open(path, encoding='utf-8') as f:
            for line in f:
                pos, pos_code = line.strip().split('\t')
                pos_map[pos] = pos_code
                pos_map_inv[pos_code] = pos
        return pos_map, pos_map_inv

    @staticmethod
    def _read_conllu_from_buf(conllu_out: str) -> Tuple[Dict[int, int], Dict[int, str]]:
        head_ids, dpnd_types = {}, {}
        for line in conllu_out.splitlines():
            if line == '\n' or line.startswith('EOS'):
                break
            items = line.strip().split('\t')
            _id = int(items[0]) - 1
            dpnd_id = int(items[6]) - 1
            head_ids[_id] = dpnd_id
            dpnd_types[_id] = items[7]
        return head_ids, dpnd_types

    @staticmethod
    def _modify_knp(knp_result, head_ids, dpnd_types):

        def modify(tags_, head_ids_, dpnd_types_, mode_: str) -> None:
            mrph_id2tag = {}
            for tag in tags_:
                for mrph in tag.mrph_list():
                    mrph_id2tag[mrph.mrph_id] = tag

            for tag in tags_:
                # この基本句内の形態素IDリスト
                in_tag_mrph_ids = {}
                last_mrph_id_in_tag = -1
                for mrph in tag.mrph_list():
                    in_tag_mrph_ids[mrph.mrph_id] = 1
                    if last_mrph_id_in_tag < mrph.mrph_id:
                        last_mrph_id_in_tag = mrph.mrph_id

                for mrph_id in list(in_tag_mrph_ids.keys()):
                    # 形態素係り先ID
                    mrph_head_id = head_ids_[mrph_id]
                    # 形態素係り先がROOTの場合は何もしない
                    if mrph_head_id == -1:
                        break
                    # 形態素係り先が基本句外に係る場合: 既存の係り先と異なるかチェック
                    if mrph_head_id > last_mrph_id_in_tag:
                        new_parent_tag = mrph_id2tag[mrph_head_id]
                        if mode_ == 'tag':
                            new_parent_id = new_parent_tag.tag_id
                            old_parent_id = tag.parent.tag_id
                        else:
                            new_parent_id = new_parent_tag.bnst_id
                            old_parent_id = tag.parent.bnst_id
                        # 係りタイプの更新
                        if dpnd_types_[mrph_id] != tag.dpndtype:
                            tag.dpndtype = dpnd_types_[mrph_id]
                        # 係り先の更新
                        if new_parent_id != old_parent_id:
                            # 形態素係り先IDを基本句IDに変換しparentを設定
                            tag.parent_id = new_parent_id
                            tag.parent = new_parent_tag
                            # children要更新?
                            break

        tags = knp_result.tag_list()
        bnsts = knp_result.bnst_list()

        # modify tag dependencies
        modify(tags, head_ids, dpnd_types, 'tag')

        # modify bnst dependencies
        modify(bnsts, head_ids, dpnd_types, 'bunsetsu')

    @staticmethod
    def sanitize_string(string: str):
        string = ''.join(string.split())  # remove space character
        string = jaconv.h2z(string, digit=True, ascii=True)
        return string
