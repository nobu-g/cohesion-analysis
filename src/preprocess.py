"""Preprocess dataset."""
import argparse
import json
import tempfile
from pathlib import Path
import _pickle as cPickle
from typing import List, Dict
from collections import defaultdict

from tqdm import tqdm
from pyknp import BList
from kyoto_reader import KyotoReader
from transformers import BertTokenizer

from data_loader.dataset.commonsense_dataset import CommonsenseExample

NUM_SPECIAL_TOKENS = 5


def process_kwdlc(input_path: Path, output_path: Path, cases: List[str], corefs: List[str]) -> int:
    output_path.mkdir(exist_ok=True)
    reader = KyotoReader(input_path, cases, corefs, extract_nes=False)
    for document in tqdm(reader.process_all_documents(), desc='kwdlc', total=len(reader.did2source)):
        with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
            cPickle.dump(document, f)
    return len(reader.did2source)


def split_kc(input_dir: Path, output_dir: Path, max_subword_length: int, tokenizer: BertTokenizer):
    """
    各文書を，tokenize したあとの長さが max_subword_length 以下になるように複数の文書に分割する．
    1文に分割しても max_subword_length を超えるような長い文はそのまま出力する
    """
    did2sids: Dict[str, List[str]] = defaultdict(list)
    did2cumlens: Dict[str, List[int]] = {}
    sid2knp: Dict[str, str] = {}

    for knp_file in input_dir.glob('*.knp'):
        with knp_file.open() as fin:
            did = knp_file.stem
            did2cumlens[did] = [0]
            buff = ''
            for line in fin:
                buff += line
                if line.strip() == 'EOS':
                    blist = BList(buff)
                    did2sids[did].append(blist.sid)
                    did2cumlens[did].append(
                        did2cumlens[did][-1] + len(tokenizer.tokenize(' '.join(m.midasi for m in blist.mrph_list())))
                    )
                    sid2knp[blist.sid] = buff
                    buff = ''

    for did, sids in did2sids.items():
        cum: List[int] = did2cumlens[did]
        end = 1
        # end を探索
        while end < len(sids) and cum[end+1] - cum[0] <= max_subword_length:
            end += 1

        idx = 0
        while end < len(sids) + 1:
            start = 0
            # start を探索
            while cum[end] - cum[start] > max_subword_length:
                start += 1
                if start == end - 1:
                    break
            with output_dir.joinpath(f'{did}-{idx:02}.knp').open('wt') as fout:
                fout.write(''.join(sid2knp[sid] for sid in sids[start:end]))  # start から end まで書き出し
            idx += 1
            end += 1


def process_kc(input_path: Path, output_path: Path, config: dict, tokenizer: BertTokenizer, split: bool = False) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        if split:
            tmp_dir = Path(tmp_dir)
            # 京大コーパスは1文書が長いのでできるだけ多くの context を含むように複数文書に分割する
            max_subword_length = config['max_seq_length'] - 2 - NUM_SPECIAL_TOKENS
            print('splitting kc...')
            split_kc(input_path, tmp_dir, max_subword_length, tokenizer)
            input_path = tmp_dir

        output_path.mkdir(exist_ok=True)
        reader = KyotoReader(input_path, config['target_cases'], config['target_corefs'], extract_nes=False)
        for document in tqdm(reader.process_all_documents(), desc='kc', total=len(reader.did2source)):
            with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
                cPickle.dump(document, f)

    return len(reader.did2source)


def process_commonsense(input_path: Path, output_path: Path) -> int:
    examples = []
    print('processing commonsense...')
    with input_path.open() as f:
        for line in f:
            label, string = line.strip().split(',')
            former_string, latter_string = string.split('@')
            examples.append(CommonsenseExample(former_string, latter_string, bool(int(label))))
    with output_path.open(mode='wb') as f:
        cPickle.dump(examples, f)
    return len(examples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kwdlc', type=str, default=None,
                        help='path to directory where KWDLC data exists')
    parser.add_argument('--kc', type=str, default=None,
                        help='path to directory where Kyoto Corpus data exists')
    parser.add_argument('--commonsense', type=str, default=None,
                        help='path to directory where commonsense inference data exists')
    parser.add_argument('--out', type=(lambda p: Path(p)), required=True,
                        help='path to directory where dataset to be located')
    parser.add_argument('--coref-string', type=str, default='=,=構,=≒,=構≒',
                        help='Coreference strings. Separate by ","')
    parser.add_argument('--case-string', type=str, default='ガ,ヲ,ニ,ガ２,ノ,ノ？,判ガ',
                        help='Case strings. Separate by ","')
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--bert-path', type=str, required=True,
                        help='path to pre-trained BERT model')
    parser.add_argument('--bert-name', type=str, required=True,
                        help='BERT model name')
    args = parser.parse_args()

    # make directories to save dataset
    args.out.mkdir(exist_ok=True)

    target_cases: List[str] = args.case_string.split(',')
    target_corefs: List[str] = args.coref_string.split(',')
    config_path: Path = args.out / 'config.json'
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    else:
        config = {}
    config.update(
        {
            'target_cases': target_cases,
            'target_corefs': target_corefs,
            # 'num_examples': {},
            'max_seq_length': args.max_seq_length,
            'bert_name': args.bert_name,
            'bert_path': args.bert_path,
        }
    )
    if 'num_examples' not in config:
        config['num_examples'] = {}

    if args.kwdlc is not None:
        input_dir = Path(args.kwdlc).resolve()
        output_dir: Path = args.out / 'kwdlc'
        output_dir.mkdir(exist_ok=True)
        num_examples_train = process_kwdlc(input_dir / 'train', output_dir / 'train', target_cases, target_corefs)
        num_examples_valid = process_kwdlc(input_dir / 'valid', output_dir / 'valid', target_cases, target_corefs)
        num_examples_test = process_kwdlc(input_dir / 'test', output_dir / 'test', target_cases, target_corefs)
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['kwdlc'] = num_examples_dict

    if args.kc is not None:
        input_dir = Path(args.kc).resolve()
        output_dir: Path = args.out / 'kc_split'
        output_dir.mkdir(exist_ok=True)
        tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=False, tokenize_chinese_chars=False)
        num_examples_train = process_kc(input_dir / 'train', output_dir / 'train', config, tokenizer, split=True)
        num_examples_valid = process_kc(input_dir / 'valid', output_dir / 'valid', config, tokenizer, split=True)
        num_examples_test = process_kc(input_dir / 'test', output_dir / 'test', config, tokenizer, split=True)
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['kc'] = num_examples_dict

        output_dir: Path = args.out / 'kc'
        output_dir.mkdir(exist_ok=True)
        _ = process_kc(input_dir / 'valid', output_dir / 'valid', config, tokenizer, split=False)
        _ = process_kc(input_dir / 'test', output_dir / 'test', config, tokenizer, split=False)

    if args.commonsense is not None:
        input_dir = Path(args.commonsense).resolve()
        output_dir: Path = args.out / 'commonsense'
        output_dir.mkdir(exist_ok=True)
        num_examples_train = process_commonsense(input_dir / 'train.csv', output_dir / 'train.pkl')
        num_examples_valid = process_commonsense(input_dir / 'valid.csv', output_dir / 'valid.pkl')
        num_examples_test = process_commonsense(input_dir / 'test.csv', output_dir / 'test.pkl')
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['commonsense'] = num_examples_dict

    with config_path.open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
