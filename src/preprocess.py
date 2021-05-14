"""Preprocess dataset."""
import _pickle as cPickle
import argparse
import json
import logging
import shutil
import subprocess
import tempfile
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Dict, Optional

from kyoto_reader import KyotoReader
from pyknp import KNP, BList
from tqdm import tqdm
from transformers import BertTokenizer

from data_loader.dataset.commonsense_dataset import CommonsenseExample

logger = logging.getLogger(__file__)


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
        while end < len(sids) and cum[end + 1] - cum[0] <= max_subword_length:
            end += 1

        idx = 0
        while end < len(sids) + 1:
            start = 0
            # start を探索
            while cum[end] - cum[start] > max_subword_length:
                start += 1
                if start == end - 1:
                    break
            with output_dir.joinpath(f'{did}-{idx:02}.knp').open(mode='w') as fout:
                fout.write(''.join(sid2knp[sid] for sid in sids[start:end]))  # start から end まで書き出し
            idx += 1
            end += 1


def reparse_knp(knp_file: Path,
                output_dir: Path,
                knp: KNP,
                keep_dep: bool
                ) -> None:
    """係り受けなどを再付与"""
    blists: List[BList] = []
    with knp_file.open() as fin:
        buff = ''
        for line in fin:
            if line.startswith('+') or line.startswith('*'):
                if keep_dep is False:
                    buff += line[0] + '\n'  # ex) +
                else:
                    buff += ' '.join(line.split()[:2]) + '\n'  # ex) + 3D
            else:
                buff += line
            if line.strip() == 'EOS':
                blists.append(knp.reparse_knp_result(buff))
                buff = ''
    output_dir.joinpath(knp_file.name).write_text(''.join(blist.spec() for blist in blists))


def reparse(input_dir: Path,
            output_dir: Path,
            knp: KNP,
            bertknp: Optional[str] = None,
            n_jobs: int = 0,
            keep_dep: bool = False,
            ) -> None:
    if bertknp is None:
        args_iter = ((path, output_dir, knp, keep_dep) for path in input_dir.glob('*.knp'))
        if n_jobs > 0:
            with Pool(n_jobs) as pool:
                pool.starmap(reparse_knp, args_iter)
        else:
            for args in args_iter:
                reparse_knp(*args)
        return

    assert keep_dep is False, 'If you use BERTKNP, you cannot keep dependency labels.'

    buff = ''
    for knp_file in input_dir.glob('*.knp'):
        with knp_file.open() as fin:
            for line in fin:
                if line.startswith('+') or line.startswith('*'):
                    buff += line[0] + '\n'
                else:
                    buff += line

    out = subprocess.run(
        [
            bertknp,
            '-p',
            Path(bertknp).parents[1] / '.venv/bin/python',
            '-O',
            Path(__file__).parent.joinpath('bertknp_options.txt').resolve().__str__(),
            '-tab',
        ],
        input=buff, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf-8'
    )
    logger.warning(out.stderr)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_dir = Path(tmp_dir)
        tmp_dir.joinpath('tmp.knp').write_text(out.stdout)
        for did, knp_string in KyotoReader(tmp_dir / 'tmp.knp').did2knps.items():
            output_dir.joinpath(f'{did}.knp').write_text(knp_string)


def process(input_path: Path,
            output_path: Path,
            corpus: str,
            do_reparse: bool,
            n_jobs: int,
            bertknp: Optional[str],
            knp: KNP,
            keep_dep: bool = False,
            split: bool = False,
            max_subword_length: int = None,
            tokenizer: BertTokenizer = None,
            ) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir1, tempfile.TemporaryDirectory() as tmp_dir2:
        tmp_dir1, tmp_dir2 = Path(tmp_dir1), Path(tmp_dir2)
        if do_reparse is True:
            reparse(input_path, tmp_dir1, knp, bertknp=bertknp, n_jobs=n_jobs, keep_dep=keep_dep)
            input_path = tmp_dir1
        if split is True:
            # Because the length of the documents in KyotoCorpus is very long, split them into multiple documents
            # so that the tail sentence of each document has as much preceding sentences as possible.
            print('splitting corpus...')
            split_kc(input_path, tmp_dir2, max_subword_length, tokenizer)
            input_path = tmp_dir2

        output_path.mkdir(exist_ok=True)
        reader = KyotoReader(input_path, extract_nes=False, did_from_sid=(not split), n_jobs=n_jobs)
        for document in tqdm(reader.process_all_documents(), desc=corpus, total=len(reader)):
            with output_path.joinpath(document.doc_id + '.pkl').open(mode='wb') as f:
                cPickle.dump(document, f)

    return len(reader)


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
    parser.add_argument('--fuman', type=str, default=None,
                        help='path to directory where Fuman Corpus data exists')
    parser.add_argument('--commonsense', type=str, default=None,
                        help='path to directory where commonsense inference data exists')
    parser.add_argument('--out', type=(lambda p: Path(p)), required=True,
                        help='path to directory where dataset to be located')
    parser.add_argument('--max-seq-length', type=int, default=128,
                        help='The maximum total input sequence length after WordPiece tokenization. Sequences '
                             'longer than this will be truncated, and sequences shorter than this will be padded.')
    parser.add_argument('--exophors', '--exo', type=str, default='著者,読者,不特定:人,不特定:物',
                        help='exophor strings separated by ","')
    parser.add_argument('--bert-path', type=str, required=True,
                        help='path to pre-trained BERT model')
    parser.add_argument('--bert-name', type=str, required=True,
                        help='BERT model name')
    parser.add_argument('--jumanpp', type=str, default=shutil.which('jumanpp'),
                        help='path to jumanpp')
    parser.add_argument('--knp', type=str, default=shutil.which('knp'),
                        help='path to knp')
    parser.add_argument('--bertknp', type=str, default=None,
                        help='If specified, use BERTKNP instead of KNP for preprocessing.')
    parser.add_argument('--n-jobs', type=int, default=-1,
                        help='number of processes of multiprocessing (default: number of cores)')
    args = parser.parse_args()

    # make directories to save dataset
    args.out.mkdir(exist_ok=True)
    exophors = args.exophors.split(',')
    tokenizer = BertTokenizer.from_pretrained(args.bert_path, do_lower_case=False, tokenize_chinese_chars=False)

    knp = KNP(command=args.knp, jumancommand=args.jumanpp)
    knp_case = KNP(command=args.knp, option='-tab -case2', jumancommand=args.jumanpp)

    config_path: Path = args.out / 'config.json'
    if config_path.exists():
        with config_path.open() as f:
            config = json.load(f)
    else:
        config = {}
    config.update(
        {
            'max_seq_length': args.max_seq_length,
            'exophors': exophors,
            'vocab_size': tokenizer.vocab_size,
            'bert_name': args.bert_name,
            'bert_path': args.bert_path,
        }
    )
    if 'num_examples' not in config:
        config['num_examples'] = {}

    for corpus in ('kwdlc', 'kc', 'fuman'):
        if getattr(args, corpus) is None:
            continue
        in_dir = Path(getattr(args, corpus)).resolve()
        out_dir: Path = args.out / corpus
        out_dir.mkdir(exist_ok=True)
        kwargs = {
            'n_jobs': cpu_count() if args.n_jobs == -1 else args.n_jobs,
            'bertknp': args.bertknp,
            'knp': knp,
        }
        if corpus == 'kc':
            kwargs.update({
                'split': True,
                'max_subword_length': args.max_seq_length - len(exophors) - 4,  # [CLS], [SEP], [NULL], [NA]
                'tokenizer': tokenizer,
            })
        num_examples_train = process(in_dir / 'train', out_dir / 'train', corpus, do_reparse=False, **kwargs)
        num_examples_valid = process(in_dir / 'valid', out_dir / 'valid', corpus, do_reparse=True, **kwargs)
        num_examples_test = process(in_dir / 'test', out_dir / 'test', corpus, do_reparse=True, **kwargs)
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples'][corpus] = num_examples_dict

        # When each tag line of input is like "+ 3D\n", `knp -case` does not add <格解析> feature.
        # To add <格解析> feature, we need `-case2` option.
        kwargs.update({'knp': knp_case, 'bertknp': None})
        _ = process(in_dir / 'valid', out_dir / 'valid_oracle', corpus, do_reparse=True, keep_dep=True, **kwargs)
        _ = process(in_dir / 'test', out_dir / 'test_oracle', corpus, do_reparse=True, keep_dep=True, **kwargs)
        kwargs.update({'knp': knp, 'bertknp': args.bertknp})

        if corpus == 'kc':
            kwargs['split'] = False
        _ = process(in_dir / 'valid', out_dir / 'valid_gold', corpus, do_reparse=False, **kwargs)
        _ = process(in_dir / 'test', out_dir / 'test_gold', corpus, do_reparse=False, **kwargs)

    if args.commonsense is not None:
        in_dir = Path(args.commonsense).resolve()
        out_dir: Path = args.out / 'commonsense'
        out_dir.mkdir(exist_ok=True)
        num_examples_train = process_commonsense(in_dir / 'train.csv', out_dir / 'train.pkl')
        num_examples_valid = process_commonsense(in_dir / 'valid.csv', out_dir / 'valid.pkl')
        num_examples_test = process_commonsense(in_dir / 'test.csv', out_dir / 'test.pkl')
        num_examples_dict = {'train': num_examples_train, 'valid': num_examples_valid, 'test': num_examples_test}
        config['num_examples']['commonsense'] = num_examples_dict

    with config_path.open(mode='wt') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
