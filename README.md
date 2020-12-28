# cohesion-analysis

BERT based Japanese cohesion analyzer.

## Description

This project provides a system to perform the following analyses in a multi-task manner:
- Verbal Predicate argument structure Analysis (VPA)
- Nominal Predicate argument structure Analysis (NPA)
- Bridging Anaphora Resolution (BAR)
- Coreference Resolution (CR)

The process is as follows:
1. Apply Juman++ and KNP to an input text and split the text into base phrases
2. Extract predicates from base phrases by seeing whether the phrase has "<用言>" feature, which was tagged by KNP
3. Split each phrase into subwords using BPE
4. For each predicate subword, select its arguments using BERT

For more information, please refer to [the original paper](#reference)

## Demo

<https://lotus.kuee.kyoto-u.ac.jp/~ueda/demo/cohesion-analysis-demo/public/>

<img width="910" alt="demo-view" src="https://user-images.githubusercontent.com/25974220/103130065-7e43f900-46de-11eb-8d26-6226e498b6d0.png">

## Requirements

- Python 3.7.2+
- [Juman++](https://github.com/ku-nlp/jumanpp) 2.0.0-rc3
- [KNP](https://github.com/ku-nlp/knp) 4.2+

## Setup Environment

Use [poetry](https://github.com/python-poetry/poetry)

`poetry install`

## Quick Start

```zsh
$ wget https://lotus.kuee.kyoto-u.ac.jp/~ueda/dist/cohesion_analysis_model.tar.gz  # trained checkpoint
$ tar xvzf cohesion_analysis_model.tar.gz  # make sure that the extracted directory is located at the root directory of this project
$ MODEL=cohesion_analysis_model/model_best.pth

$ python src/predict.py \
--model $MODEL \
--input "太郎はパンを買って食べた。"
```

Result:

```text
太郎は──┐
  パンを┐│
    買って┤  太郎:ガ パン:ヲ :ニ :ガ２
    食べた。  太郎:ガ パン:ヲ :ニ :ガ２
```

Options:

- `--model, -m, -r`: a path to trained checkpoint
- `--device, -d`: GPU IDs separated by "," (if not specified, use CPU)
- `--input, -i`: input sentence or document separated by "。"
- `-tab`: output results in KNP tab format if specified

`predict.py` requires Juman++ and KNP for the analysis.
Make sure you have Juman++ and KNP installed before you run the above command.
In addition, I recommend you to create `src/analyzer/config.ini`
so that the system can find Juman++, KNP, and their configurations.
For details, see `src/analyzer/config.example.ini`

## Analyze a Large Number of Documents

Given raw sentences, first you need to apply KNP to them and create `.knp` files.
Then, run `predict.py` specifying the directory where created `.knp` files exist in the option `--knp-dir`.
When the analysis finishes, `.knp` files with `<rel >` and `<述語項構造:>` tags are exported to the directory
specified in `--export-dir` option.
To utilize this result, I recommend you to use [kyoto-reader](https://github.com/ku-nlp/kyoto-reader).

```zsh
python src/predict.py \
--model /path/to/trained/checkpoint \
--knp-dir /path/to/parsed/document/directory
--export-dir path/to/export/directory
```

Note: \
Each .knp file, which KNP created, contains a line indicating the sentence id of the following lines: `# S-ID:***`.
This project regard S-ID without its tail as the document id.
For example, the document id of a sentence whose S-ID is `w201106-0000060050-1` is `w201106-0000060050`.
Sentences that have the same document ids are analyzed as a single document.

## Training Your Model

### Preparing Corpus

```zsh
cd /somewhere
mkdir kwdlc kc
```

Download corpora:
- `git clone https://github.com/ku-nlp/KWDLC kwdlc/KWDLC`
- `git clone https://github.com/ku-nlp/KyotoCorpus kc/KyotoCorpus`
  - This repository contains only the annotations, not the body texts.
  - To obtain the body text, follow [the instructions of KyotoCorpus](https://github.com/ku-nlp/KyotoCorpus#conversion-to-the-complete-annotated-corpus)

Add features:

[kyoto-reader](https://github.com/ku-nlp/kyoto-reader), which this project depends on,
provides [some commands](https://kyoto-reader.readthedocs.io/en/latest/#corpus-preprocessor) to preprocess corpora.
Make sure you are in the virtual environment of this project when you run `configure` and `idsplit` commands.

```
$ git clone https://github.com/ku-nlp/JumanDIC
$ configure --corpus-dir /somewhere/kwdlc/KWDLC/knp \
--data-dir /somewhere/kwdlc \
--juman-dic-dir /somewhere/JumanDIC/dic
created Makefile at /somewhere/kwdlc
$ configure --corpus-dir /somewhere/kc/KyotoCorpus/knp \
--data-dir /somewhere/kc \
--juman-dic-dir /somewhere/JumanDIC/dic
created Makefile at /somewhere/kc
$ cd /somewhere/kwdlc && make -i
$ cd /somewhere/kc && make -i
$ idsplit --corpus-dir /somewhere/kwdlc/knp \
--output-dir /somewhere/kwdlc \
--train /somewhere/kwdlc/KWDLC/id/split_for_pas/train.id \
--valid /somewhere/kwdlc/KWDLC/id/split_for_pas/dev.id \
--test /somewhere/kwdlc/KWDLC/id/split_for_pas/test.id
$ idsplit --corpus-dir /somewhere/kc/knp \
--output-dir /somewhere/kc \
--train /somewhere/kc/KyotoCorpus/id/split_for_pas/train.full.id \
--valid /somewhere/kc/KyotoCorpus/id/split_for_pas/dev.full.id \
--test /somewhere/kc/KyotoCorpus/id/split_for_pas/test.full.id
```

### Preprocessing Documents

After preparing corpora, you need to load and pickle them.

```zsh
python src/preprocess.py \
--kwdlc /somewhere/kwdlc \
--kc /somewhere/kc \
--out /somewhere/dataset \
--bert-name nict
--bert-path /somewhere/NICT_BERT-base_JapaneseWikipedia_32K_BPE
```

Don't care if many "sentence not found" messages are shown when processing KyotoCorpus.
It is a natural result of splitting kc documents.

### Configuring Settings

Before starting model training, prepare the configuration files.
The resultant files will be located at `config/`.

```zsh
python src/configure.py \
-c /path/to/config/directory \
-d /somewhere/dataset \
-e <num-epochs> \
-b <batch-size> \
--model <model-name> \
--corpus all
```

example:

```zsh
python src/configure.py -c config -d data/dataset -e 4 8 -b 8 --model CAModel --corpus all
```

### Training Models

Launch the trainer with a configuration.

```zsh
python src/train.py \
-c /path/to/config/file \
-d <gpu-ids>
```

Example:

```zsh
python src/train.py -c config/CAModel-all-4e-nict-cz-vpa.json -d 0,1
```

### Testing Models

```zsh
python src/test.py \
-r /path/to/trained/model \
-d <gpu-ids>
```

If you specify a config file besides the trained model, the setting will be overwritten. \

You can perform an ensemble test as well.
In this case, `test.py` gather all files named `model_best.pth` under the directory specified in `--ens` option.

```zsh
python src/test.py \
--ens /path/to/model/set/directory \
-d <gpu-ids>
```

### Scoring From System Output

```zsh
python src/scorer.py \
--prediction-dir /path/to/system/output/directory \
--gold-dir /path/to/gold/directory \
```

## Perform Training Process with Make

You can also perform training and testing using `make` command. \
Here is an example of training your own model 5 times with different random seeds:

```zsh
make train GPUS=<gpu-ids> CONFIG=/path/to/config/file TRAIN_NUM=5
```

Testing command is as follows (outputs confidence interval):

```zsh
make test GPUS=<gpu-ids> RESULT=/path/to/result/dir
```

This command executes two commands above all at once.

```zsh
make all GPUS=<gpu-ids> CONFIG=/path/to/config/file TRAIN_NUM=5
```

Ensemble test:

```zsh
make test-ens GPUS=<gpu-ids> RESULT=/path/to/result/dir
```

## Environment Variables

- `BPA_CACHE_DIR`: A directory where processed documents are cached. Default is `/data/$USER/bpa_cache`.
- `BPA_OVERWRITE_CACHE`: If set, cohesion-analysis doesn't load cache even if it exists.
- `BPA_DISABLE_CACHE`: If set, cohesion-analysis doesn't load or save cache.

## Dataset

- Kyoto University Web Document Leads Corpus ([KWDLC](https://github.com/ku-nlp/KWDLC))
- Kyoto University Text Corpus ([KyotoCorpus](https://github.com/ku-nlp/KyotoCorpus))

## Reference

[BERT-based Cohesion Analysis of Japanese Texts](https://www.aclweb.org/anthology/2020.coling-main.114/) [Ueda+, COLING2020]

## Licence

- This project: MIT
- [NICT BERT](https://alaginrc.nict.go.jp/nict-bert/index.html) (included in the trained model): [Creative Commons license 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/deed.ja)

## Author

Nobuhiro Ueda <ueda **at** nlp.ist.i.kyoto-u.ac.jp>
