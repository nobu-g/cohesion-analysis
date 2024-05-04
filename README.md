# Cohesion: A Japanese cohesion analyzer

## Description

This project provides a system to perform the following analyses in a multi-task manner.

- Verbal predicate-argument structure analysis
- Nominal predicate-argument structure analysis
- Bridging reference resolution
- Coreference resolution

The process is as follows.

1. Apply Juman++ and KNP to an input text and split the text into base phrases.
2. Extract target base phrases to analyze by referring to the features added by KNP such as `<用言>` and `<体言>`.
3. For each target base phrase, select its arguments (or antecedents).

For more information, please refer to [the original paper](#reference)

## Requirements

- Python 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).
- [Juman++](https://github.com/ku-nlp/jumanpp) 2.0.0-rc4
- [KNP](https://github.com/ku-nlp/knp) 5.0

## Getting started

- Create a virtual environment and install dependencies.
    ```shell
    $ poetry env use /path/to/python
    $ poetry install
    ```

- Log in to [wandb](https://wandb.ai/site).
    ```shell
    $ wandb login
    ```

## Quick Start

- Install Juman++/KNP or KWJA.

  - [Juman++](https://github.com/ku-nlp/jumanpp)/[KNP](https://github.com/ku-nlp/knp)
    ```shell
      docker pull kunlp/jumanpp-knp:latest
      echo 'docker run -i --rm --platform linux/amd64 kunlp/jumanpp-knp jumanpp' > /somewhere/in/your/path/jumanpp
      echo 'docker run -i --rm --platform linux/amd64 kunlp/jumanpp-knp knp' > /somewhere/in/your/path/knp
    ```

  - [KWJA](https://github.com/ku-nlp/kwja)
    ```shell
      pipx install kwja
    ```

- Download a pre-trained model.

```shell
$ wget https://lotus.kuee.kyoto-u.ac.jp/~ueda/dist/cohesion_analysis_model.tar.gz  # trained checkpoint
$ tar xvzf cohesion_analysis_model.tar.gz  # make sure that the extracted directory is located at the root directory of this project
$ ls cohesion_analysis_model
model_best.pth
```

- Predict the cohesion of a sentence.

```shell
poetry run python src/predict.py checkpoint="cohesion_analysis_model/model_best.pth" input_file=<(echo "太郎はパンを買って食べた。") [num_workers=0] [devices=1]
```

The output is in the KNP format, which looks like the following:

```
# S-ID:202210010000-0-0 kwja:1.0.2
* 2D
+ 5D <rel type="=" target="ツール" sid="202210011918-0-0" id="5"/><体言><NE:ARTIFACT:KWJA>
KWJA ＫWＪＡ KWJA 名詞 6 固有名詞 3 * 0 * 0 <基本句-主辞>
は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は" <代表表記:は/は>
* 2D
+ 2D <体言>
日本 にほん 日本 名詞 6 地名 4 * 0 * 0 "代表表記:日本/にほん 地名:国" <代表表記:日本/にほん><地名:国><基本
句-主辞>
+ 4D <体言><係:ノ格>
語 ご 語 名詞 6 普通名詞 1 * 0 * 0 "代表表記:語/ご 漢字読み:音 カテゴリ:抽象物" <代表表記:語/ご><漢字読み:
音><カテゴリ:抽象物><基本句-主辞>
の の の 助詞 9 接続助詞 3 * 0 * 0 "代表表記:の/の" <代表表記:の/の>
...
```

You can read a KNP format file with [rhoknp](https://github.com/ku-nlp/rhoknp).

```python
from rhoknp import Document
with open("analyzed.knp") as f:
    parsed_document = Document.from_knp(f.read())
```

For more details about KNP format, see [rhoknp documentation](https://rhoknp.readthedocs.io/en/latest/format/index.html#knp).


## Building a dataset

```shell
$ OUT_DIR=data/dataset [JOBS=4] ./scripts/build_dataset.sh
$ ls data/dataset
fuman/  kwdlc/  wac/
```

## Creating a `.env` file and set `DATA_DIR`.

```shell
echo 'DATA_DIR="data/dataset"' >> .env
```

## Training

```shell
poetry run python src/train.py -cn default devices=[0,1] max_batches_per_device=4
```

Here are commonly used options:

- `-cn`: Config name (default: `default`).
- `devices`: GPUs to use (default: `0`).
- `max_batches_per_device`: Maximum number of batches to process per device (default: `4`).
- `compile`: JIT-compile the model
  with [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) for faster training (
  default: `false`).
- `model_name_or_path`: Path to a pre-trained model or model identifier from
  the [Huggingface Hub](https://huggingface.co/models) (default: `ku-nlp/deberta-v2-large-japanese`).

For more options, see YAML config files under [configs](./configs).

## Testing

```shell
poetry run python src/test.py checkpoint=/path/to/trained/checkpoint eval_set=valid devices=[0,1]
```

## Debugging

```shell
poetry run python src/train.py -cn debug
```

If you are on a machine with MPS devices (e.g. Apple M1), specify `trainer=cpu.debug` to use CPU.

```shell
# For debugging word segmenter
poetry run python scripts/train.py -cn debug trainer=cpu.debug
```

If you are on a machine with GPUs, you can specify the GPUs to use with the `devices` option.

```shell
# For debugging word segmenter
poetry run python scripts/train.py -cn debug devices=[0]
```

## Environment Variables

- `COHESION_CACHE_DIR`: A directory where processed documents are cached. Default value is `/tmp/$USER/cohesion_cache`.
- `COHESION_OVERWRITE_CACHE`: If set, the data loader does not load cache even if it exists.
- `COHESION_DISABLE_CACHE`: If set, the data loader does not load or save cache.

## Dataset

- Kyoto University Web Document Leads Corpus ([KWDLC](https://github.com/ku-nlp/KWDLC))
- Kyoto University Text Corpus ([KyotoCorpus](https://github.com/ku-nlp/KyotoCorpus))
- [Annotated FKC Corpus](https://github.com/ku-nlp/AnnotatedFKCCorpus)
- [Wikipedia Annotated Corpus](https://github.com/ku-nlp/WikipediaAnnotatedCorpus)

## Reference

[BERT-based Cohesion Analysis of Japanese Texts](https://www.aclweb.org/anthology/2020.coling-main.114/) [Ueda+, COLING2020]

## Author

Nobuhiro Ueda <ueda **at** nlp.ist.i.kyoto-u.ac.jp>
