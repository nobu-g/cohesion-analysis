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

## Demo

<https://lotus.kuee.kyoto-u.ac.jp/cohesion-analysis/pub/>

<img width="1273" alt="demo-view" src="https://user-images.githubusercontent.com/25974220/207257969-383c2db4-e28e-447f-af58-12f3cd33ffda.png">

## Requirements

- Python 3.9+
- Dependencies: See [pyproject.toml](./pyproject.toml).
- [Juman++](https://github.com/ku-nlp/jumanpp) 2.0.0-rc4 (optional)
- [KNP](https://github.com/ku-nlp/knp) 5.0 (optional)
- [KWJA](https://github.com/ku-nlp/kwja) 2.3.0 (optional)

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

- Download pre-trained models.

```shell
$ wget https://lotus.kuee.kyoto-u.ac.jp/~ueda/dist/cohesion_analysis_v2/model_base.bin  # trained checkpoint (base)
$ wget https://lotus.kuee.kyoto-u.ac.jp/~ueda/dist/cohesion_analysis_v2/model_large.bin  # trained checkpoint (large)
$ ls model_*.bin
model_base.bin  model_large.bin
```

- Run prediction.

```shell
$ poetry run python src/predict.py checkpoint=model_large.bin input_file=<(echo "太郎はパンを買って食べた。") [devices=1] > analyzed.knp; rhoknp show -r analyzed.knp
# S-ID:0-1 KNP:5.0-25425d33 DATE:2024/01/01 SCORE:59.00000
太郎は─────┐
  パンを─┐ │
    買って─┤  ガ:太郎 ヲ:パン
    食べた。  ガ:太郎 ヲ:パン

```

The output of `predict.py` is in the KNP format, which looks like the following:

```
# S-ID:0-1 KNP:5.0-25425d33 DATE:2024/05/05 SCORE:59.00000
* 3D <文頭><人名><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><正規化代表表記:太郎/たろう><主辞代表表記:太郎/たろう>
+ 3D <文頭><人名><ハ><助詞><体言><係:未格><提題><区切:3-5><主題表現><格要素><連用要素><名詞項候補><先行詞候補><SM-人><SM-主体><正規化代表表記:太郎/たろう><主辞代表表記:太郎/たろう><bridging対象><coreference対象>
太郎 たろう 太郎 名詞 6 人名 5 * 0 * 0 "代表表記:太郎/たろう 人名:日本:名:45:0.00106" <代表表記:太郎/たろう><人名:日本:名:45:0.00106><正規化代表表記:太郎/たろう><漢字><かな漢字><名詞相当語><文頭><自立><内容語><タグ単位始><文節始><固有キー><文節主辞>
は は は 助詞 9 副助詞 2 * 0 * 0 "代表表記:は/は" <代表表記:は/は><正規化代表表記:は/は><かな漢字><ひらがな><付属>
* 2D <BGH:パン/ぱん><ヲ><助詞><体言><係:ヲ格><区切:0-0><格要素><連用要素><正規化代表表記:パン/ぱん><主辞代表表記:パン/ぱん>
+ 2D <BGH:パン/ぱん><ヲ><助詞><体言><係:ヲ格><区切:0-0><格要素><連用要素><名詞項候補><先行詞候補><正規化代表表記:パン/ぱん><主辞代表表記:パン/ぱん><bridging対象><coreference対象>
パン ぱん パン 名詞 6 普通名詞 1 * 0 * 0 "代表表記:パン/ぱん ドメイン:料理・食事 カテゴリ:人工物-食べ物" <代表表記:パン/ぱん><ドメイン:料理・食事><カテゴリ:人工物-食べ物><正規化代表表記:パン/ぱん><記英数カ><カタカナ><名詞相当語><自立><内容語><タグ単位始><文節始><固有キー><文節主辞>
を を を 助詞 9 格助詞 1 * 0 * 0 "代表表記:を/を" <代表表記:を/を><正規化代表表記:を/を><かな漢字><ひらがな><付属>
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

- Set up the [J-CRe3](https://github.com/riken-grp/J-CRe3) dataset.

- Create a `.env` file and set `JCRE3_DATASET_DIR`.

```shell
echo 'JCRE3_DATASET_DIR="/path/to/J-CRe3/recording"' >> .env
```

- Preprocess datasets.

```shell
$ OUT_DIR=data/dataset [JOBS=4] ./scripts/build_dataset.sh
$ ls data/dataset
fuman/  kwdlc/  wac/
```

- Set `DATA_DIR`.

```shell
echo 'DATA_DIR="data/dataset"' >> .env
```

## Training

```shell
poetry run python src/train.py -cn default datamodule=all_wo_kc devices=[0,1] max_batches_per_device=4
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
poetry run python scripts/train.py -cn debug trainer=cpu.debug
```

If you are on a machine with GPUs, you can specify the GPUs to use with the `devices` option.

```shell
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
