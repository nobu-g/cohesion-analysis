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

TBW

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
