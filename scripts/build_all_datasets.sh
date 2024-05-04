#!/usr/bin/env bash

set -euCo pipefail

readonly JOBS="${JOBS:-"1"}"
readonly OUT_DIR="${OUT_DIR:-""}"

usage() {
  cat << _EOT_
Usage:
  OUT_DIR=data/dataset [JOBS=4] $0

Options:
  OUT_DIR      path to output directory
  JOBS         number of jobs (default=1)
_EOT_
  exit 1
}

if [[ $# -gt 0 ]]; then
  usage
fi

if [[ -z "${OUT_DIR}" ]]; then
  echo "missing required variable -- OUT_DIR" >&2
  usage
fi

WORK_DIR="$(mktemp -d)"
readonly WORK_DIR

mkdir -p "${WORK_DIR}" "${OUT_DIR}"/{kc,kwdlc,fuman,wac,jcre3}
git clone --depth 1 git@github.com:ku-nlp/KyotoCorpusFull.git "${WORK_DIR}/KyotoCorpus"
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "${WORK_DIR}/KWDLC"
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "${WORK_DIR}/AnnotatedFKCCorpus"
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "${WORK_DIR}/WikipediaAnnotatedCorpus"
git clone --depth 1 git@github.com:nobu-g/multimodal-annotation-data.git "${WORK_DIR}/multimodal-annotation-data"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}"/KyotoCorpus/knp "${OUT_DIR}/kc" \
  --id "${WORK_DIR}/KyotoCorpus/id/full" \
  -j "${JOBS}"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/KWDLC/knp" "${OUT_DIR}/kwdlc" \
  --id "${WORK_DIR}/KWDLC/id/split_for_pas" \
  -j "${JOBS}" \
  --doc-id-format kwdlc
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/AnnotatedFKCCorpus/knp" "${OUT_DIR}/fuman" \
  --id "${WORK_DIR}/AnnotatedFKCCorpus/id/split_for_pas" \
  -j "${JOBS}"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/WikipediaAnnotatedCorpus/knp" "${OUT_DIR}/wac" \
  --id "${WORK_DIR}/WikipediaAnnotatedCorpus/id" \
  -j "${JOBS}" \
  --doc-id-format wac
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/multimodal-annotation-data/knp" "${OUT_DIR}/jcre3" \
  --id "${WORK_DIR}/multimodal-annotation-data/id" \
  -j "${JOBS}"

rm -rf "${WORK_DIR}"
