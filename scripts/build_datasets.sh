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

mkdir -p "${WORK_DIR}" "${OUT_DIR}"/{kwdlc,fuman,wac}
git clone --depth 1 git@github.com:ku-nlp/KWDLC.git "${WORK_DIR}/KWDLC"
git clone --depth 1 git@github.com:ku-nlp/AnnotatedFKCCorpus.git "${WORK_DIR}/AnnotatedFKCCorpus"
git clone --depth 1 git@github.com:ku-nlp/WikipediaAnnotatedCorpus.git "${WORK_DIR}/WikipediaAnnotatedCorpus"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/KWDLC/knp" "${OUT_DIR}/kwdlc" \
  --id "${WORK_DIR}/KWDLC/id/split_for_pas" \
  -j "${JOBS}"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/AnnotatedFKCCorpus/knp" "${OUT_DIR}/fuman" \
  --id "${WORK_DIR}/AnnotatedFKCCorpus/id/split_for_pas" \
  -j "${JOBS}"
poetry run python ./scripts/build_dataset.py "${WORK_DIR}/WikipediaAnnotatedCorpus/knp" "${OUT_DIR}/wac" \
  --id "${WORK_DIR}/WikipediaAnnotatedCorpus/id" \
  -j "${JOBS}" \
  --doc-id-format wac

rm -rf "${WORK_DIR}"
