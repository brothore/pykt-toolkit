#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

for dataset in nips_task34 assist2009 peiyou; do
    archive="${SCRIPT_DIR}/${dataset}.zip"
    if [[ ! -f "${archive}" ]]; then
        echo "Missing dataset archive: ${archive}" >&2
        exit 1
    fi
    unzip -o -q "${archive}" -d "${SCRIPT_DIR}/${dataset}"
done

cd "${PROJECT_ROOT}/examples"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for dataset in assist2009 nips_task34 peiyou; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Processing ${dataset}"
    python data_preprocess.py -d "${dataset}"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] All datasets processed successfully"
