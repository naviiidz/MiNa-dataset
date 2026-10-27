#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
SIZES=${SIZES:-"128 256 512"}
SEEDS=${SEEDS:-"0 1 2"}

$PYTHON build_datasets.py --sizes $SIZES
$PYTHON train.py --sizes $SIZES --seeds $SEEDS
$PYTHON evaluate.py --sizes $SIZES --seeds $SEEDS
