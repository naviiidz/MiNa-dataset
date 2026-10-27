#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

UL=${UL:-/home/navid/miniconda3/envs/yolo-sam/bin/python}
D2=${D2:-/home/navid/miniconda3/envs/detectron2/bin/python}

$UL predict_rtdetr.py
$D2 -W ignore predict_maskrcnn.py
$D2 evaluate.py --check
$D2 evaluate.py
