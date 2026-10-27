#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python}
CLASSIFIERS=${CLASSIFIERS:-"yolov8s-cls yolo26s-cls hgnetv2_b4 resnet101 convnext_tiny dinov2_vits14"}
STAGES=${*:-features splits descriptors classifiers evaluate}

for stage in $STAGES; do
  echo "=== $stage"
  case $stage in
    features)        $PYTHON -m polymer_id.features ;;
    splits)          $PYTHON -m polymer_id.splits
                     $PYTHON -m polymer_id.images ;;
    descriptors)     $PYTHON -m polymer_id.descriptor_models ;;
    classifiers)     $PYTHON -m polymer_id.classifiers --model $CLASSIFIERS ;;
    evaluate)        $PYTHON -m polymer_id.evaluate
                     $PYTHON -m polymer_id.plots ;;
    interpretability) $PYTHON -m polymer_id.interpretability ;;
    *) echo "unknown stage: $stage" >&2; exit 1 ;;
  esac
done
