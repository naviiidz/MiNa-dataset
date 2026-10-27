#!/usr/bin/env bash
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D2=/home/navid/miniconda3/envs/detectron2/bin/python
UL=/home/navid/miniconda3/envs/yolo-sam/bin/python
LOG="$REPO/results/run_detectors_agnostic.log"

run() {
  echo "$(date '+%F %T') START $1" | tee -a "$LOG"
  shift
  if "$@"; then echo "$(date '+%F %T') DONE" | tee -a "$LOG"; else echo "$(date '+%F %T') FAIL" | tee -a "$LOG"; fi
}

for s in 0 1 2; do
  run "YOLO26s agnostic seed $s" bash -c "cd '$REPO/YOLOv26-detection' && $UL quick_train.py --model yolo26s --single_cls --seed $s --name yolo26s_agnostic_seed$s > train_agnostic_seed$s.log 2>&1"
done
for s in 0 1 2; do
  run "RT-DETR-L agnostic seed $s" bash -c "cd '$REPO/RT-DETR' && $UL train.py --single_cls --seed $s --patience 30 --name rtdetr-l_agnostic_seed$s > logs/train_agnostic_seed$s.log 2>&1"
done
for v in R101 X101; do
  for s in 42 1 2; do
    d="FasterRCNN-${v}_agnostic"; [ "$s" != 42 ] && d="${d}_seed$s"
    run "Faster R-CNN $v agnostic seed $s" bash -c "mkdir -p '$REPO/FasterRCNN/zoo/current/$d' && cd '$REPO/FasterRCNN/R101-X101-FPN' && $D2 -W ignore train.py --model-variant FasterRCNN-$v --single-class --seed $s > '$REPO/FasterRCNN/zoo/current/$d/train.log' 2>&1"
  done
done
python3 "$REPO/results/localization_seeds.py" >> "$LOG" 2>&1
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
