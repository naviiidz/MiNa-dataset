#!/usr/bin/env bash
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D2=/home/navid/miniconda3/envs/detectron2/bin/python
UL=/home/navid/miniconda3/envs/yolo-sam/bin/python
LOG="$REPO/results/run_seeds.log"
SEEDS=${SEEDS:-"1 2"}

until grep -qE "ALL DONE|FAIL" "$REPO/results/run_queue.log"; do sleep 60; done

run() {
  echo "$(date '+%F %T') START $1" | tee -a "$LOG"
  shift
  if "$@"; then echo "$(date '+%F %T') DONE" | tee -a "$LOG"; else echo "$(date '+%F %T') FAIL" | tee -a "$LOG"; fi
}

for s in $SEEDS; do
  run "YOLOv8n-seg seed $s" bash -c "cd '$REPO/YOLOv8-seg' && $UL train.py --seed $s > train_seed$s.log 2>&1"
  run "YOLO26n-seg seed $s" bash -c "cd '$REPO/YOLOv26-seg' && $UL quick_train.py --seed $s > train_seed$s.log 2>&1"
done
for s in $SEEDS; do
  for v in R101-FPN X101-FPN; do
    run "Mask R-CNN $v seed $s" bash -c "mkdir -p '$REPO/MaskRCNN/zoo/current/${v}_seed$s' && cd '$REPO/MaskRCNN/R101-X101-FPN' && $D2 -W ignore train.py --model-variant $v --seed $s > '$REPO/MaskRCNN/zoo/current/${v}_seed$s/train.log' 2>&1"
  done
done
python3 "$REPO/results/segmentation_seeds.py" >> "$LOG" 2>&1
echo "$(date '+%F %T') ALL SEEDS DONE" | tee -a "$LOG"
