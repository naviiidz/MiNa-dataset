#!/usr/bin/env bash
# Train the patch models one at a time (single GPU). After each run, check that a
# test-metrics file scored on 130 images exists; stop the queue at the first failure.
# Usage: bash results/run_queue.sh [first_step]   (steps are numbered 1-7 below)
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
D2=/home/navid/miniconda3/envs/detectron2/bin/python
UL=/home/navid/miniconda3/envs/yolo-sam/bin/python
LOG="$REPO/results/run_queue.log"
FIRST=${1:-1}

# Wait for a run that is already training (e.g. started by hand) to finish first
while pgrep -f "quick_train.py|R101-X101-FPN/train.py|YOLOv8-seg.*train.py" >/dev/null; do sleep 30; done

check() {  # check <metrics.json>
  python3 - "$1" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
n = d["test"]["num_images"] if "test" in d else d["num_images"]
sys.exit(0 if n == 260 else 1)
EOF
}

rcnn() {  # rcnn <project> <variant> <zoo-dir-name>
  local dir="$REPO/$1/zoo/current/$3"; mkdir -p "$dir"
  (cd "$REPO/$1/R101-X101-FPN" && $D2 -W ignore train.py --model-variant "$2" > "$dir/train.log" 2>&1) || return 1
  check "$dir/results/test_metrics.json" || return 1
  if [ "$1" = FasterRCNN ]; then
    (cd "$REPO/$1/R101-X101-FPN" && $D2 -W ignore evaluate.py --model_path "$dir/model_final.pth" --model_variant "$2" --split test > "$dir/evaluate_test.log" 2>&1)
  else
    (cd "$REPO/$1/R101-X101-FPN" && $D2 -W ignore evaluate_improved_model.py --model_path "$dir/model_final.pth" --split test > "$dir/evaluate_test.log" 2>&1)
  fi
}

ultra() {  # ultra <folder> <command...>; metrics glob is checked afterwards
  local folder=$1; shift
  (cd "$REPO/$folder" && $UL "$@" > "$REPO/$folder/train_$(echo "$*" | tr ' /' '__').log" 2>&1)
}

latest() { ls -t $1 2>/dev/null | head -1; }

step() {  # step <n> <label> <command...>
  local n=$1 label=$2; shift 2
  [ "$n" -lt "$FIRST" ] && return 0
  echo "$(date '+%F %T') START $n $label" | tee -a "$LOG"
  local t0=$(date +%s)
  if "$@"; then
    echo "$(date '+%F %T') DONE  $n $label ($(( ($(date +%s) - t0) / 60 )) min)" | tee -a "$LOG"
    python3 "$REPO/results/collect_metrics.py" >> "$LOG" 2>&1
  else
    echo "$(date '+%F %T') FAIL  $n $label, queue stopped" | tee -a "$LOG"; exit 1
  fi
}

yolo_det() { ultra YOLOv26-detection quick_train.py --model "$1" && check "$(latest "$REPO/YOLOv26-detection/runs/$1/mnp_$1*/test_metrics.json")"; }
yolo_seg() { ultra "$1" "$2" && check "$(latest "$REPO/$1/runs/segment/*_patches*/test_metrics.json")"; }

step 1 "Faster R-CNN R101"   rcnn FasterRCNN FasterRCNN-R101 FasterRCNN-R101
step 2 "Faster R-CNN X101"   rcnn FasterRCNN FasterRCNN-X101 FasterRCNN-X101
step 3 "YOLO26s detection"   yolo_det yolo26s
step 4 "Mask R-CNN R101"     rcnn MaskRCNN R101-FPN R101-FPN
step 5 "Mask R-CNN X101"     rcnn MaskRCNN X101-FPN X101-FPN
step 6 "YOLOv8n-seg"         yolo_seg YOLOv8-seg train.py
step 7 "YOLO26n-seg"         yolo_seg YOLOv26-seg quick_train.py
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
