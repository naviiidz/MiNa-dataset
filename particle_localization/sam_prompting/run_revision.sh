#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=${PY:-/home/navid/miniconda3/envs/yolo-sam/bin/python}
LOG=outputs/run_revision.log
mkdir -p outputs
EXTERNAL="fragment_1 fragment_2 dataset2_high_mag dataset2_low_mag"

run() {
  echo "$(date '+%F %T') START $1" | tee -a "$LOG"
  shift
  if "$@" >> "$LOG" 2>&1; then echo "$(date '+%F %T') DONE" | tee -a "$LOG"; else echo "$(date '+%F %T') FAIL" | tee -a "$LOG"; fi
}

$PY -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "No GPU available" | tee -a "$LOG"; exit 1; }

for seed in 0 1 2; do
  det=../RT-DETR/runs/rtdetr-l/rtdetr-l_agnostic_seed$seed/weights/best.pt
  for prompt in box point; do
    name=detector_sam_${prompt}_seed$seed
    run "select confidence $name" $PY select_confidence.py --detector "$det" --prompt $prompt --name $name
    run "$name MiNa_dataset" $PY run_detector_sam.py --dataset MiNa_dataset --detector "$det" --prompt $prompt \
      --name $name --conf-file outputs/$name/MiNa_val/confidence.json
  done
  for ds in $EXTERNAL; do
    name=detector_sam_box_seed$seed
    run "$name $ds" $PY run_detector_sam.py --dataset $ds --detector "$det" --prompt box \
      --name $name --conf-file outputs/$name/MiNa_val/confidence.json
  done
done
echo "$(date '+%F %T') ALL DONE" | tee -a "$LOG"
