# SAM prompting for microplastic segmentation

Segment microplastic particles in microscope images by prompting SAM with boxes from an object detector (RT-DETR or YOLO), and compare it with two baselines, micro-sam and watershed, on several labelled datasets.

```
image ──► RT-DETR / YOLO ──► boxes ──► SAM 2.1 ──► masks ──► polygons ──► metrics / overlays
```

## Contents

| File | Purpose |
|---|---|
| `run_detector_sam.py` | **Main method.** Detector boxes prompt SAM 2.1. |
| `run_micro_sam.py` | Baseline: micro-sam automatic segmentation (AMG, AIS or APG), no detector. |
| `run_watershed.py` | Baseline without SAM: thresholding, morphology and watershed. |
| `evaluation.py` | Shared by the three `run_*` scripts: datasets, ground truth, matching, metrics, outputs. Not run directly. |
| `model_zoo/` | Weights: `rt-detr.pt`, `yolo8.pt`, `yolo26.pt` (detectors), `sam2.1_l.pt` (SAM 2.1-L), `vit_l_lm` (micro-sam). |
| `data/` | Datasets (see below). |
| `requirements_sam.txt` | Package list of the original environment. |

`data/`, `model_zoo/` and `outputs/` are not in git (see `.gitignore`).

## Setup

```bash
conda activate yolo-sam
cd ~/MiNa-dataset/sam_prompting   # all paths in the scripts are relative to this folder
```

The environment has PyTorch 2.10 (CUDA 13), ultralytics 8.4, micro-sam 1.8, opencv, shapely, pandas and scipy. Models run on the GPU automatically when CUDA is available.

## Quick start

Each `run_*` script predicts on every image of a dataset, saves the results and scores them against the labels:

```bash
python run_detector_sam.py --dataset MiNa_dataset
python run_micro_sam.py    --dataset MiNa_dataset
python run_watershed.py    --dataset MiNa_dataset
```

Results go to `outputs/<method>/<dataset>/`. Run one script at a time; each loads its own models onto the GPU.

## Datasets

`--dataset` takes one of these names, defined in the `DATASETS` table at the top of `evaluation.py`:

| `--dataset` | Folder in `data/` | Images | Labels |
|---|---|---|---|
| `MiNa_dataset` | `patches_256_combined/images/test`, `labels/test` | 119 × 256×256 PNG | YOLO text files, one polygon per particle (`class x1 y1 x2 y2 …`, normalized). The held-out test split of the patches the detector was trained on. |
| `dataset2_high_mag` | `dataset2_high_mag/image`, `label` | 40 × 7680×4800 TIFF | Masks `X_label.tif`, values 0–9 reused across particles |
| `dataset2_low_mag` | `dataset2_low_mag/image`, `label` | 20 × 1920×1200 PNG | Masks `X_label.tif`, values 0–9 reused across particles |
| `fragment_1` | `fragment_1/image`, `label` | 55 × 2560×1794 PNG | Masks `X.png`, binary 0/255 with a few stray low-value pixels |
| `fragment_2` | `fragment_2/image`, `label` | 41 PNG | Masks `X.png`, binary 0/255 |

### Adding a dataset

1. Put it in `data/<name>/image` and `data/<name>/label` (or set `image_dir` / `label_dir` in its entry).
2. Add an entry to `DATASETS` in `evaluation.py` with its `label_format` and, for masks, `foreground_threshold`, `instances_per_value` and `label_suffix`.
3. Run any method with `--dataset <name>`.

## Methods

### `run_detector_sam.py`: detector + SAM

For each image:

1. Resize to 260×260 and run the detector.
2. Scale the boxes back to full resolution and give them to SAM 2.1 as box prompts.
3. Remove duplicate masks. The detector often reports one particle several times (once per class), so a mask overlapping an already kept mask with IoU > 0.5 is dropped.
4. Keep the largest outline of each mask as that particle's polygon.

| Option | Default | Meaning |
|---|---|---|
| `--conf` | `0.6` | Detector confidence threshold, the same for every image. |
| `--sweep` | off | Try 0.5, 0.6, 0.7, 0.8 and 0.9 and keep each image's best by F1 (see below). |
| `--detector` | `model_zoo/rt-detr.pt` | Detector weights (RT-DETR or YOLO). |
| `--sam` | `model_zoo/sam2.1_l.pt` | SAM weights. |
| `--prompt` | `box` | `box`, or `point` to prompt with box centres. |
| `--dedupe-iou` | `0.5` | IoU above which a mask counts as a duplicate. |
| `--exclude-background` | off | Use SAM 3 with a text prompt (`--background-text`, default "porous filter membrane") to remove the filter membrane from the masks. Needs `model_zoo/sam3.pt`, which is gated on Hugging Face. |

**About `--sweep`:** each image's best threshold is chosen by looking at its labels, so the results are an optimistic upper bound, not a fair score. Use it to see how sensitive the method is to the threshold (it also writes `confidence_sweep.csv` and a sweep AP50), and report results at one fixed `--conf`.

### `run_micro_sam.py`: micro-sam

Automatic segmentation of the whole image with micro-sam, no detector. Results go to `outputs/micro_sam_<mode>/<dataset>/`.

| Option | Default | Meaning |
|---|---|---|
| `--model` | `vit_l` | micro-sam model, e.g. `vit_l` (SAM), `vit_b_lm` / `vit_l_lm` (light microscopy). |
| `--mode` | `amg` | `amg` (automatic mask generation, any model), or `ais` / `apg` (need a micro-sam model). |
| `--checkpoint` | none | Optional fine-tuned checkpoint. |

The post-processing settings of each mode are in `GENERATE_KWARGS` at the top of the script.

### `run_watershed.py`: watershed

Otsu threshold, opening and closing, distance transform, watershed from the distance peaks, then removal of small regions.

| Option | Default | Meaning |
|---|---|---|
| `--threshold-method` | `otsu` | `otsu`, `adaptive` or `fixed` (with `--threshold-value`). |
| `--morph-kernel-size` | `5` | Kernel size for opening and closing. |
| `--min-size` | `15` | Smallest particle kept, in pixels. |

The threshold is inverted (dark regions become particles), so on filter-membrane images it mostly picks up the membrane's pores; expect low scores.

## Evaluation

All methods are scored by the same code in `evaluation.py`. Each predicted particle is matched to at most one ground-truth particle (Hungarian algorithm on polygon IoU); a pair with IoU ≥ 0.5 is a true positive.

| Metric | Formula | Meaning |
|---|---|---|
| IoU (intersection over union) | \|P ∩ G\| / \|P ∪ G\| | Overlap between a predicted region P and the true region G, from 0 (no overlap) to 1 (identical). |
| Dice (DSC) | 2 \|P ∩ G\| / (\|P\| + \|G\|) | The same overlap measured differently; also 0 to 1, and always at least as high as IoU. |
| MAPE (mean absolute percentage error) | mean of \|area(P) − area(G)\| / area(G) × 100 | How far the predicted particle sizes are from the true sizes, in %. Lower is better. |

IoU and Dice are reported twice: for the matched particle pairs (`matched_iou`, `matched_dsc`: how well each found particle is outlined) and for the whole image (`mask_iou`, `mask_dice`: all predicted pixels against all true particle pixels). MAPE (`area_mape`) is computed over the matched pairs.

**`--no-eval`** (all scripts) skips labels and metrics: the script only predicts and saves overlays and `predictions.csv`. Use it for images without labels.

## Outputs (`outputs/<method>/<dataset>/`)

| File | Contents |
|---|---|
| `overlays/<image>.png` | Predictions drawn on the image (colour fill, red outlines), downscaled to at most 2000 px. |
| `predictions.csv` | One row per predicted particle: area, bounding box, centroid, polygon coordinates. |
| `per_image_metrics.csv` | One row per image: counts, metrics, run time. |
| `summary.csv` | One row with the dataset-level results. |
| `confidence_sweep.csv` | Only with `--sweep`: one row per image and threshold; `is_best` marks the chosen one. |
| `run_config.json` | The method, dataset and settings used for the run. |
