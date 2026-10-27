# Particle localization in SEM images

Find and outline every micro- and nanoplastic particle in SEM images, and compare detection, instance
segmentation and foundation-model segmentation on the same data.

All models are trained on 256 × 256 px patches of the MiNa micrographs as a single "particle" class
(polymer-agnostic), with three random seeds, and scored on the same 260 test patches.

| Task | Input | Models |
|---|---|---|
| Detection | image patch → boxes | RT-DETR-L, YOLO26s, Faster R-CNN (ResNet-101, ResNeXt-101) |
| Instance segmentation | image patch → masks | Mask R-CNN (ResNet-101, ResNeXt-101), YOLOv8n-seg, YOLO26n-seg |
| Detector-prompted SAM | image → boxes → SAM 2.1 masks | RT-DETR-L + SAM 2.1-L (box or point prompts); micro-sam and watershed baselines |

The detectors and YOLO segmentation models use Ultralytics; Faster R-CNN and Mask R-CNN use Detectron2. SAM is
used with its released weights, without fine-tuning.

## Repository layout

```
particle_localization/
├── RT-DETR/                RT-DETR-L training and test scoring (class-agnostic and 4-class)
├── YOLOv26-detection/      YOLO26s detection training and evaluation
├── FasterRCNN/             Faster R-CNN (Detectron2), R101 / X101 FPN
├── MaskRCNN/               Mask R-CNN (Detectron2), R101 / X101 FPN
├── YOLOv8-seg/             YOLOv8n-seg training and evaluation
├── YOLOv26-seg/            YOLO26n-seg training and evaluation
├── sam_prompting/          detector boxes prompt SAM 2.1; micro-sam and watershed baselines
│   ├── evaluation.py       shared datasets, ground truth, matching, metrics and outputs
│   ├── run_detector_sam.py RT-DETR / YOLO + SAM 2.1
│   ├── select_confidence.py detector threshold chosen on the validation patches
│   ├── run_micro_sam.py    micro-sam automatic segmentation
│   ├── run_watershed.py    classical watershed baseline
│   ├── run_revision.sh     all detector + SAM runs (3 seeds, box and point prompts)
│   └── make_revision_figures.py  result table and figures, one file per panel
├── size_analysis/          detection and segmentation quality vs particle size
├── patch_size_study/       YOLO26s trained on 128, 256 and 512 px patches
├── results/                collects test metrics of all models into tables
└── Yolo-sam/               earlier YOLO + SAM scripts (superseded by sam_prompting/)
```

Weights (`zoo/`, `model_zoo/`, `weights/`), training runs (`runs/`) and outputs are not in git.

## Installation

Two conda environments are used: one for Ultralytics and SAM, one for Detectron2.

```bash
# Ultralytics, SAM 2.1 and micro-sam
conda create -n yolo-sam python=3.13
conda activate yolo-sam
pip install -r sam_prompting/requirements_sam.txt

# Detectron2 (Faster R-CNN, Mask R-CNN)
conda create -n detectron2 python=3.10
conda activate detectron2
conda install -c nvidia cuda-toolkit=12.4
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
CUDA_HOME=$CONDA_PREFIX pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python pycocotools shapely scikit-learn matplotlib tqdm
```

A CUDA GPU is required for training. Run one training or inference script at a time; each loads its models onto
the GPU.

## Input data

Everything is read from the repository's `data/` folder, built by `../data_processing/`:

| Path | Content |
|---|---|
| `data/patches/coco/patches_256_combined/{train,val,test}/` | patches with COCO annotations (Detectron2 models) |
| `data/patches/yolo_det/` | patches with YOLO box labels (RT-DETR, YOLO26s) |
| `data/patches/yolo_seg/` | patches with YOLO polygon labels (YOLO segmentation, SAM evaluation) |
| `data/full_images/<POLYMER>/` | full micrographs and COCO annotations (size analysis, patch-size study) |
| `sam_prompting/data/<dataset>/` | external datasets for SAM evaluation (`image/`, `label/`) |

Rebuild the patches from `../data_processing/`:

```bash
python create_patches.py ALL
python combine_patches.py --oversample PE=3
python coco_to_yolo.py --mode det --output_dir ../data/patches/yolo_det
python coco_to_yolo.py --mode seg --output_dir ../data/patches/yolo_seg
```

### External datasets

The detector + SAM pipeline is also evaluated on four public SEM datasets. They are not included in this
repository; download them from their original sources and place each one in `sam_prompting/data/<name>/`, with the
images in `image/` and the labels in `label/`:

| `--dataset` | Source | Images | Labels |
|---|---|---|---|
| `fragment_1` | [Shi et al.](https://data.mendeley.com/datasets/z6459vntbr/1) | 55 PNG, 2560 × 1800 px | binary masks, same file name as the image |
| `fragment_2` | [Shi et al.](https://data.mendeley.com/datasets/z6459vntbr/1) | 41 PNG, 1280 × 898 px (2 at 800 × 561 px) | binary masks, same file name as the image |
| `dataset2_low_mag` | [Rettenberger et al.](https://osf.io/f2y8w/overview), low magnification | 20 PNG, 1920 × 1200 px | instance masks `<image>_label.tif` |
| `dataset2_high_mag` | [Rettenberger et al.](https://osf.io/f2y8w/overview), high magnification | 40 TIFF, 7680 × 4800 px (10 at 3840 × 2400 px) | instance masks `<image>_label.tif` |

Please cite the original publications when using these datasets.

## Quick start

Train all class-agnostic detectors with three seeds and collect the results:

```bash
cd particle_localization
./results/run_detectors_agnostic.sh
python results/localization_seeds.py      # -> results/localization_seeds.md
python results/segmentation_seeds.py      # -> results/segmentation_seeds.md
```

## Step-by-step usage

Run every script from its own folder; dataset paths are relative to it. Use `--help` on any script for all options.

**1. Detection** (`conda activate yolo-sam`, or `detectron2` for Faster R-CNN)

```bash
cd RT-DETR             && python train.py --single_cls --seed 0 --name rtdetr-l_agnostic_seed0
cd YOLOv26-detection   && python quick_train.py --model yolo26s --single_cls --seed 0 --name yolo26s_agnostic_seed0
cd FasterRCNN/R101-X101-FPN && python train.py --model-variant FasterRCNN-R101 --single-class --seed 42
```

**2. Instance segmentation**

```bash
cd MaskRCNN/R101-X101-FPN && python train.py --model-variant R101-FPN --seed 42      # detectron2
cd YOLOv8-seg            && python train.py --seed 0                                 # yolo-sam
cd YOLOv26-seg           && python quick_train.py --seed 0
```

Each training script scores its best (Ultralytics) or final (Detectron2) checkpoint on the test patches when it
finishes. Use `--epochs 1` (Ultralytics) or `--max-iter 50` (Detectron2) for a dry run.

**3. Detector-prompted SAM** (`conda activate yolo-sam`, from `sam_prompting/`)

```bash
python select_confidence.py --detector ../RT-DETR/runs/rtdetr-l/rtdetr-l_agnostic_seed0/weights/best.pt \
    --prompt box --name detector_sam_box_seed0
python run_detector_sam.py --dataset MiNa_dataset --prompt box --name detector_sam_box_seed0 \
    --detector ../RT-DETR/runs/rtdetr-l/rtdetr-l_agnostic_seed0/weights/best.pt \
    --conf-file outputs/detector_sam_box_seed0/MiNa_val/confidence.json
python run_micro_sam.py --dataset MiNa_dataset --model vit_l_lm --mode ais
python run_watershed.py --dataset MiNa_dataset
```

`--dataset` takes any entry of the `DATASETS` table in `evaluation.py` (`MiNa_dataset`, `MiNa_val`,
`fragment_1`, `fragment_2`, `dataset2_low_mag`, `dataset2_high_mag`).

**4. Size analysis** (from `size_analysis/`)

```bash
./run.sh
```

**5. Patch-size study** (from `patch_size_study/`)

```bash
./run.sh                    # 128, 256 and 512 px
SIZES="128 512" ./run.sh    # selected sizes
```

## Outputs

| Path | Content |
|---|---|
| `RT-DETR/runs/rtdetr-l/<run>/` | weights, training curves and `test_metrics.json` |
| `YOLOv26-detection/runs/yolo26s/<run>/` | weights, training curves and test metrics |
| `YOLOv8-seg/runs/segment/<run>/`, `YOLOv26-seg/runs/segment/<run>/` | weights, training curves and test metrics |
| `FasterRCNN/zoo/current/<variant>/`, `MaskRCNN/zoo/current/<variant>/` | weights and `results/test_metrics.json` |
| `results/localization_seeds.md`, `segmentation_seeds.md` | box and mask AP, mean ± SD over seeds |
| `sam_prompting/outputs/<method>/<dataset>/` | `summary.csv`, `per_image_metrics.csv`, `predictions.csv`, overlays |
| `sam_prompting/outputs/<method>/MiNa_val/` | `confidence.json` and `confidence_grid.csv` (threshold selection) |
| `size_analysis/outputs/results/` | metrics per size bin, D50 / D90 fits and figures |
| `patch_size_study/outputs/results/` | full-image metrics per patch size and `patch_size.png` |

## Configuration

**Training settings.** Ultralytics models: up to 100 epochs, input 640 × 640 px, batch 8, early stopping on
validation (patience 30 for RT-DETR-L, 10 for the YOLO models), framework-default optimizer and augmentation.
Detectron2 models: 25 epochs, batch 2, learning rate 5 × 10⁻⁴, halved after 15 and 20 epochs, shortest edge
400–800 px, anchors 8–192 px with aspect ratios 0.25–4. All models start from COCO-pretrained weights.

**Detector + SAM.** Set in `sam_prompting/run_detector_sam.py` and `select_confidence.py`:

| Setting | Default | Meaning |
|---|---|---|
| `--prompt` | `box` | box prompts, or `point` for box centres |
| `--conf-file` | none | JSON with the detector threshold chosen on validation |
| `GRID` | 0.3–0.7, step 0.1 | thresholds tried on the validation patches |
| `--dedupe-iou` | 0.5 | mask IoU above which a mask counts as a duplicate |

**Datasets for SAM.** Add a dataset by putting it in `sam_prompting/data/<name>/image` and `label` and adding
an entry to `DATASETS` in `sam_prompting/evaluation.py`.

## Evaluation protocol

- **No image leakage:** the split is made by source micrograph (58 / 13 / 13 micrographs for train, validation and
  test), so no micrograph contributes patches to more than one split.
- **Same test set for every model:** all models are scored on the same 260 test patches, including patches
  without particles.
- **AP:** COCO-style box and mask AP (AP, AP50, AP75) computed at a low score threshold so that the whole
  precision–recall curve is included.
- **Model selection:** checkpoints, early stopping and confidence thresholds are chosen on the validation patches;
  the test patches are used once.
- **Repeated runs:** every model is trained with three seeds and reported as mean ± SD.
- **Detector + SAM:** predictions are matched one-to-one to the ground truth (Hungarian algorithm on polygon IoU);
  pairs with IoU ≥ 0.5 are true positives. Precision, recall and F1 are pooled over all test patches; matched IoU,
  Dice and area MAPE are averaged over matched pairs.
