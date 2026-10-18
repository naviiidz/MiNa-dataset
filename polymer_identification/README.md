# Polymer identification of single microplastic particles

Identify the polymer (PP, PE, PS, PET) of individual microplastic particles in SEM images, and compare
classical classifiers on measured particle descriptors with deep image classifiers.

Each annotated particle is cut out of the full SEM image and masked, so every method sees one particle
with the background removed. All methods are trained and evaluated with the same 5-fold cross-validation,
grouped by source image and class-balanced. Images of the `PETL` series are excluded.

| Method family | Input | Models |
|---|---|---|
| Descriptors + ML | size, shape and texture descriptors measured from the particle mask | logistic regression, k-NN, SVM, random forest, gradient boosting |
| Deep classifiers | single-particle image | YOLOv8s-cls, YOLO26s-cls, HGNetV2-B4, ResNet-101, ConvNeXt-T, DINOv2 ViT-S/14 |

Because every input holds a single, isolated particle, the task is image classification. HGNetV2-B4 and
ResNet-101 are the backbones of the RT-DETR and Faster R-CNN detectors used for particle detection
(`../particle_localization/`).

## Repository layout

```
polymer_identification/
├── polymer_id/               Python package, one module per pipeline stage
│   ├── config.py             paths, excluded images, model registry and all settings
│   ├── features.py           cut out particles, measure descriptors
│   ├── splits.py             cross-validation folds and class-balanced particle sets
│   ├── images.py             normalised single-particle images
│   ├── descriptor_models.py  classical ML on the measured descriptors
│   ├── classifiers.py        deep image classifiers
│   ├── evaluate.py           precision, recall, F1 and result tables
│   ├── plots.py              performance vs particle resolution
│   └── interpretability.py   input ablations, Grad-CAM, descriptor surrogate
├── pretrained/               pretrained classifier weights (YOLO)
├── outputs/                  everything the pipeline writes (git-ignored)
├── run_pipeline.sh           runs the whole pipeline or selected stages
└── requirements.txt
```

## Installation

```bash
conda create -n polymer-id python=3.12
conda activate polymer-id
pip install -r requirements.txt
```

A CUDA GPU is required to train the deep classifiers. Pretrained weights for the timm models are downloaded
on first use.

## Input data

The pipeline reads from the repository's `data/` folder:

| Path | Content |
|---|---|
| `data/full_images/<POLYMER>/` | full SEM images and their COCO annotations (`pp.json`, `pe.json`, `ps.json`, `pet.json`) |
| `data/patches/coco/patches_256_<polymer>/split_source_images.json` | source-image split from the patch pipeline |
| `../data_processing/create_patches.py` | provides the excluded-image list and SEM info-bar height |

Intermediate data is written to `data/polymer_identification/`.

## Quick start

Run the full pipeline:

```bash
cd polymer_identification
./run_pipeline.sh
```

Run selected stages only:

```bash
./run_pipeline.sh splits descriptors evaluate
CLASSIFIERS="resnet101 dinov2_vits14" ./run_pipeline.sh classifiers evaluate
```

Available stages, in order: `features`, `splits`, `descriptors`, `classifiers`, `evaluate`, `interpretability`.
Set `PYTHON` to use an interpreter other than `python` on the path.

## Step-by-step usage

Every stage is a module that can be run on its own. Run the commands from `polymer_identification/`.
The training stages accept `--folds` to run selected folds only; use `--help` on any module for the full list
of options.

**1. Extract particles and descriptors**

```bash
python -m polymer_id.features
```

Writes masked particle crops, masks and `features.csv` to `data/polymer_identification/particle_features/`.
Pixel size is taken from the SEM scale bar, so size descriptors are in µm.

**2. Build folds and particle images**

```bash
python -m polymer_id.splits
python -m polymer_id.images
```

Source images are assigned to 5 folds, stratified by magnification. In each fold, one fold is test, the next is
validation and the rest is training. Every split is balanced to the size of the smallest polymer class. Each
particle is rendered at native pixel size with per-particle intensity normalisation.

**3. Train the models**

```bash
python -m polymer_id.descriptor_models
python -m polymer_id.classifiers --model resnet101 dinov2_vits14
```

Train a single fold with `--folds 0`, or shorten training with `--epochs`.

**4. Evaluate**

```bash
python -m polymer_id.evaluate
python -m polymer_id.plots
python -m polymer_id.interpretability     # optional
```

## Outputs

| Path | Content |
|---|---|
| `outputs/splits/fold_<k>/` | image-to-fold assignment and the balanced particle lists |
| `outputs/runs/fold_<k>/` | trained weights and training histories |
| `outputs/predictions/fold_<k>/` | one CSV of test predictions per method |
| `outputs/results/summary.md` | macro precision, recall and F1 of every method |
| `outputs/results/metrics.csv` | per-polymer and macro scores |
| `outputs/results/fold_f1.csv` | macro F1 per fold |
| `outputs/results/resolution.csv`, `resolution.png` | macro F1 vs particle size in pixels |
| `outputs/results/per_image_accuracy.csv` | accuracy of every method on every test image |
| `outputs/results/interpretability/` | ablations, Grad-CAM maps and surrogate-model importances |

## Configuration

All settings are in `polymer_id/config.py`.

**Excluded images.** `EXCLUDED_IMAGE_PREFIXES` lists source-image name prefixes left out of all splits
(default: the `PETL` series). Images whose file name names a different polymer are always excluded.

**Models.** Deep classifiers are listed in `CLASSIFIERS`. Any model from [timm](https://huggingface.co/timm) can be
added with one line:

```python
"efficientnet_b0": ("timm", "efficientnet_b0.ra_in1k", 3e-4),   # source, model id, learning rate
```

**Main settings**

| Setting | Default | Meaning |
|---|---|---|
| `N_FOLDS` | 5 | cross-validation folds |
| `MIN_AREA_PX` | 10 | smallest particle area used |
| `CANVAS`, `IMAGE_SIZE` | 112, 224 | particle canvas in native pixels, network input size |
| `EPOCHS`, `BATCH` | 100, 64 | classifier training |
| `TABLE` | 6 descriptors | features used by the descriptor models |

## Using the package in Python

```python
from polymer_id import classifiers, evaluate, splits

splits.make_split(fold=0)
classifiers.train_fold("convnext_tiny", fold=0, epochs=50)
model, mean, std = classifiers.load_trained("convnext_tiny", fold=0)
metrics = evaluate.evaluate()
```

## Evaluation protocol

- **No image leakage:** folds are formed from whole source images, so particles from one image never appear in
  both training and test.
- **Class balance:** every polymer contributes the same number of particles to each split, so chance level is
  25% and accuracy equals balanced accuracy.
- **Model selection:** checkpoints and hyperparameters are chosen on validation macro F1. The test fold is used
  only once.
- **Metrics:** precision, recall and F1 per polymer and macro-averaged, pooled over the 5 test folds, with the
  spread across folds.
