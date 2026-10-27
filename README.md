# MiNa: SEM images of micro- and nanoplastic particles

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23627019.svg)](https://doi.org/10.5281/zenodo.23627019)

MiNa is a dataset of scanning electron microscopy (SEM) images of micro- and nanoplastic particles of four
polymers: polypropylene (PP), polyethylene (PE), polystyrene (PS) and polyethylene terephthalate (PET). Every
particle is outlined by hand. The dataset supports two tasks: **particle localization** (detecting and segmenting
every particle in an image) and **polymer identification** (recognizing the polymer of a single particle from its
image).

This repository contains the code to prepare the data, train and evaluate all models, and reproduce the results of
the paper. The data are available on Zenodo: **[doi.org/10.5281/zenodo.23627019](https://doi.org/10.5281/zenodo.23627019)**.

## Dataset

The Zenodo record contains three archives:

| Archive | Content |
|---|---|
| `full_images.zip` | the original SEM micrographs (1280 × 960 px), grouped by polymer, with a polygon outline of every particle in COCO and YOLO format |
| `patches.zip` | the 256 × 256 px patches used for the localization benchmark, split into training, validation and test sets; COCO annotations with polymer labels or a single "particle" class, and YOLO detection and segmentation labels |
| `particles.zip` | one cropped image and binary mask per particle for polymer identification, a table of particle metadata and size and shape descriptors, and the particle sets of the five cross-validation folds |

The micrographs were acquired at magnifications from 100× to 10,000×, corresponding to pixel sizes of about
10–1,000 nm.

## Repository layout

```
MiNa-dataset/
├── data_processing/          patch generation, COCO to YOLO conversion, particle crops, tests
├── particle_localization/    detection, instance segmentation and detector-prompted SAM
├── polymer_identification/   descriptor-based and deep polymer classifiers
└── data/                     the dataset, downloaded from Zenodo (not in git)
```

| Folder | What it does | Details |
|---|---|---|
| `data_processing/` | cuts the micrographs into patches, converts COCO annotations to YOLO format and extracts single-particle crops | scripts are run from this folder; output goes to `data/` |
| `particle_localization/` | trains and evaluates RT-DETR-L, YOLO26s, Faster R-CNN, Mask R-CNN, YOLOv8n-seg and YOLO26n-seg, and RT-DETR-L + SAM 2.1 | [particle_localization/README.md](particle_localization/README.md) |
| `polymer_identification/` | identifies the polymer of single particles with classical classifiers on size and shape descriptors and with six deep image classifiers | [polymer_identification/README.md](polymer_identification/README.md) |

## Getting started

1. Clone the repository:

   ```bash
   git clone https://github.com/naviiidz/MiNa-dataset.git
   cd MiNa-dataset
   ```

2. Download the archives from [Zenodo](https://doi.org/10.5281/zenodo.23627019) and unpack them into `data/`:

   ```
   data/
   ├── full_images/<POLYMER>/   micrographs and COCO annotations (pp.json, pe.json, ps.json, pet.json)
   ├── patches/                 patches in COCO and YOLO format
   └── particles/               particle crops, masks and particles.csv
   ```

3. Follow the README of the task you are interested in:
   [particle localization](particle_localization/README.md) or
   [polymer identification](polymer_identification/README.md).

Each part has its own environment and installation instructions. A CUDA GPU is needed to train the networks.

## Citation

If you use the dataset or the code, please cite:

```bibtex
@article{mina,
  title   = {[TITLE]},
  author  = {Zarrabi, Navid and Rezvani, Hadi and Mehta, Ishaan and Kolios, Christopher and Jaafar, Hussein Ali and
             Kao, Cheng-Hao and Saeedi, Sajad and Yousefi, Nariman},
  journal = {[JOURNAL]},
  year    = {[YEAR]},
  doi     = {[DOI]}
}

@dataset{mina_data,
  title     = {MiNa dataset: SEM images of micro- and nanoplastic particles},
  author    = {Zarrabi, Navid and Rezvani, Hadi and Mehta, Ishaan and Kolios, Christopher and Jaafar, Hussein Ali and
               Kao, Cheng-Hao and Saeedi, Sajad and Yousefi, Nariman},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.23627019}
}
```

## Licence

The code is released under the [MIT licence](LICENSE). The dataset is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
