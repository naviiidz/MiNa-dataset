import argparse
import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(PACKAGE_DIR)
REPO = os.path.dirname(PROJECT_DIR)

DATA_DIR = os.path.join(REPO, "data")
FULL_IMAGES = os.path.join(DATA_DIR, "full_images")
PATCH_COCO = os.path.join(DATA_DIR, "patches", "coco")
WORK_DATA_DIR = os.path.join(DATA_DIR, "polymer_identification")
FEATURES_DIR = os.path.join(WORK_DATA_DIR, "particle_features")
FEATURES_CSV = os.path.join(FEATURES_DIR, "features.csv")
PARTICLE_IMAGES_DIR = os.path.join(WORK_DATA_DIR, "single_particle")

OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PRETRAINED_DIR = os.path.join(PROJECT_DIR, "pretrained")

POLYMERS = ["PP", "PE", "PS", "PET"]
SPLITS = ["train", "val", "test"]

EXCLUDED_IMAGE_PREFIXES = ["PETL"]
N_FOLDS = 5
FOLDS = list(range(N_FOLDS))

MIN_AREA_PX = 10
EXCLUDE_NAME_MISMATCH = True
DRAW_SEED = 0

CANVAS = 112
IMAGE_SIZE = 224
NORM_CENTER = 128
NORM_SCALE = 40

EPOCHS = 100
BATCH = 64
SEED = 0

CLASSIFIERS = {
    "yolov8s-cls": ("ultralytics", os.path.join(PRETRAINED_DIR, "yolov8s-cls.pt"), 1e-3),
    "yolo26s-cls": ("ultralytics", os.path.join(PRETRAINED_DIR, "yolo26s-cls.pt"), 1e-3),
    "hgnetv2_b4": ("timm", "hgnetv2_b4.ssld_stage2_ft_in1k", 3e-4),
    "resnet101": ("timm", "resnet101.a1h_in1k", 3e-4),
    "convnext_tiny": ("timm", "convnext_tiny.fb_in22k_ft_in1k", 2e-4),
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m", 5e-5),
}

TABLE = ["eq_diameter_um", "area_um2", "aspect_ratio", "form_factor", "roundness", "convexity"]
SIZE = ["eq_diameter_um", "area_um2", "perimeter_um", "major_axis_um", "minor_axis_um", "feret_max_um"]
SHAPE = ["aspect_ratio", "form_factor", "roundness", "convexity", "solidity", "extent", "eccentricity"]
GLCM_PROPS = ("contrast", "homogeneity", "energy", "correlation")
APPEARANCE = ["grey_mean", "grey_std", "grey_p10", "grey_p50", "grey_p90", "edge_mean"] + [f"glcm_{g}" for g in GLCM_PROPS]
APPEARANCE_Z = ["z_p10", "z_p50", "z_p90", "z_edge_mean"] + [f"z_glcm_{g}" for g in GLCM_PROPS]
APPEARANCE_RANK = ["rank_edge_mean"] + [f"rank_glcm_{g}" for g in GLCM_PROPS]


def fold_paths(fold):
    sub = f"fold_{fold}"
    return {"images": os.path.join(PARTICLE_IMAGES_DIR, sub),
            "splits": os.path.join(OUTPUT_DIR, "splits", sub),
            "runs": os.path.join(OUTPUT_DIR, "runs", sub),
            "predictions": os.path.join(OUTPUT_DIR, "predictions", sub)}


def particle_id(polymer, source_image, annotation_id):
    return f"{polymer}|{source_image}|{int(annotation_id)}"


def add_fold_args(parser):
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS, choices=FOLDS,
                        help="cross-validation folds to run (default: all)")
    return parser


def parser(description):
    return add_fold_args(argparse.ArgumentParser(description=description))
