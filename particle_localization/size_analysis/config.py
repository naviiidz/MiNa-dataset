import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOC = os.path.dirname(HERE)
REPO = os.path.dirname(LOC)
PATCHES = os.path.join(REPO, "data", "patches", "coco", "patches_256_combined")
FULL_IMAGES = os.path.join(REPO, "data", "full_images")
PIXEL_SIZE_CSV = os.path.join(REPO, "data", "polymer_identification", "particle_features", "features.csv")
OUTPUT_DIR = os.path.join(HERE, "outputs")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

SPLITS = ["val", "test"]
SEEDS = [0, 1, 2]
MODELS = {
    "rtdetr": {"name": "RT-DETR-L", "task": "detection",
               "weights": [os.path.join(LOC, "RT-DETR", "runs", "rtdetr-l", f"rtdetr-l_agnostic_seed{s}", "weights", "best.pt") for s in SEEDS]},
    "maskrcnn": {"name": "Mask R-CNN R101", "task": "segmentation",
                 "weights": [os.path.join(LOC, "MaskRCNN", "zoo", "current", d, "model_final.pth")
                             for d in ("R101-FPN", "R101-FPN_seed1", "R101-FPN_seed2")]},
    "maskrcnn_x101": {"name": "Mask R-CNN X101", "task": "segmentation",
                      "weights": [os.path.join(LOC, "MaskRCNN", "zoo", "current", d, "model_final.pth")
                                  for d in ("X101-FPN", "X101-FPN_seed1", "X101-FPN_seed2")]},
    "yolov8seg": {"name": "YOLOv8n-seg", "task": "segmentation",
                  "weights": [os.path.join(LOC, "YOLOv8-seg", "runs", "segment", d, "weights", "best.pt")
                              for d in ("yolov8n-seg_patches", "yolov8n-seg_patches_seed1", "yolov8n-seg_patches_seed2")]},
    "yolo26seg": {"name": "YOLO26n-seg", "task": "segmentation",
                  "weights": [os.path.join(LOC, "YOLOv26-seg", "runs", "segment", d, "weights", "best.pt")
                              for d in ("yolo26n-seg_patches", "yolo26n-seg_patches_seed1", "yolo26n-seg_patches_seed2")]},
}

PRED_CONF = 0.001
MASKRCNN_SCORE_THRESH = 0.05
MAX_DET = 300
COCO_MAX_DETS = 100
IMGSZ = 640

MIN_VISIBLE_FRACTION = 0.9
PX_BINS = [2, 3, 4, 6, 8, 12, 16, 24, 32, 64, 1e9]
NM_BINS = [0, 250, 500, 1000, 2000, 5000, 1e9]
FIT_MAX_PX = 16
SIZE_CATEGORIES = {"small": (0, 15), "medium": (15, 40), "large": (40, 1e9)}
IOU_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]


def prediction_path(model, seed, split):
    return os.path.join(PRED_DIR, f"{model}_seed{seed}_{split}.json")
