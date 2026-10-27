import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
DATA_PROCESSING = os.path.join(REPO, "data_processing")
FULL_IMAGES = os.path.join(REPO, "data", "full_images")
REFERENCE_SPLITS = os.path.join(REPO, "data", "patches", "coco")
STUDY_DATA = os.path.join(REPO, "data", "patch_size_study")
OUTPUT_DIR = os.path.join(HERE, "outputs")

PATCH_SIZES = [128, 256, 512]
POLYMERS = ["PP", "PE", "PS", "PET"]
OVERSAMPLE = "PE=3"

MODEL = "yolo26s"
PRETRAINED = os.path.join(REPO, "particle_localization", "YOLOv26-detection", "yolo26s.pt")
IMGSZ = 640
EPOCHS = 100
PATIENCE = 30
BATCH = 16
SEEDS = [0, 1, 2]

CONF = 0.001
MERGE_IOU = 0.5
MAX_DET_PER_IMAGE = 3000


def size_paths(size, seed=0):
    root = os.path.join(STUDY_DATA, str(size))
    return {"root": root, "coco": os.path.join(root, "coco"), "combined": os.path.join(root, "coco", "patches_256_combined"),
            "yolo": os.path.join(root, "yolo_det"), "runs": os.path.join(OUTPUT_DIR, "runs"),
            "run_name": f"patch_{size}_seed{seed}",
            "weights": os.path.join(OUTPUT_DIR, "runs", f"patch_{size}_seed{seed}", "weights", "best.pt")}
