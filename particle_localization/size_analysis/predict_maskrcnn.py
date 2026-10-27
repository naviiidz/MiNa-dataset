import argparse
import json
import os
import sys

import cv2
import numpy as np

from config import LOC, MASKRCNN_SCORE_THRESH, MAX_DET, MODELS, PATCHES, PRED_DIR, SEEDS, SPLITS, prediction_path

sys.path.insert(0, os.path.join(LOC, "MaskRCNN", "R101-X101-FPN"))

VARIANTS = {"maskrcnn": "R101-FPN", "maskrcnn_x101": "X101-FPN"}


def predictor(seed, key):
    from detectron2.engine import DefaultPredictor
    from train import get_improved_cfg
    cfg = get_improved_cfg(1462, model_variant=VARIANTS[key])
    cfg.MODEL.WEIGHTS = MODELS[key]["weights"][seed]
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = MASKRCNN_SCORE_THRESH
    cfg.TEST.DETECTIONS_PER_IMAGE = MAX_DET
    return DefaultPredictor(cfg)


def predict(seed, split, key="maskrcnn"):
    from pycocotools import mask as mask_utils
    model = predictor(seed, key)
    with open(os.path.join(PATCHES, split, "annotations.json")) as fh:
        images = json.load(fh)["images"]
    out = []
    for im in images:
        inst = model(cv2.imread(os.path.join(PATCHES, split, "images", im["file_name"])))["instances"].to("cpu")
        boxes, scores, masks = inst.pred_boxes.tensor.numpy(), inst.scores.numpy(), inst.pred_masks.numpy()
        for (x0, y0, x1, y1), s, m in zip(boxes.tolist(), scores.tolist(), masks):
            rle = mask_utils.encode(np.asfortranarray(m.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode()
            out.append({"image_id": im["id"], "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s,
                        "segmentation": rle})
    path = prediction_path(key, seed, split)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"{MODELS[key]['name']} seed {seed} {split}: {len(out)} detections on {len(images)} patches -> {path}")


def main():
    p = argparse.ArgumentParser(description="Predict masks with the Mask R-CNN R101 runs on the val and test patches.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--models", nargs="+", default=["maskrcnn"], choices=list(VARIANTS))
    p.add_argument("--splits", nargs="+", default=SPLITS)
    args = p.parse_args()
    os.makedirs(PRED_DIR, exist_ok=True)
    for key in args.models:
        for seed in args.seeds:
            for split in args.splits:
                predict(seed, split, key)


if __name__ == "__main__":
    main()
