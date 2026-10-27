import argparse
import json
import os

import numpy as np

from config import IMGSZ, MAX_DET, MODELS, PATCHES, PRED_CONF, PRED_DIR, SEEDS, SPLITS, prediction_path


def uncompressed_rle(m):
    flat = np.concatenate([[0], m.T.ravel().astype(np.uint8), [0]])
    change = np.flatnonzero(np.diff(flat)) 
    runs = np.diff(np.concatenate([[0], change, [m.size]]))
    return {"size": list(m.shape), "counts": runs.tolist()}


def predict(key, seed, split):
    from ultralytics import YOLO
    model = YOLO(MODELS[key]["weights"][seed])
    with open(os.path.join(PATCHES, split, "annotations.json")) as fh:
        images = json.load(fh)["images"]
    out = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        paths = [os.path.join(PATCHES, split, "images", im["file_name"]) for im in batch]
        for im, r in zip(batch, model.predict(paths, imgsz=IMGSZ, conf=PRED_CONF, max_det=MAX_DET, retina_masks=True,
                                              device=0, verbose=False)):
            if r.masks is None:
                continue
            masks = r.masks.data.cpu().numpy() > 0.5
            for (x0, y0, x1, y1), s, m in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist(), masks):
                rle = uncompressed_rle(m)
                out.append({"image_id": im["id"], "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s,
                            "segmentation": rle})
    path = prediction_path(key, seed, split)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"{MODELS[key]['name']} seed {seed} {split}: {len(out)} detections on {len(images)} patches -> {path}")


def main():
    p = argparse.ArgumentParser(description="Predict masks with the YOLO segmentation runs on the val and test patches.")
    p.add_argument("--models", nargs="+", default=["yolov8seg", "yolo26seg"], choices=["yolov8seg", "yolo26seg"])
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--splits", nargs="+", default=SPLITS)
    args = p.parse_args()
    os.makedirs(PRED_DIR, exist_ok=True)
    for key in args.models:
        for seed in args.seeds:
            for split in args.splits:
                predict(key, seed, split)


if __name__ == "__main__":
    main()
