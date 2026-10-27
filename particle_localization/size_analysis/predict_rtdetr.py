import argparse
import json
import os

from config import IMGSZ, MAX_DET, MODELS, PATCHES, PRED_CONF, PRED_DIR, SEEDS, SPLITS, prediction_path


def predict(seed, split):
    from ultralytics import RTDETR
    model = RTDETR(MODELS["rtdetr"]["weights"][seed])
    with open(os.path.join(PATCHES, split, "annotations.json")) as fh:
        images = json.load(fh)["images"]
    out = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        paths = [os.path.join(PATCHES, split, "images", im["file_name"]) for im in batch]
        for im, r in zip(batch, model.predict(paths, imgsz=IMGSZ, conf=PRED_CONF, max_det=MAX_DET, device=0, verbose=False)):
            for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist()):
                out.append({"image_id": im["id"], "category_id": 1, "bbox": [x0, y0, x1 - x0, y1 - y0], "score": s})
    path = prediction_path("rtdetr", seed, split)
    with open(path, "w") as fh:
        json.dump(out, fh)
    print(f"RT-DETR-L seed {seed} {split}: {len(out)} detections on {len(images)} patches -> {path}")


def main():
    p = argparse.ArgumentParser(description="Predict boxes with the polymer-agnostic RT-DETR-L runs on the val and test patches.")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()
    os.makedirs(PRED_DIR, exist_ok=True)
    for seed in args.seeds:
        for split in SPLITS:
            predict(seed, split)


if __name__ == "__main__":
    main()
