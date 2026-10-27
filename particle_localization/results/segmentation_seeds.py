import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "segmentation_seeds.md")


def mask_rcnn(variant):
    rows = []
    for d in [variant] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(REPO, "MaskRCNN", "zoo", "current", f"{variant}_seed*")))]:
        f = os.path.join(REPO, "MaskRCNN", "zoo", "current", d, "results", "test_metrics.json")
        if os.path.exists(f):
            s = json.load(open(f))["test"]["segm"]
            rows.append((s["AP50"], s["AP75"], s["AP"]))
    return rows


def yolo(project, name):
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, project, "runs", "segment", f"{name}*", "test_metrics.json"))):
        m = json.load(open(f))["mask"]
        rows.append((100 * m["map50"], 100 * m["map75"], 100 * m["map"]))
    return rows


def fmt(rows):
    a = np.array(rows)
    if len(a) == 1:
        return [f"{v:.1f}" for v in a[0]]
    return [f"{m:.1f} ± {s:.1f}" for m, s in zip(a.mean(0), a.std(0, ddof=1))]


models = [("Mask R-CNN", "ResNet 101", mask_rcnn("R101-FPN")), ("", "ResNeXt 101", mask_rcnn("X101-FPN")),
          ("YOLO Segmentation", "yolov8n-seg", yolo("YOLOv8-seg", "yolov8n-seg_patches")),
          ("", "yolov26n-seg", yolo("YOLOv26-seg", "yolo26n-seg_patches"))]
lines = ["Mask AP (%) on the 260 test patches, mean ± SD over training seeds", "",
         "| Model | Backbone | AP50 | AP75 | AP | Runs |", "|---|---|---|---|---|---|"]
for model, backbone, rows in models:
    if rows:
        lines.append(f"| {model} | {backbone} | " + " | ".join(fmt(rows)) + f" | {len(rows)} |")
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
