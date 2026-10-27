import glob
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "localization_seeds.md")


def detectron(project, variant, task):
    rows = []
    root = os.path.join(REPO, project, "zoo", "current")
    for d in [variant] + [os.path.basename(p) for p in sorted(glob.glob(os.path.join(root, f"{variant}_seed*")))]:
        f = os.path.join(root, d, "results", "test_metrics.json")
        if os.path.exists(f):
            s = json.load(open(f))["test"][task]
            rows.append((s["AP50"], s["AP75"], s["AP"]))
    return rows


def ultralytics(pattern, key):
    rows = []
    for f in sorted(glob.glob(os.path.join(REPO, pattern, "test_metrics.json"))):
        m = json.load(open(f))
        m = m["class_agnostic"] if key == "class_agnostic" else m[key]
        rows.append((100 * m["map50"], 100 * m["map75"], 100 * m["map"]))
    return rows


def fmt(rows):
    a = np.array(rows)
    if len(a) == 1:
        return [f"{v:.1f}" for v in a[0]]
    return [f"{m:.1f} ± {s:.1f}" for m, s in zip(a.mean(0), a.std(0, ddof=1))]


detection = [("Faster R-CNN", "ResNet 101", detectron("FasterRCNN", "FasterRCNN-R101_agnostic", "bbox")),
             ("", "ResNeXt 101", detectron("FasterRCNN", "FasterRCNN-X101_agnostic", "bbox")),
             ("RT-DETR", "RT-DETR-L", ultralytics("RT-DETR/runs/rtdetr-l/rtdetr-l_agnostic_seed*", "class_agnostic")),
             ("YOLO", "yolo26s", ultralytics("YOLOv26-detection/runs/yolo26s/yolo26s_agnostic_seed*", "box"))]
segmentation = [("Mask R-CNN", "ResNet 101", detectron("MaskRCNN", "R101-FPN", "bbox")),
                ("", "ResNeXt 101", detectron("MaskRCNN", "X101-FPN", "bbox")),
                ("YOLO Segmentation", "yolov8n-seg", ultralytics("YOLOv8-seg/runs/segment/yolov8n-seg_patches*", "box")),
                ("", "yolov26n-seg", ultralytics("YOLOv26-seg/runs/segment/yolo26n-seg_patches*", "box"))]
lines = ["Polymer-agnostic (1-class) models: box AP (%) on the 260 test patches, mean ± SD over training seeds", "",
         "| Task | Model | Backbone | Box AP50 | Box AP75 | Box AP | Runs |", "|---|---|---|---|---|---|---|"]
for task, models in (("Detection", detection), ("Segmentation", segmentation)):
    for model, backbone, rows in models:
        if rows:
            lines.append(f"| {task} | {model} | {backbone} | " + " | ".join(fmt(rows)) + f" | {len(rows)} |")
            task = ""
open(OUT, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
