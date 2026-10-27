#!/usr/bin/env python3
"""
Collect the test-set metrics of every model trained on the patch split into one table.

Reads the metrics files the training scripts write:
    FasterRCNN/zoo/current/<variant>/results/test_metrics.json         (train.py)
    FasterRCNN/zoo/current/<variant>/evaluation_test/detailed_metrics.json  (evaluate.py, P/R/F1)
    MaskRCNN/zoo/current/<variant>/results/test_metrics.json
    MaskRCNN/zoo/current/<variant>/evaluation_test/detailed_metrics.json    (evaluate_improved_model.py)
    RT-DETR/runs/rtdetr-l/*/test_metrics.json                      (train.py / evaluate.py)
    YOLOv26-detection/runs/<model>/mnp_<model>*/test_metrics.json   (quick_train.py)
    YOLOv8-seg/runs/segment/*/test_metrics.json                     (train.py)
    YOLOv26-seg/runs/segment/*/test_metrics.json                    (quick_train.py)

Writes results/test_metrics.md. All AP values are on a 0-100 scale.

Usage: python results/collect_metrics.py
"""

import csv
import glob
import json
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "test_metrics.md")


def load(path):
    with open(path) as f:
        return json.load(f)


def fmt(v):
    return "–" if v is None else f"{v:.1f}"


def detectron2_time(model_dir):
    """'Total training time: 0:41:22' from the run's train.log (stdout) or detectron2's log.txt."""
    for name in ("train.log", "log.txt"):
        log = os.path.join(model_dir, name)
        if os.path.exists(log):
            m = re.findall(r"Total training time: ([0-9:]+)", open(log, errors="ignore").read())
            if m:
                return m[-1]
    return "–"


def ultralytics_time(run_dir):
    """Last value of the cumulative 'time' column in results.csv."""
    path = os.path.join(run_dir, "results.csv")
    if os.path.exists(path):
        rows = list(csv.DictReader(open(path)))
        if rows and "time" in rows[-1]:
            s = int(float(rows[-1]["time"]))
            return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"
    return "–"


def rcnn_rows(project, task, classes):
    rows = []
    for metrics_file in sorted(glob.glob(os.path.join(REPO, project, "zoo", "current", "*", "results", "test_metrics.json"))):
        model_dir = os.path.dirname(os.path.dirname(metrics_file))
        if "_dryrun" in model_dir:
            continue
        m = load(metrics_file)["test"]
        detailed = os.path.join(model_dir, "evaluation_test", "detailed_metrics.json")
        prf = load(detailed) if os.path.exists(detailed) else {}
        box, mask = m.get("bbox", {}), m.get("segm", {})
        rows.append({
            "model": f"{project} {os.path.basename(model_dir)}", "tool": "detectron2", "task": task, "classes": classes,
            "images": m.get("num_images"),
            "box": (box.get("AP"), box.get("AP50"), box.get("AP75")),
            "mask": (mask.get("AP"), mask.get("AP50"), mask.get("AP75")) if mask else (None, None, None),
            "prf": tuple(100 * prf[k] if k in prf else None for k in ("precision", "recall", "f1_score")),
            "time": detectron2_time(model_dir), "weights": os.path.relpath(os.path.join(model_dir, "model_final.pth"), REPO),
        })
    return rows


def ultralytics_rows(pattern, task, classes):
    rows = []
    for metrics_file in sorted(glob.glob(os.path.join(REPO, pattern))):
        run_dir = os.path.dirname(metrics_file)
        m = load(metrics_file)
        if "masked" in os.path.basename(m.get("data", "")):
            continue  # GT-masked morphology experiments are diagnostics, not benchmark rows
        box, mask = m["box"], m.get("mask")
        p, r = 100 * (mask or box)["mp"], 100 * (mask or box)["mr"]
        rows.append({
            "model": os.path.relpath(run_dir, REPO).split("/runs/")[0] + " " + os.path.basename(run_dir),
            "tool": "ultralytics", "task": task, "classes": classes, "images": m.get("num_images"),
            "box": tuple(100 * box[k] for k in ("map", "map50", "map75")),
            "mask": tuple(100 * mask[k] for k in ("map", "map50", "map75")) if mask else (None, None, None),
            "prf": (p, r, 2 * p * r / (p + r) if p + r else 0.0),
            "time": ultralytics_time(run_dir), "weights": os.path.relpath(m["weights"], REPO),
        })
    return rows


def main():
    rows = (rcnn_rows("FasterRCNN", "detection", 4)
            + ultralytics_rows("RT-DETR/runs/rtdetr-l/*/test_metrics.json", "detection", 4)
            + ultralytics_rows("YOLOv26-detection/runs/*/mnp_*/test_metrics.json", "detection", 4)
            + rcnn_rows("MaskRCNN", "segmentation", 1)
            + ultralytics_rows("YOLOv8-seg/runs/segment/*/test_metrics.json", "segmentation", 1)
            + ultralytics_rows("YOLOv26-seg/runs/segment/*/test_metrics.json", "segmentation", 1))

    lines = [
        "# Test-set metrics on the patch split",
        "",
        "Patch split: 1462 / 260 / 260 patches, split by source image (no leakage). Patches cover the whole image",
        "area above the SEM info bar (5x4 grid per image, 64 px row overlap). PE patches with particles are",
        "repeated 3x in train only. Every model is",
        "scored on the same 260 test patches, including patches without particles. AP values are on a 0–100 scale.",
        "",
        "| Model | Tool | Task | Classes | Test imgs | Box AP | AP50 | AP75 | Mask AP | AP50 | AP75 | P | R | F1 | Train time | Weights |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], r["tool"], r["task"], str(r["classes"]), str(r["images"]),
            *map(fmt, r["box"]), *map(fmt, r["mask"]), *map(fmt, r["prf"]), r["time"], f"`{r['weights']}`",
        ]) + " |")
    lines += [
        "",
        "## How to read this table",
        "- **AP**: detectron2 reports COCO AP (pycocotools, 101-point interpolation, score threshold 0.05).",
        "  Ultralytics reports its own mAP (its own interpolation, conf 0.001). The two are close but not",
        "  identical. For a strict comparison, re-score the ultralytics predictions with pycocotools.",
        "- **P / R / F1**: detectron2 rows use a fixed score threshold of 0.25, with greedy matching at",
        "  IoU 0.5 (`evaluate.py` / `evaluate_improved_model.py`). Ultralytics rows use its mean",
        "  precision/recall at the max-F1 confidence. The mask values are used for seg models.",
        "- **Classes**: detection models predict 4 polymers (PP, PE, PS, PET). Segmentation models predict one",
        "  class (Microplastic).",
        "- **YOLO detection labels**: ultralytics drops 11 of 14,425 boxes that are exact duplicates of another",
        "  box in the same patch (different polygons, same bbox).",
    ]
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(rows)} models -> {OUT}")


if __name__ == "__main__":
    main()
