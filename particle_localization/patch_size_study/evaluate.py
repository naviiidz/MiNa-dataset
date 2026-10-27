import argparse
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Polygon, box
from shapely.ops import unary_union
from torchvision.ops import batched_nms

from config import (CONF, DATA_PROCESSING, FULL_IMAGES, IMGSZ, MAX_DET_PER_IMAGE, MERGE_IOU, OUTPUT_DIR, PATCH_SIZES,
                    POLYMERS, SEEDS, size_paths)

sys.path.insert(0, DATA_PROCESSING)
from create_patches import EXCLUDED_IMAGES, MIN_ANNOTATION_AREA, SCALE_BAR_HEIGHT, patch_starts

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def split_images(size, split):
    names = []
    for p in POLYMERS:
        path = os.path.join(size_paths(size)["coco"], f"patches_256_{p.lower()}", "split_source_images.json")
        with open(path) as fh:
            names += [(p, n) for n in json.load(fh)[split] if n not in EXCLUDED_IMAGES]
    return names


def ground_truth(polymer, name, height):
    with open(os.path.join(FULL_IMAGES, polymer, f"{polymer.lower()}.json")) as fh:
        coco = json.load(fh)
    image_id = next(i["id"] for i in coco["images"] if i["file_name"] == name)
    width = next(i["width"] for i in coco["images"] if i["id"] == image_id)
    area = box(0, 0, width, height)
    boxes = []
    for a in coco["annotations"]:
        if a["image_id"] != image_id:
            continue
        seg = json.loads(a["segmentation"]) if isinstance(a["segmentation"], str) else a["segmentation"]
        if not isinstance(seg, list):
            continue
        shapes = [Polygon(np.array(s, dtype=float).reshape(-1, 2)).buffer(0) for s in seg if isinstance(s, list) and len(s) >= 6]
        if not shapes:
            continue
        shape = unary_union(shapes).intersection(area)
        if shape.is_empty or shape.area < MIN_ANNOTATION_AREA:
            continue
        boxes.append(shape.bounds)
    return np.array(boxes, dtype=float).reshape(-1, 4), POLYMERS.index(polymer)


def predict_full_image(model, image, size):
    height = image.shape[0] - SCALE_BAR_HEIGHT
    tiles = [(x, y) for y in patch_starts(height, size) for x in patch_starts(image.shape[1], size)]
    crops = [image[y:y + size, x:x + size] for x, y in tiles]
    boxes, scores, classes = [], [], []
    for i in range(0, len(crops), 32):
        for (x, y), r in zip(tiles[i:i + 32], model.predict(crops[i:i + 32], imgsz=IMGSZ, conf=CONF, max_det=300,
                                                             device=0, verbose=False)):
            b = r.boxes.xyxy.cpu().numpy() + np.array([x, y, x, y])
            boxes.append(b), scores.append(r.boxes.conf.cpu().numpy()), classes.append(r.boxes.cls.cpu().numpy().astype(int))
    boxes, scores, classes = np.concatenate(boxes).astype(np.float32), np.concatenate(scores).astype(np.float32), np.concatenate(classes)
    keep = batched_nms(torch.tensor(boxes), torch.tensor(scores), torch.tensor(classes), MERGE_IOU).numpy()[:MAX_DET_PER_IMAGE]
    return boxes[keep], scores[keep], classes[keep], height


def iou_matrix(a, b):
    x0, y0 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x1, y1 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def correct_matrix(gt_boxes, gt_cls, pred_boxes, pred_cls):
    correct = np.zeros((len(pred_boxes), len(IOU_THRESHOLDS)), dtype=bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return correct
    iou = iou_matrix(gt_boxes, pred_boxes) * (gt_cls[:, None] == pred_cls[None, :])
    for i, t in enumerate(IOU_THRESHOLDS):
        matches = np.array(np.nonzero(iou >= t)).T
        if len(matches):
            matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1], i] = True
    return correct


def scores(records, agnostic):
    from ultralytics.utils.metrics import ap_per_class
    tp, conf, pcls, tcls = [], [], [], []
    for gt_boxes, gt_c, boxes, s, c in records:
        g, p = (np.zeros_like(gt_c), np.zeros_like(c)) if agnostic else (gt_c, c)
        tp.append(correct_matrix(gt_boxes, g, boxes, p)), conf.append(s), pcls.append(p), tcls.append(g)
    out = ap_per_class(np.concatenate(tp), np.concatenate(conf), np.concatenate(pcls), np.concatenate(tcls))
    p, r, ap, classes = out[2], out[3], out[5], out[6]
    row = {"mAP50": 100 * ap[:, 0].mean(), "mAP50-95": 100 * ap.mean(), "precision": 100 * p.mean(), "recall": 100 * r.mean()}
    if not agnostic:
        row.update({f"AP50_{POLYMERS[int(k)]}": 100 * ap[i, 0] for i, k in enumerate(classes)})
    return row


def evaluate(size, seed, splits=("val", "test")):
    from ultralytics import YOLO
    model = YOLO(size_paths(size, seed)["weights"])
    rows = []
    for split in splits:
        records = []
        for polymer, name in split_images(size, split):
            image = cv2.imread(os.path.join(FULL_IMAGES, polymer, name))
            boxes, s, c, height = predict_full_image(model, image, size)
            gt_boxes, cls = ground_truth(polymer, name, height)
            records.append((gt_boxes, np.full(len(gt_boxes), cls), boxes, s, c))
        n_images, n_particles = len(records), sum(len(r[0]) for r in records)
        for agnostic in (False, True):
            rows.append({"patch_size": size, "seed": seed, "split": split, "classes": "agnostic" if agnostic else "4-class",
                         "images": n_images, "particles": n_particles, **scores(records, agnostic)})
        print(f"patch {size} seed {seed} {split}: {n_images} images, {n_particles} particles, "
              f"mAP50 {rows[-2]['mAP50']:.1f} (agnostic {rows[-1]['mAP50']:.1f})", flush=True)
    return rows


def summarize(results):
    metrics = [c for c in results.columns if c.startswith(("mAP", "precision", "recall", "AP50"))]
    g = results.groupby(["patch_size", "split", "classes"])[metrics]
    mean, sd = g.mean(), g.std(ddof=1)
    out = mean.round(1).astype(str) + " ± " + sd.round(1).astype(str)
    return out.reset_index(), mean.reset_index(), sd.reset_index()


def plot(mean, sd, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, classes in zip(axes, ("4-class", "agnostic")):
        for split, style in (("val", "o-"), ("test", "s--")):
            sel = (mean["classes"] == classes) & (mean["split"] == split)
            m, s = mean[sel].sort_values("patch_size"), sd[sel].sort_values("patch_size")
            ax.errorbar(m["patch_size"], m["mAP50"], yerr=s["mAP50"], fmt=style, capsize=3, label=f"{split} mAP50")
            ax.errorbar(m["patch_size"], m["mAP50-95"], yerr=s["mAP50-95"], fmt=style, capsize=3, alpha=0.5,
                        label=f"{split} mAP50-95")
        ax.set_xscale("log", base=2)
        ax.set_xticks(PATCH_SIZES)
        ax.set_xticklabels(PATCH_SIZES)
        ax.set_xlabel("Patch size (px)")
        ax.set_title(f"YOLO26s, full-image evaluation ({classes}), mean ± SD over seeds", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mAP (%)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=200)


def main():
    p = argparse.ArgumentParser(description="Tile full val/test images at each patch size, merge the detections and "
                                            "score all patch sizes on the same full images.")
    p.add_argument("--sizes", type=int, nargs="+", default=PATCH_SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    args = p.parse_args()
    rows = []
    for size in args.sizes:
        for seed in args.seeds:
            if os.path.exists(size_paths(size, seed)["weights"]):
                rows += evaluate(size, seed)
    results = pd.DataFrame(rows).round(1)
    table, mean, sd = summarize(results)
    out = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(out, exist_ok=True)
    results.to_csv(os.path.join(out, "full_image_metrics_per_seed.csv"), index=False)
    table.to_csv(os.path.join(out, "full_image_metrics.csv"), index=False)
    plot(mean, sd, os.path.join(out, "patch_size.png"))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
