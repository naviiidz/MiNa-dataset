import argparse
import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from config import COCO_MAX_DETS, MODELS, RESULTS_DIR, SEEDS, SIZE_CATEGORIES, prediction_path
from evaluate import coco_ap, load_dets, load_gt, prepare

SEG_MODELS = ["maskrcnn", "maskrcnn_x101", "yolov8seg", "yolo26seg"]
LABELS = {"maskrcnn": "Mask R-CNN (ResNet-101)", "maskrcnn_x101": "Mask R-CNN (ResNeXt-101)",
          "yolov8seg": "YOLOv8n-seg", "yolo26seg": "YOLO26n-seg"}
COLORS = ["#0072BD", "#D95319", "#EDB120"]


def max_feret(rle):
    m = mask_utils.decode(rle).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    pts = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return 0.0
    d = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def evaluate(gt, image_ids):
    gt = gt.assign(feret_px=[max_feret(r) for r in gt["rle"]])
    rows = []
    for key in SEG_MODELS:
        for seed in SEEDS:
            if not os.path.exists(prediction_path(key, seed, "test")):
                continue
            data = prepare(gt, image_ids, load_dets(key, seed, "test"), "mask", {i: 1.0 for i in image_ids})
            for d in data.values():
                top = d["dt_masks"][:COCO_MAX_DETS]
                d["dt_feret"] = np.array([max_feret(m) for m in top] + [0.0] * (len(d["dt_masks"]) - len(top)))
            for name, (lo, hi) in [("all", (0, 1e12))] + list(SIZE_CATEGORIES.items()):
                in_gt = ((gt["feret_px"] >= lo) & (gt["feret_px"] < hi)).to_numpy()
                ap50, ap = coco_ap(data, gt, in_gt, lambda d, lo=lo, hi=hi: (d["dt_feret"] >= lo) & (d["dt_feret"] < hi))
                rows.append({"model": key, "seed": seed, "size": name, "lo_px": lo, "hi_px": hi,
                             "n_particles": int((in_gt & ~gt["ignore"].to_numpy()).sum()), "AP50": ap50, "AP": ap})
            print(f"{MODELS[key]['name']} seed {seed}: " + ", ".join(f"{r['size']} AP {r['AP']:.1f}" for r in rows[-4:]), flush=True)
    return pd.DataFrame(rows)


def plot(summary, out):
    sizes = list(SIZE_CATEGORIES)
    names = {"small": f"Small (< {SIZE_CATEGORIES['small'][1]:g} px)",
             "medium": f"Medium ({SIZE_CATEGORIES['medium'][0]:g}–{SIZE_CATEGORIES['medium'][1]:g} px)",
             "large": f"Large (≥ {SIZE_CATEGORIES['large'][0]:g} px)"}
    fig, ax = plt.subplots(figsize=(9, 4.6))
    w = 0.22
    for k, size in enumerate(sizes):
        s = summary[summary["size"] == size].set_index("model").loc[SEG_MODELS]
        ax.bar(np.arange(len(SEG_MODELS)) + (k - 1) * w, s["AP_mean"], w, yerr=s["AP_sd"], capsize=3,
               color=COLORS[k], edgecolor="black", linewidth=0.6, label=names[size])
    ax.set_xticks(range(len(SEG_MODELS)))
    ax.set_xticklabels([LABELS[m] for m in SEG_MODELS], rotation=20, ha="right")
    ax.set_ylabel("Mask AP (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=300)
    plt.close(fig)


def main():
    argparse.ArgumentParser(description="Mask AP of the segmentation models per particle size category "
                                        "(maximum Feret diameter in pixels) on the test patches.").parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    gt, image_ids, _ = load_gt("test")
    per_seed = evaluate(gt, image_ids)
    per_seed.to_csv(os.path.join(RESULTS_DIR, "size_categories_per_seed.csv"), index=False)
    g = per_seed.groupby(["model", "size"], sort=False)
    summary = g[["AP50", "AP"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in summary.columns]
    summary = summary.join(g["n_particles"].first()).join(g["seed"].count().rename("runs")).reset_index()
    summary.to_csv(os.path.join(RESULTS_DIR, "size_categories.csv"), index=False)
    plot(summary, os.path.join(RESULTS_DIR, "size_categories"))
    print(summary.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
