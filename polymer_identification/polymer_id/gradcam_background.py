import argparse
import json
import math
import os
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import classifiers
from .config import CANVAS, FEATURES_CSV, FEATURES_DIR, FOLDS, FULL_IMAGES, IMAGE_SIZE, NORM_CENTER, NORM_SCALE, POLYMERS, RESULTS_DIR, particle_id
from .features import SCALE_BAR_HEIGHT, parse_segmentation, particle_mask
from .images import particle_mask_on_canvas
from .interpretability import LABELS, MODEL_NAMES, cam_shares, gradcam, load_fold, macro_f1, overlay, probs

warnings.filterwarnings("ignore")

OUT_DIR = os.path.join(RESULTS_DIR, "gradcam_background")


def load_annotations():
    anns = {}
    for polymer in POLYMERS:
        with open(os.path.join(FULL_IMAGES, polymer, f"{polymer.lower()}.json")) as fh:
            coco = json.load(fh)
        names = {i["id"]: i["file_name"] for i in coco["images"]}
        for a in coco["annotations"]:
            anns[particle_id(polymer, names[a["image_id"]], a["id"])] = a
    return anns


class Images:
    def __init__(self):
        self.cache = {}

    def get(self, polymer, name):
        key = (polymer, name)
        if key not in self.cache:
            if len(self.cache) > 8:
                self.cache.clear()
            grey = cv2.imread(os.path.join(FULL_IMAGES, polymer, name), cv2.IMREAD_GRAYSCALE)
            self.cache[key] = grey[:grey.shape[0] - SCALE_BAR_HEIGHT]
        return self.cache[key]


def context_image(grey, ann):
    m, x0, y0 = particle_mask(parse_segmentation(ann["segmentation"]), grey.shape, grey.shape[0])
    ys, xs = np.nonzero(m)
    top, left = y0 + ys.min(), x0 + xs.min()
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    limit = int(CANVAS / math.sqrt(2))
    s = limit / max(h, w) if max(h, w) > limit else 1.0
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    side = CANVAS / s
    wy0 = top - ((CANVAS - nh) // 2) / s
    wx0 = left - ((CANVAS - nw) // 2) / s
    fill = float(np.median(grey))
    padded = cv2.copyMakeBorder(grey, int(side) + 2, int(side) + 2, int(side) + 2, int(side) + 2,
                                cv2.BORDER_CONSTANT, value=fill)
    py, px = int(round(wy0)) + int(side) + 2, int(round(wx0)) + int(side) + 2
    n = int(round(side))
    window = padded[py:py + n, px:px + n]
    window = cv2.resize(window, (CANVAS, CANVAS), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    full_mask = np.zeros(grey.shape, np.uint8)
    full_mask[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    mpad = cv2.copyMakeBorder(full_mask, int(side) + 2, int(side) + 2, int(side) + 2, int(side) + 2, cv2.BORDER_CONSTANT, value=0)
    mwin = cv2.resize(mpad[py:py + n, px:px + n], (CANVAS, CANVAS), interpolation=cv2.INTER_NEAREST)
    inside = window[mwin > 0].astype(np.float64)
    if inside.size == 0:
        inside = window.astype(np.float64).ravel()
    z = (window.astype(np.float64) - inside.mean()) / max(inside.std(), 1.0)
    out = np.clip(NORM_CENTER + NORM_SCALE * z, 0, 255).astype(np.uint8)
    return cv2.resize(out, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


def context_images(part, anns, images):
    return np.stack([context_image(images.get(p, s), anns[pid])
                     for pid, p, s in zip(part["particle_id"], part["polymer"], part["particle_id"].str.split("|").str[1])])


def figure(model_name, masked, context, masks, cams_m, cams_c, yt, pm, pc, path, n=3):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(POLYMERS) * n, 4, figsize=(4 * 1.4, len(POLYMERS) * n * 1.45 + 0.5))
    titles = ["Background removed", "Grad-CAM", "Background kept", "Grad-CAM"]
    for r, c in enumerate(LABELS):
        idx = rng.choice(np.flatnonzero(yt == c), n, replace=False)
        for k, i in enumerate(idx):
            row = axes[r * n + k]
            row[0].imshow(masked[i], cmap="gray", vmin=0, vmax=255)
            row[1].imshow(cv2.cvtColor(overlay(masked[i], cams_m[i], masks[i]), cv2.COLOR_BGR2RGB))
            row[2].imshow(context[i], cmap="gray", vmin=0, vmax=255)
            row[3].imshow(cv2.cvtColor(overlay(context[i], cams_c[i], masks[i]), cv2.COLOR_BGR2RGB))
            for a in row:
                a.set_xticks([])
                a.set_yticks([])
            row[0].set_ylabel(POLYMERS[c], fontsize=8)
            for a, p in ((row[1], pm[i]), (row[3], pc[i])):
                a.set_title(f"→ {POLYMERS[p]}", fontsize=7, loc="left", color="green" if p == c else "red", pad=2)
    for a, t in zip(axes[0], titles):
        a.text(0.5, 1.25, t, fontsize=7, ha="center", transform=a.transAxes)
    fig.tight_layout(h_pad=0.4, w_pad=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM on test particles shown with their real background "
                                                 "(membrane pores and neighbouring particles) instead of a uniform canvas.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    args = parser.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    feats = pd.read_csv(FEATURES_CSV)
    feats["particle_id"] = [particle_id(p, s, a) for p, s, a in zip(feats["polymer"], feats["source_image"], feats["annotation_id"])]
    feats = feats.set_index("particle_id")
    anns, images = load_annotations(), Images()

    score_rows, share_rows, pred_rows = [], [], []
    for fold in args.folds:
        _, parts, imgs, masks = load_fold(fold, feats)
        part, masked, msk = parts["test"], imgs["test"], masks["test"]
        context = context_images(part, anns, images)
        check = context.copy()
        check[msk == 0] = NORM_CENTER
        diff = np.abs(check.astype(int) - masked.astype(int)).mean()
        print(f"fold {fold}: {len(part)} test particles; mean |difference| inside particle vs. training image {diff:.2f} grey levels")
        if fold == args.folds[0]:
            np.save(os.path.join(OUT_DIR, f"context_fold{fold}.npy"), context[:16])
        y = part["label"].to_numpy()
        for model_name in args.models:
            model, mean, std = classifiers.load_trained(model_name, fold)
            pm = probs(model, mean, std, masked).argmax(1)
            pc = probs(model, mean, std, context).argmax(1)
            cams_m = gradcam(model, model_name, mean, std, masked, pm)
            cams_c = gradcam(model, model_name, mean, std, context, pc)
            for cond, pred, cams in (("background_removed", pm, cams_m), ("background_kept", pc, cams_c)):
                score_rows.append({"model": model_name, "fold": fold, "condition": cond, "macro_f1": macro_f1(y, pred),
                                   "accuracy": 100 * np.mean(pred == y)})
                sh = cam_shares(cams, msk)
                sh["model"], sh["fold"], sh["condition"] = model_name, fold, cond
                sh["polymer"], sh["correct"] = part["polymer"].to_numpy(), pred == y
                share_rows.append(sh)
            pred_rows.append(pd.DataFrame({"model": model_name, "fold": fold, "particle_id": part["particle_id"],
                                           "true": y, "pred_background_removed": pm, "pred_background_kept": pc}))
            if fold == args.folds[0]:
                figure(model_name, masked, context, msk, cams_m, cams_c, y, pm, pc,
                       os.path.join(OUT_DIR, f"gradcam_background_{model_name}.png"))
            print(f"  {MODEL_NAMES[model_name]}: macro F1 {macro_f1(y, pm):.1f} (background removed) -> "
                  f"{macro_f1(y, pc):.1f} (background kept)", flush=True)

    scores = pd.DataFrame(score_rows)
    shares = pd.concat(share_rows, ignore_index=True)
    preds = pd.concat(pred_rows, ignore_index=True)
    scores.to_csv(os.path.join(OUT_DIR, "scores.csv"), index=False)
    shares.to_csv(os.path.join(OUT_DIR, "cam_shares.csv"), index=False)
    preds.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    summary = scores.groupby(["model", "condition"])[["macro_f1", "accuracy"]].agg(["mean", "std"]).round(1)
    sh = shares.groupby(["model", "condition"]).agg(interior_median=("interior", "median"), boundary_median=("boundary", "median"),
                                                     background_median=("background", "median"),
                                                     background_above_average=("background", lambda v: 100 * np.mean(v > 1)))
    flips = preds.assign(changed=preds["pred_background_removed"] != preds["pred_background_kept"]).groupby("model")["changed"].mean() * 100
    lines = ["# Grad-CAM with the real background kept", "",
             "Test particles were shown to the trained classifiers with the membrane background and neighbouring particles "
             "around them, instead of the uniform canvas used in training. Both versions use the same framing and the same "
             "per-particle normalisation.", "", "## Classification (mean and SD over folds)", "", summary.to_string(), "",
             "## Grad-CAM attention per unit area relative to the image average (medians over particles)", "",
             sh.round(2).to_string(), "", "## Predictions that changed when the background was kept (%)", "",
             flips.round(1).to_frame("changed_%").to_string()]
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
