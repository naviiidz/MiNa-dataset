import math
import os
import re
import shutil

import cv2
import numpy as np
import pandas as pd

from .config import CANVAS, FEATURES_DIR, IMAGE_SIZE, NORM_CENTER, NORM_SCALE, fold_paths, parser


def normalize(crop, mask):
    p = crop[mask > 0].astype(np.float64)
    z = (crop.astype(np.float64) - p.mean()) / max(p.std(), 1.0)
    out = np.clip(NORM_CENTER + NORM_SCALE * z, 0, 255).astype(np.uint8)
    out[mask == 0] = NORM_CENTER
    return out


def fit_to_canvas(mask, crop=None):
    ys, xs = np.nonzero(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = crop[ys.min():ys.max() + 1, xs.min():xs.max() + 1] if crop is not None else None
    h, w = mask.shape
    limit = int(CANVAS / math.sqrt(2))
    shrunk = max(h, w) > limit
    if shrunk:
        s = limit / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA) if crop is not None else None
        h, w = nh, nw
    return mask, crop, (CANVAS - h) // 2, (CANVAS - w) // 2, shrunk


def particle_image(crop, mask):
    mask, crop, oy, ox, shrunk = fit_to_canvas(mask, crop)
    h, w = mask.shape
    crop = normalize(crop, mask)
    img = np.full((CANVAS, CANVAS), NORM_CENTER, dtype=np.uint8)
    region = img[oy:oy + h, ox:ox + w]
    region[mask > 0] = crop[mask > 0]
    return cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST), shrunk


def particle_mask_on_canvas(mask):
    mask, _, oy, ox, _ = fit_to_canvas(mask)
    out = np.zeros((CANVAS, CANVAS), np.uint8)
    out[oy:oy + mask.shape[0], ox:ox + mask.shape[1]] = (mask > 0).astype(np.uint8)
    return cv2.resize(out, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)


def safe_name(pid):
    polymer, source, ann = pid.split("|")
    return re.sub(r"[^A-Za-z0-9_-]+", "_", f"{polymer}__{os.path.splitext(source)[0]}__{ann}") + ".png"


def make_particle_images(fold):
    paths = fold_paths(fold)
    pool = pd.read_csv(os.path.join(paths["splits"], "balanced.csv"))
    root = paths["images"]
    if os.path.isdir(root):
        shutil.rmtree(root)
    files, shrunk = [], []
    for pid, split, polymer, c, m in zip(pool["particle_id"], pool["split"], pool["polymer"], pool["crop"], pool["mask"]):
        crop = cv2.imread(os.path.join(FEATURES_DIR, c), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(FEATURES_DIR, m), cv2.IMREAD_GRAYSCALE)
        img, was_shrunk = particle_image(crop, mask)
        rel = os.path.join(split, polymer, safe_name(pid))
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        cv2.imwrite(os.path.join(root, rel), img)
        files.append(rel)
        shrunk.append(int(was_shrunk))
    pool["image"], pool["shrunk"] = files, shrunk
    pool[["particle_id", "split", "polymer", "label", "image", "shrunk"]].to_csv(
        os.path.join(root, "index.csv"), index=False)
    return len(pool), sum(shrunk)


def main():
    args = parser("Render each balanced particle as a normalised single-particle image.").parse_args()
    for fold in args.folds:
        n, shrunk = make_particle_images(fold)
        print(f"fold {fold}: {n} images ({shrunk} large particles shrunk to fit the canvas)")


if __name__ == "__main__":
    main()
