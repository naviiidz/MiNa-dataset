import argparse
import csv
import json
import os
import re
import shutil
import sys

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

from .config import PATCH_COCO as SPLITS_ROOT
from .config import FEATURES_DIR, FULL_IMAGES, GLCM_PROPS, POLYMERS, REPO

sys.path.insert(0, os.path.join(REPO, "data_processing"))
from create_patches import EXCLUDED_IMAGES, SCALE_BAR_HEIGHT

FILL = 114
MARGIN = 3
MIN_MASK_PX = 4
BAR_LABELS_UM = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
STANDARD_MAGS = [100, 150, 200, 250, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 5000, 7000, 10000]


def magnification_from_name(name):
    m = re.search(r"(\d+)\s*[xX]", name) or re.search(r"[ -](\d{3,5})-", name)
    return int(m.group(1)) if m else None


def scale_bar_px(image):
    bar = cv2.cvtColor(image[-SCALE_BAR_HEIGHT:], cv2.COLOR_BGR2GRAY) > 200
    best = 0
    for row in bar:
        padded = np.concatenate([[0], row.astype(np.int8), [0]])
        d = np.diff(padded)
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts):
            best = max(best, int((ends - starts).max()))
    return best


def check_magnification(mag, bar_px, tol=0.05):
    for label in BAR_LABELS_UM:
        expected = label * mag / 100.0
        if abs(bar_px - expected) <= tol * expected:
            return label
    return None


def magnification_from_bar(bar_px, name_mag, tol=0.05):
    fits = [m for m in STANDARD_MAGS if check_magnification(m, bar_px, tol)]
    return min(fits, key=lambda m: abs(np.log(m / name_mag))) if fits and name_mag else None


def parse_segmentation(seg):
    if isinstance(seg, str):
        seg = json.loads(seg)
    if isinstance(seg, dict):
        counts, size = seg["counts"], seg["size"]
        return {"counts": json.loads(counts) if isinstance(counts, str) else counts,
                "size": json.loads(size) if isinstance(size, str) else size}
    return [s for s in seg if len(s) >= 6 and len(s) % 2 == 0]


def rle_decode(rle):
    h, w = rle["size"]
    flat = np.zeros(h * w, dtype=np.uint8)
    pos, value = 0, 0
    for run in rle["counts"]:
        flat[pos:pos + run] = value
        pos += run
        value = 1 - value
    return flat.reshape((w, h)).T.copy()


def particle_mask(segments, shape, info_bar_top):
    h, w = info_bar_top, shape[1]
    if isinstance(segments, dict):
        full = rle_decode(segments)[:h]
        n, lab, stats, _ = cv2.connectedComponentsWithStats(full, connectivity=8)
        if n < 2:
            return None
        full = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        ys, xs = np.nonzero(full)
        x0, y0 = max(0, xs.min() - MARGIN), max(0, ys.min() - MARGIN)
        x1, y1 = min(w, xs.max() + 1 + MARGIN), min(h, ys.max() + 1 + MARGIN)
        return full[y0:y1, x0:x1].copy(), x0, y0
    pts = [np.array(s, dtype=np.float32).reshape(-1, 2).round().astype(np.int32) for s in segments]
    allp = np.concatenate(pts)
    x0, y0 = max(0, int(allp[:, 0].min()) - MARGIN), max(0, int(allp[:, 1].min()) - MARGIN)
    x1, y1 = min(w, int(allp[:, 0].max()) + 1 + MARGIN), min(h, int(allp[:, 1].max()) + 1 + MARGIN)
    if x1 <= x0 or y1 <= y0:
        return None
    m = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    cv2.fillPoly(m, [p - [x0, y0] for p in pts], 1)
    return m, x0, y0


def masked_crop(grey, m, x0, y0):
    crop = grey[y0:y0 + m.shape[0], x0:x0 + m.shape[1]].copy()
    crop[m == 0] = FILL
    return crop


def ellipse_axes(m):
    ys, xs = np.nonzero(m)
    cov = np.cov(np.vstack([xs, ys]).astype(np.float64), bias=True) + np.eye(2) / 12.0
    l2, l1 = np.linalg.eigvalsh(cov)
    return 4 * np.sqrt(l1), 4 * np.sqrt(l2), float(np.sqrt(max(0.0, 1 - l2 / l1)))


def contour_perimeter(m):
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(c)
    return cv2.arcLength(c, True), cv2.arcLength(hull, True), cv2.contourArea(hull), c


def max_feret(contour):
    pts = cv2.convexHull(contour).reshape(-1, 2).astype(np.float64)
    if len(pts) < 2:
        return 0.0
    d = pts[:, None, :] - pts[None, :, :]
    return float(np.sqrt((d ** 2).sum(-1)).max())


def features(crop, m, um_per_px):
    area_px = float(m.sum())
    perim_px, hull_perim_px, hull_area_px, contour = contour_perimeter(m)
    perim_px = max(perim_px, 1e-6)
    major, minor, eccentricity = ellipse_axes(m)
    ys, xs = np.nonzero(m)
    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))

    f = {
        "area_px": area_px,
        "eq_diameter_um": np.sqrt(4 * area_px / np.pi) * um_per_px,
        "area_um2": area_px * um_per_px ** 2,
        "perimeter_um": perim_px * um_per_px,
        "major_axis_um": major * um_per_px,
        "minor_axis_um": minor * um_per_px,
        "feret_max_um": max_feret(contour) * um_per_px,
        "aspect_ratio": major / minor,
        "form_factor": min(4 * np.pi * area_px / perim_px ** 2, 1.0),
        "roundness": 4 * area_px / (np.pi * major ** 2),
        "convexity": perim_px / max(hull_perim_px, 1e-6),
        "solidity": area_px / max(hull_area_px, area_px),
        "extent": area_px / bbox_area,
        "eccentricity": eccentricity,
    }

    px = crop[m > 0].astype(np.float64)
    gx, gy = cv2.Sobel(crop, cv2.CV_64F, 1, 0), cv2.Sobel(crop, cv2.CV_64F, 0, 1)
    edge = np.hypot(gx, gy)[m > 0]
    f.update({"grey_mean": px.mean(), "grey_std": px.std(), "grey_p10": np.percentile(px, 10),
              "grey_p50": np.percentile(px, 50), "grey_p90": np.percentile(px, 90), "edge_mean": edge.mean()})


    f.update(glcm_features(crop.astype(np.int32) * 31 // 256 + 1, m, ""))
    f.update(normalised_appearance(crop, m))
    return f


def glcm_features(levels_img, m, prefix):
    q = levels_img.astype(np.uint8)
    q[m == 0] = 0
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=33)
    glcm[0, :, :, :] = 0
    glcm[:, 0, :, :] = 0
    if glcm.sum() == 0:
        return {f"{prefix}glcm_{g}": np.nan for g in GLCM_PROPS}
    glcm = glcm.astype(np.float64) / glcm.sum(axis=(0, 1), keepdims=True).clip(min=1)
    return {f"{prefix}glcm_{g}": float(np.nan_to_num(graycoprops(glcm, g)).mean()) for g in GLCM_PROPS}


def normalised_appearance(crop, m):
    inside = m > 0
    p = crop[inside].astype(np.float64)
    interior = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
    if not interior.any():
        interior = inside
    out = {}

    z = (p - p.mean()) / max(p.std(), 1e-6)
    zr = (np.argsort(np.argsort(p, kind="stable"), kind="stable") + 0.5) / len(p)
    for prefix, vals, lo, hi in (("z_", z, -3.0, 3.0), ("rank_", zr, 0.0, 1.0)):
        img = np.full(crop.shape, float(np.median(vals)))
        img[inside] = vals
        gx, gy = cv2.Sobel(img, cv2.CV_64F, 1, 0), cv2.Sobel(img, cv2.CV_64F, 0, 1)
        out[f"{prefix}edge_mean"] = float(np.hypot(gx, gy)[interior].mean())
        levels = np.clip((img - lo) / (hi - lo), 0, 1 - 1e-9) * 32 + 1
        out.update(glcm_features(np.floor(levels), m, prefix))
    out.update({"z_p10": float(np.percentile(z, 10)), "z_p50": float(np.percentile(z, 50)),
                "z_p90": float(np.percentile(z, 90))})
    return out


def load_splits():
    split_of = {}
    for p in POLYMERS:
        with open(os.path.join(SPLITS_ROOT, f"patches_256_{p.lower()}", "split_source_images.json")) as fh:
            for split, names in json.load(fh).items():
                for n in names:
                    split_of[(p, n)] = split
    return split_of


def main():
    parser = argparse.ArgumentParser(description="Cut every annotated particle from the full SEM images as a masked image "
                                                 "and measure its size, shape and texture descriptors.")
    parser.add_argument("--output-dir", default=FEATURES_DIR, help="where crops, masks and features.csv are written")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print(f"Removing previous output in {args.output_dir}")
        shutil.rmtree(args.output_dir)
    split_of = load_splits()
    rows, image_rows = [], []
    info_bar_top = None

    for polymer in POLYMERS:
        with open(os.path.join(FULL_IMAGES, polymer, f"{polymer.lower()}.json")) as fh:
            coco = json.load(fh)
        anns = {}
        for a in coco["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)

        for img in sorted(coco["images"], key=lambda i: i["file_name"]):
            name = img["file_name"]
            path = os.path.join(FULL_IMAGES, polymer, name)
            if name in EXCLUDED_IMAGES or not os.path.exists(path):
                continue
            split = split_of.get((polymer, name))
            if split is None:
                print(f"  WARNING {polymer}/{name}: not in the patch split, skipped")
                continue
            image = cv2.imread(path)
            grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            info_bar_top = image.shape[0] - SCALE_BAR_HEIGHT
            name_mag = magnification_from_name(name)
            bar_px = scale_bar_px(image)
            mag, mag_source = name_mag, "file name, matches scale bar"
            bar_um = check_magnification(name_mag, bar_px) if name_mag else None
            if bar_um is None:
                bar_mag = magnification_from_bar(bar_px, name_mag)
                if bar_mag:
                    mag, mag_source = bar_mag, f"scale bar ({bar_px} px); file name says {name_mag}"
                    bar_um = check_magnification(bar_mag, bar_px)
                else:
                    mag_source = f"file name only (no readable scale bar, {bar_px} px)"
                print(f"  NOTE {polymer}/{name}: magnification {mag} from {mag_source}")
            um_per_px = 100.0 / mag
            name_polymer = re.match(r"[A-Za-z]+", name).group(0).upper()
            mismatch = not name_polymer.startswith(polymer) or (polymer == "PE" and name_polymer.startswith("PET"))

            n_kept = n_skipped = 0
            for a in anns.get(img["id"], []):
                segments = parse_segmentation(a["segmentation"])
                if not segments:
                    n_skipped += 1
                    continue
                window = particle_mask(segments, image.shape, info_bar_top)
                if window is None or window[0].sum() < MIN_MASK_PX:
                    n_skipped += 1
                    continue
                m, x0, y0 = window
                crop = masked_crop(grey, m, x0, y0)
                ys, xs = np.nonzero(m)
                touches_border = bool(x0 + xs.min() == 0 or y0 + ys.min() == 0
                                      or x0 + xs.max() == image.shape[1] - 1 or y0 + ys.max() == info_bar_top - 1)
                stem = f"{os.path.splitext(name)[0]}__{a['id']}.png"
                rel_crop = os.path.join("crops", split, polymer, stem)
                rel_mask = os.path.join("masks", split, polymer, stem)
                for rel, data in ((rel_crop, crop), (rel_mask, m * 255)):
                    os.makedirs(os.path.dirname(os.path.join(args.output_dir, rel)), exist_ok=True)
                    cv2.imwrite(os.path.join(args.output_dir, rel), data)

                row = {"polymer": polymer, "split": split, "source_image": name, "annotation_id": a["id"],
                       "magnification": mag, "um_per_px": um_per_px, "magnification_source": mag_source, "touches_border": int(touches_border),
                       "from_rle": int(isinstance(segments, dict)),
                       "name_polymer_mismatch": int(mismatch), "crop": rel_crop, "mask": rel_mask}
                row.update(features(crop, m.astype(np.uint8), um_per_px))
                rows.append(row)
                n_kept += 1
            image_rows.append({"polymer": polymer, "split": split, "source_image": name, "magnification": mag,
                               "magnification_source": mag_source,
                               "scale_bar_px": bar_px, "scale_bar_um": bar_um, "um_per_px": um_per_px,
                               "name_polymer_mismatch": int(mismatch), "particles": n_kept, "skipped": n_skipped})
        print(f"{polymer}: {sum(r['polymer'] == polymer for r in rows)} particles from "
              f"{sum(r['polymer'] == polymer for r in image_rows)} images")

    cols = list(rows[0].keys())
    with open(os.path.join(args.output_dir, "features.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.output_dir, "images.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(image_rows[0].keys()))
        w.writeheader()
        w.writerows(image_rows)


    print(f"\n{'':4} {'n':>7} {'diam um':>8} {'area um2':>9} {'AR':>6} {'FF':>6} {'round':>6} {'convex':>7}")
    for p in ["PS", "PP", "PET", "PE"]:
        sel = [r for r in rows if r["polymer"] == p]
        mean = lambda k: np.mean([r[k] for r in sel])
        print(f"{p:4} {len(sel):7d} {mean('eq_diameter_um'):8.3f} {mean('area_um2'):9.3f} {mean('aspect_ratio'):6.3f} "
              f"{mean('form_factor'):6.3f} {mean('roundness'):6.3f} {mean('convexity'):7.3f}")
    print(f"\n{len(rows)} particles -> {args.output_dir}/features.csv")


if __name__ == "__main__":
    main()
