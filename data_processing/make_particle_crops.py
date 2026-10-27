#!/usr/bin/env python3
"""
Build a polymer classification dataset of single-particle crops.

Each crop shows ONE particle: its bounding box (plus a small margin) is cut from the
patch, every pixel outside the particle mask is set to a fixed grey (FILL), and the crop
is padded to a square with the same grey. The fill is the same for every crop, so the
classifier cannot learn the polymer from the background of the source image.

Every particle is used once: particles that appear in two overlapping patches, and the
oversampled PE copies, are reduced to the copy with the largest visible area.

Output (ImageFolder layout, as ultralytics classification expects):
    <output_dir>/{train,val,test}/{PP,PE,PS,PET}/<patch>__<annotation id>.png
    <output_dir>/crops.csv   (split, class, file, source image, area, bbox side)

--balance_train N repeats/subsamples train crops so every class has N crops
(val and test always keep all particles once).

Usage:
    python make_particle_crops.py --coco_dir ../data/patches/coco/patches_256_combined \
        --output_dir ../data/particle_crops --balance_train 3000
"""

import argparse
import csv
import json
import os
import random
import shutil

import cv2
import numpy as np

FILL = 114          # grey used outside the particle mask and for padding (same for every crop)
MARGIN = 3          # pixels of margin around the particle's bounding box
DILATE_PX = 1       # grow the mask slightly so the particle edge is kept
CLASSES = ["PP", "PE", "PS", "PET"]


def masked_crop(image, mask, margin=MARGIN, fill=FILL, dilate_px=DILATE_PX):
    """Crop one particle given its binary mask (H, W); outside the mask -> fill; pad to square."""
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask.astype(np.uint8), k)
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    h, w = mask.shape
    x0, x1 = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    y0, y1 = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    crop = image[y0:y1, x0:x1].copy()
    crop[mask[y0:y1, x0:x1] == 0] = fill
    side = max(crop.shape[:2])
    out = np.full((side, side) + crop.shape[2:], fill, dtype=image.dtype)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    out[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return out


def polygon_mask(segmentation, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for seg in segmentation:
        if len(seg) >= 6:
            pts = np.array(seg, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            cv2.fillPoly(mask, [pts], 1)
    return mask


def unique_particles(coco):
    """One annotation per particle: largest visible copy across overlapping/oversampled patches."""
    images = {im["id"]: im for im in coco["images"]}
    best = {}
    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        key = (img["polymer_type"], img["source_image"], ann["source_annotation_id"])
        if key not in best or ann["area"] > best[key]["area"]:
            best[key] = ann
    return [(images[a["image_id"]], a) for a in best.values()]


def main():
    parser = argparse.ArgumentParser(description="Single-particle masked crops for polymer classification")
    parser.add_argument("--coco_dir", required=True, help="COCO patches_256_combined folder")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--balance_train", type=int, default=0, help="Crops per class in train (0 = keep as is)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print(f"Removing previous output in {args.output_dir}")
        shutil.rmtree(args.output_dir)
    rows = []
    rng = random.Random(args.seed)
    for split in ["train", "val", "test"]:
        with open(os.path.join(args.coco_dir, split, "annotations.json")) as f:
            coco = json.load(f)
        cache = {}
        files = {c: [] for c in CLASSES}
        for c in CLASSES:
            os.makedirs(os.path.join(args.output_dir, split, c))
        for img, ann in sorted(unique_particles(coco), key=lambda t: (t[0]["file_name"], t[1]["id"])):
            if img["file_name"] not in cache:
                cache = {img["file_name"]: cv2.imread(os.path.join(args.coco_dir, split, "images", img["file_name"]))}
            image = cache[img["file_name"]]
            crop = masked_crop(image, polygon_mask(ann["segmentation"], image.shape))
            if crop is None:
                continue
            cls = img["polymer_type"]
            name = f"{os.path.splitext(img['file_name'])[0]}__{ann['id']}.png"
            path = os.path.join(args.output_dir, split, cls, name)
            cv2.imwrite(path, crop)
            files[cls].append(path)
            rows.append([split, cls, os.path.relpath(path, args.output_dir), img["source_image"], ann["area"], crop.shape[0]])

        if split == "train" and args.balance_train:
            for cls, paths in files.items():
                if len(paths) > args.balance_train:      # subsample: delete the extra crops
                    for p in rng.sample(paths, len(paths) - args.balance_train):
                        os.remove(p)
                else:                                     # repeat: hard-link copies until N
                    for k in range(args.balance_train - len(paths)):
                        src = paths[k % len(paths)]
                        os.link(src, src.replace(".png", f"__rep{k // len(paths) + 1}.png"))
        counts = {c: len(os.listdir(os.path.join(args.output_dir, split, c))) for c in CLASSES}
        print(f"{split}: unique particles {{{', '.join(f'{c}: {len(v)}' for c, v in files.items())}}} -> crops {counts}")

    with open(os.path.join(args.output_dir, "crops.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "class", "file", "source_image", "area", "crop_side"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
