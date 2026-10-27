#!/usr/bin/env python3
"""
Make a YOLO dataset whose images have the background masked out (train only by default).

Pixels outside the annotated polygons (dilated by a few pixels so particle edges are
kept) are replaced by the patch's median background grey. By default only TRAIN is
masked, so evaluation still runs on real SEM patches. With --mask_splits train val test
every split is masked: a GT-masked ("oracle segmentation") morphology experiment, NOT a
valid detection benchmark, because the test polygons go into the test inputs.
Labels are reused as is.

Input:  a YOLO dataset (images/, labels/) made by coco_to_yolo.py, and the COCO
        patches_256_combined it was made from (for the polygons).
Output: a new YOLO dataset folder; val/test images and all labels are hard-linked.

Usage:
    python mask_background.py --yolo_dir ../data/patches/yolo_det \
        --coco_dir ../data/patches/coco/patches_256_combined \
        --output_dir ../data/patches_train_masked/yolo_det
"""

import argparse
import json
import os
import shutil

import cv2
import numpy as np

SPLITS = ["train", "val", "test"]


def link_or_copy(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def mask_image(image, segmentations, dilate_px):
    """Replace everything outside the polygons with the median background grey."""
    keep = np.zeros(image.shape[:2], dtype=np.uint8)
    for seg in segmentations:
        pts = np.array(seg, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
        cv2.fillPoly(keep, [pts], 1)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        keep = cv2.dilate(keep, k)
    background = keep == 0
    fill = np.median(image[background], axis=0) if background.any() else np.median(image.reshape(-1, image.shape[-1]), axis=0)
    out = image.copy()
    out[background] = fill.astype(image.dtype)
    return out, float(background.mean())


def main():
    parser = argparse.ArgumentParser(description="Mask the background of train images using the GT polygons")
    parser.add_argument("--yolo_dir", required=True, help="YOLO dataset to start from (images/, labels/)")
    parser.add_argument("--coco_dir", required=True, help="COCO patches_256_combined with the polygons")
    parser.add_argument("--output_dir", required=True, help="New YOLO dataset folder to (re)create")
    parser.add_argument("--dilate_px", type=int, default=2, help="Grow polygons by this many pixels before masking")
    parser.add_argument("--mask_splits", nargs="+", default=["train"], choices=SPLITS,
                        help="Splits to mask (default: train). Masking val/test uses their GT polygons.")
    args = parser.parse_args()

    if os.path.isdir(args.output_dir):
        print(f"Removing previous output in {args.output_dir}")
        shutil.rmtree(args.output_dir)

    for split in SPLITS:
        img_in, lbl_in = os.path.join(args.yolo_dir, "images", split), os.path.join(args.yolo_dir, "labels", split)
        img_out, lbl_out = os.path.join(args.output_dir, "images", split), os.path.join(args.output_dir, "labels", split)
        os.makedirs(img_out)
        os.makedirs(lbl_out)
        for name in os.listdir(lbl_in):
            if name.endswith(".txt"):
                link_or_copy(os.path.join(lbl_in, name), os.path.join(lbl_out, name))

        if split not in args.mask_splits:
            for name in os.listdir(img_in):
                link_or_copy(os.path.join(img_in, name), os.path.join(img_out, name))
            print(f"{split}: {len(os.listdir(img_out))} images left unmasked")
            continue

        with open(os.path.join(args.coco_dir, split, "annotations.json")) as f:
            coco = json.load(f)
        segs = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            segs[ann["image_id"]].extend(s for s in ann["segmentation"] if len(s) >= 6)
        masked_share = []
        for img in coco["images"]:
            image = cv2.imread(os.path.join(img_in, img["file_name"]))
            out, share = mask_image(image, segs[img["id"]], args.dilate_px)
            cv2.imwrite(os.path.join(img_out, img["file_name"]), out)
            masked_share.append(share)
        print(f"{split}: {len(coco['images'])} images masked, on average {100 * np.mean(masked_share):.1f}% "
              f"of each patch replaced by background grey")


if __name__ == "__main__":
    main()
