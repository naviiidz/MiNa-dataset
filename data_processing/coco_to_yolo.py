#!/usr/bin/env python3
"""
Convert patches_256_combined (COCO, one annotations.json per split) to YOLO format.

Output layout (what ultralytics expects):
    <output_dir>/images/{train,val,test}/<patch>.png
    <output_dir>/labels/{train,val,test}/<patch>.txt

Modes:
    seg  single class 0, one normalized polygon per annotation. Annotations made of
         several pieces (a particle cut by the patch edge) are joined into one outline
         with ultralytics' merge_multi_segment, so instance counts match COCO.
    det  4 classes (category_id - 1: 0=PP, 1=PE, 2=PS, 3=PET), normalized xywh box.

Patches without annotations get an empty label file (background images).

Usage:
    python coco_to_yolo.py --mode det --output_dir ../data/patches/yolo_det
    python coco_to_yolo.py --mode seg --output_dir ../data/patches/yolo_seg

Images are hard-linked to the COCO patches when both are on the same file system
(no extra disk space); otherwise they are copied.
"""

import argparse
import json
import os
import shutil

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_processing/ sits at the repo root
DEFAULT_INPUT = os.path.join(REPO_ROOT, "data", "patches", "coco", "patches_256_combined")
SPLITS = ["train", "val", "test"]


def seg_line(ann, width, height):
    """YOLO segmentation line for one COCO annotation, or None if it has no polygon."""
    segments = [s for s in ann["segmentation"] if len(s) >= 6]
    if not segments:
        return None
    if len(segments) > 1:
        from ultralytics.data.converter import merge_multi_segment
        points = np.concatenate(merge_multi_segment(segments), axis=0)
    else:
        points = np.array(segments[0], dtype=float).reshape(-1, 2)
    points = points / np.array([width, height])
    points = np.clip(points, 0, 1)
    return "0 " + " ".join(f"{v:.6f}" for v in points.reshape(-1))


def det_line(ann, width, height):
    """YOLO detection line (class cx cy w h) for one COCO annotation."""
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0:
        return None
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    return f"{ann['category_id'] - 1} {cx:.6f} {cy:.6f} {w / width:.6f} {h / height:.6f}"


def convert(input_dir, output_dir, mode):
    if os.path.isdir(output_dir):
        print(f"Removing previous output in {output_dir}")
        shutil.rmtree(output_dir)

    make_line = seg_line if mode == "seg" else det_line
    for split in SPLITS:
        with open(os.path.join(input_dir, split, "annotations.json")) as f:
            coco = json.load(f)
        if mode == "det":
            names = {c["id"]: c["name"] for c in coco["categories"]}
            assert names == {1: "PP", 2: "PE", 3: "PS", 4: "PET"}, f"unexpected categories: {names}"

        img_out = os.path.join(output_dir, "images", split)
        lbl_out = os.path.join(output_dir, "labels", split)
        os.makedirs(img_out)
        os.makedirs(lbl_out)

        anns_by_image = {img["id"]: [] for img in coco["images"]}
        for ann in coco["annotations"]:
            anns_by_image[ann["image_id"]].append(ann)

        n_lines = n_skipped = 0
        for img in coco["images"]:
            src, dst = os.path.join(input_dir, split, "images", img["file_name"]), os.path.join(img_out, img["file_name"])
            try:
                os.link(src, dst)  # hard link: same file, no extra space
            except OSError:
                shutil.copy2(src, dst)
            lines = []
            for ann in anns_by_image[img["id"]]:
                line = make_line(ann, img["width"], img["height"])
                if line is None:
                    n_skipped += 1
                else:
                    lines.append(line)
            label_name = os.path.splitext(img["file_name"])[0] + ".txt"
            with open(os.path.join(lbl_out, label_name), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            n_lines += len(lines)

        print(f"{split}: {len(coco['images'])} images, {len(coco['annotations'])} annotations "
              f"-> {n_lines} label lines ({n_skipped} skipped)")


def main():
    parser = argparse.ArgumentParser(description="Convert COCO patches to YOLO format")
    parser.add_argument("--mode", choices=["seg", "det"], required=True)
    parser.add_argument("--input_dir", default=DEFAULT_INPUT,
                        help="patches_256_combined folder in COCO layout (default: <repo>/data/patches/coco/patches_256_combined)")
    parser.add_argument("--output_dir", required=True, help="YOLO dataset folder to (re)create")
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
