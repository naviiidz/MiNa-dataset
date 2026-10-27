import argparse
import json
import os
import subprocess
import sys

from config import DATA_PROCESSING, OVERSAMPLE, PATCH_SIZES, POLYMERS, REFERENCE_SPLITS, size_paths


def run(args):
    subprocess.run([sys.executable] + args, check=True, cwd=DATA_PROCESSING)


def same_split_as_reference(size):
    for p in POLYMERS:
        name = os.path.join(f"patches_256_{p.lower()}", "split_source_images.json")
        with open(os.path.join(size_paths(size)["coco"], name)) as a, open(os.path.join(REFERENCE_SPLITS, name)) as b:
            if json.load(a) != json.load(b):
                return False
    return True


def build(size):
    paths = size_paths(size)
    run(["create_patches.py", "ALL", "--output_root", paths["coco"], "--patch_size", str(size)])
    run(["combine_patches.py", "--data_root", paths["coco"], "--oversample", OVERSAMPLE])
    run(["coco_to_yolo.py", "--mode", "det", "--input_dir", paths["combined"], "--output_dir", paths["yolo"]])
    with open(os.path.join(paths["yolo"], "data.yaml"), "w") as fh:
        fh.write(f"path: {paths['yolo']}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                 f"nc: {len(POLYMERS)}\nnames: {json.dumps(POLYMERS)}\n")
    if not same_split_as_reference(size):
        raise RuntimeError(f"patch size {size}: source-image split differs from data/patches")
    counts = {s: len(os.listdir(os.path.join(paths["yolo"], "images", s))) for s in ("train", "val", "test")}
    print(f"patch size {size}: {counts} patches, same source-image split as data/patches")


def main():
    p = argparse.ArgumentParser(description="Build patch datasets of different sizes with the data_processing pipeline.")
    p.add_argument("--sizes", type=int, nargs="+", default=PATCH_SIZES)
    for size in p.parse_args().sizes:
        build(size)


if __name__ == "__main__":
    main()
