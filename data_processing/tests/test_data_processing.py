"""Tests for data_processing/ (create_patches.py + combine_patches.py).

Usage (from anywhere):
    python data_processing/tests/test_data_processing.py [--keep]

Generates patches from <repo>/data/full_images into a temporary folder and checks
geometry, the source-image split, COCO integrity, full coverage of the image area,
train-only PE oversampling, determinism and stale-file cleanup.
The project data/ folders are only read.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict

import cv2
from shapely.geometry import Polygon as ShapelyPolygon

DP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_processing/
REPO_ROOT = os.path.dirname(DP)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--keep", action="store_true", help="keep the generated patches and print where they are")
opts = parser.parse_args()

SCRATCH = tempfile.mkdtemp(prefix="test_data_processing_")
OVERSAMPLE_PE = 3
sys.path.insert(0, DP)
import create_patches as cp  # noqa: E402
FULL_IMAGES = os.path.join(cp.DATA_ROOT, "full_images")

failures = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        failures.append(msg)


def square(x, y, s):
    return [x, y, x + s, y, x + s, y + s, x, y + s]


# ---------------------------------------------------------------- unit tests
print("\n[1] Geometry helpers")
check(cp.calculate_polygon_area(square(0, 0, 10)) == 100, "area of 10x10 square is 100")
check(cp.calculate_bbox_from_polygon(square(3, 4, 10)) == [3, 4, 10, 10], "bbox of square at (3,4)")

inside = cp.clip_polygon_to_patch(square(300, 300, 20), 256, 256)
check(len(inside) == 1, "polygon fully inside gives one piece")
check(abs(cp.calculate_polygon_area(inside[0]) - 400) < 1e-6, "polygon fully inside keeps its area")
check(min(inside[0]) >= 0 and max(inside[0]) <= 256, "clipped coords are patch-relative")
check(cp.calculate_bbox_from_polygon(inside[0]) == [44, 44, 20, 20], "patch-relative offset is correct")

straddle = cp.clip_polygon_to_patch(square(246, 246, 20), 256, 256)
check(len(straddle) == 1 and abs(cp.calculate_polygon_area(straddle[0]) - 100) < 1e-6,
      "polygon across the corner is clipped to 10x10")
check(cp.clip_polygon_to_patch(square(0, 0, 20), 256, 256) == [], "polygon outside the patch is dropped")
check(cp.clip_polygon_to_patch([0, 0, 1, 1], 0, 0) == [], "degenerate polygon (2 points) is dropped")
check(cp.clip_polygon_to_patch(square(256, 0, 20), 0, 0) == [], "polygon only touching the patch edge is dropped")

# U shape whose two arms enter the patch below separately: both arms must be kept
u_shape = [0, 0, 30, 0, 30, 300, 20, 300, 20, 10, 10, 10, 10, 280, 0, 280]
u_clip = cp.clip_polygon_to_patch(u_shape, 0, 256)
check(len(u_clip) == 2, f"U shape cut by the patch edge gives 2 pieces (got {len(u_clip)})")
check(abs(sum(cp.calculate_polygon_area(p) for p in u_clip) - 680) < 1e-6, "both pieces kept: 240 + 440 = 680 px")

# Same U shape through create_patches_for_image: one annotation, 2 polygons, bbox over both arms
img_path = os.path.join(SCRATCH, "synthetic.png")
os.makedirs(SCRATCH, exist_ok=True)
cv2.imwrite(img_path, (255 * __import__("numpy").ones((512 + cp.SCALE_BAR_HEIGHT, 256, 3))).astype("uint8"))
ann = {"id": 7, "category_id": 1, "segmentation": [u_shape], "bbox": [0, 0, 30, 300], "iscrowd": 0}
_, anns, _, _ = cp.create_patches_for_image(img_path, {"file_name": "synthetic.png"}, [ann], 0, 0, "PP")
lower = [a for a in anns if a["image_id"] == 1]
check(len(lower) == 1 and len(lower[0]["segmentation"]) == 2, "lower patch: one annotation holding both pieces")
check(lower and abs(lower[0]["area"] - 680) < 1e-6, "lower patch: area is the sum of both pieces")
check(lower and lower[0]["bbox"] == [0, 0, 30, 44], f"lower patch: bbox covers both arms ({lower[0]['bbox'] if lower else None})")

# ------------------------------------------------------- end-to-end generation
print("\n[2] Generate patches from the real full-size images into scratch")
run_a, run_b = os.path.join(SCRATCH, "run_a"), os.path.join(SCRATCH, "run_b")
for d in (run_a, run_b):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    combine_args = ["--data_root", d] + (["--oversample", f"PE={OVERSAMPLE_PE}"] if d == run_a else [])
    for script, args in (("create_patches.py", ["ALL", "--output_root", d]),
                         ("combine_patches.py", combine_args)):
        r = subprocess.run([sys.executable, os.path.join(DP, script)] + args, capture_output=True, text=True)
        check(r.returncode == 0 and "Error processing" not in r.stdout,
              f"{script} {' '.join(args[:1])} ran cleanly into {os.path.basename(d)}")
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            sys.exit(1)


def load_split(root, split):
    with open(os.path.join(root, "patches_256_combined", split, "annotations.json")) as f:
        return json.load(f)


new = {s: load_split(run_a, s) for s in ("train", "val", "test")}

# ------------------------------------------------------------- leakage check
print("\n[3] No source image appears in more than one split")
src_splits = defaultdict(set)
for s, d in new.items():
    for im in d["images"]:
        src_splits[(im["polymer_type"], im["source_image"])].add(s)
leaks = [k for k, v in src_splits.items() if len(v) > 1]
check(not leaks, f"0 of {len(src_splits)} source images span splits (found {len(leaks)})")
per_poly = Counter(p for p, _ in src_splits)
check(all(any(src_splits[k] == {s} for k in src_splits if k[0] == p) for p in per_poly for s in new),
      "every polymer has at least one source image in each split")

# ------------------------------------------------------------- integrity
print("\n[4] COCO integrity of patches_256_combined")
for s, d in new.items():
    img_ids = [im["id"] for im in d["images"]]
    ann_ids = [a["id"] for a in d["annotations"]]
    poly_of = {im["id"]: im["polymer_type"] for im in d["images"]}
    cat_id = {c["name"]: c["id"] for c in d["categories"]}
    bad_ref = sum(a["image_id"] not in poly_of for a in d["annotations"])
    bad_cat = sum(a["category_id"] != cat_id[poly_of[a["image_id"]]] for a in d["annotations"] if a["image_id"] in poly_of)
    coords = [c for a in d["annotations"] for seg in a["segmentation"] for c in seg]
    bad_bbox = sum(not (a["bbox"][2] > 0 and a["bbox"][3] > 0 and a["bbox"][0] >= 0 and a["bbox"][1] >= 0
                        and a["bbox"][0] + a["bbox"][2] <= 256 and a["bbox"][1] + a["bbox"][3] <= 256)
                   for a in d["annotations"])
    small = sum(a["area"] < cp.MIN_ANNOTATION_AREA for a in d["annotations"])
    img_dir = os.path.join(run_a, "patches_256_combined", s, "images")
    files = set(os.listdir(img_dir))
    wrong_size = sum(cv2.imread(os.path.join(img_dir, im["file_name"])).shape != (256, 256, 3) for im in d["images"])
    check(len(set(img_ids)) == len(img_ids) and len(set(ann_ids)) == len(ann_ids), f"{s}: image and annotation ids unique")
    check(bad_ref == 0, f"{s}: every annotation points to an existing image")
    check(bad_cat == 0, f"{s}: category_id matches the patch's polymer")
    check(min(coords) >= 0 and max(coords) <= 256, f"{s}: all polygon coords within [0, 256]")
    check(bad_bbox == 0, f"{s}: all bboxes positive and inside the patch")
    check(small == 0, f"{s}: no annotation below MIN_ANNOTATION_AREA")
    check(files == {im["file_name"] for im in d["images"]}, f"{s}: image folder matches annotations.json exactly ({len(files)} files)")
    check(wrong_size == 0, f"{s}: every patch image is 256x256x3")

# ------------------------------------------------- coverage of the image area
print("\n[5] Patches cover the whole image area above the info bar")
originals = {sp: [im for im in d["images"] if "oversample_copy_of" not in im] for sp, d in new.items()}
covered = set()
for sp, d in new.items():
    poly_of = {im["id"]: im["polymer_type"] for im in d["images"]}
    covered |= {(poly_of[a["image_id"]], a["source_annotation_id"]) for a in d["annotations"]}
n_src = n_area = n_excl = n_invalid = 0
missing = []
n_images = 0
for p in ("PP", "PE", "PS", "PET"):
    src = json.load(open(os.path.join(FULL_IMAGES, p, f"{p.lower()}.json")))
    height = {im["id"]: im["height"] for im in src["images"]}
    excluded = {im["id"] for im in src["images"] if im["file_name"] in cp.EXCLUDED_IMAGES}
    n_images += len(src["images"]) - len(excluded)
    for a in src["annotations"]:
        n_src += 1
        if a["image_id"] in excluded:
            n_excl += 1
            continue
        if a["bbox"][1] + a["bbox"][3] / 2 >= height[a["image_id"]] - cp.SCALE_BAR_HEIGHT:
            continue  # centre inside the info bar
        segs = [sg for sg in a["segmentation"] if isinstance(sg, list) and len(sg) >= 6]
        if not segs or sum(ShapelyPolygon(list(zip(sg[::2], sg[1::2]))).buffer(0).area for sg in segs) < cp.MIN_ANNOTATION_AREA:
            n_invalid += 1  # fewer than 3 points or below MIN_ANNOTATION_AREA in the source labels
            continue
        n_area += 1
        if (p, a["id"]) not in covered:
            missing.append((p, a["id"], a["area"]))
n_patches = sum(len(v) for v in originals.values())
check(n_patches == 20 * n_images, f"20 patches per image: {n_patches} patches from {n_images} images")
check(not missing, f"every valid annotation above the info bar is in a patch ({n_area - len(missing)}/{n_area}, missing {missing[:5]})")
check(not any("PEL 500X" in im["source_image"] for d in new.values() for im in d["images"]), "PEL 500X.png has no patches")
print(f"  INFO {n_src} source annotations: {n_area} valid above the info bar, {n_invalid} invalid in the source labels "
      f"(< 3 points or < {cp.MIN_ANNOTATION_AREA} px), {n_excl} in other images")

# ---------------------------------------------------------- oversampling
print(f"\n[5b] PE oversampling (x{OVERSAMPLE_PE}) only in train")
for sp, d in new.items():
    copies = [im for im in d["images"] if "oversample_copy_of" in im]
    if sp != "train":
        check(not copies, f"{sp}: no oversampled copies")
        continue
    anns = Counter(a["image_id"] for a in d["annotations"])
    pe_with = [im for im in originals["train"] if im["polymer_type"] == "PE" and anns[im["id"]]]
    copies_of = Counter(im["oversample_copy_of"] for im in copies)
    check(all(im["polymer_type"] == "PE" for im in copies), "train: only PE patches are copied")
    check(sorted(copies_of) == sorted(im["file_name"] for im in pe_with)
          and set(copies_of.values()) == {OVERSAMPLE_PE - 1},
          f"train: each of {len(pe_with)} PE patches with particles has {OVERSAMPLE_PE - 1} copies; empty PE patches have none")
    by_name = {im["file_name"]: im["id"] for im in d["images"]}
    same = all(anns[im["id"]] == anns[by_name[im["oversample_copy_of"]]] for im in copies)
    check(same, "train: every copy has the same annotations as its original")
for sp in ("train", "val", "test"):
    d = new[sp]; anns = Counter(a["image_id"] for a in d["annotations"])
    row = {p: (sum(1 for im in d["images"] if im["polymer_type"] == p),
               sum(1 for im in d["images"] if im["polymer_type"] == p and not anns[im["id"]]),
               sum(anns[im["id"]] for im in d["images"] if im["polymer_type"] == p)) for p in ("PP", "PE", "PS", "PET")}
    print(f"  INFO {sp:5s} " + " | ".join(f"{p}: {n} patches ({e} empty), {a} anns" for p, (n, e, a) in row.items()))

# ---------------------------------------------------------- determinism
print("\n[6] Two runs give the same split")
for p in ("pp", "pe", "ps", "pet"):
    a = json.load(open(os.path.join(run_a, f"patches_256_{p}", "split_source_images.json")))
    b = json.load(open(os.path.join(run_b, f"patches_256_{p}", "split_source_images.json")))
    check(a == b, f"{p.upper()}: identical source-image split")

# ---------------------------------------------------------- stale-file cleanup
print("\n[7] Rerunning removes patches left from an earlier run")
for sub in ("patches_256_pp", "patches_256_combined"):
    stale = os.path.join(run_b, sub, "train", "images", "stale_patch.png")
    shutil.copy(os.path.join(run_b, sub, "train", "images", os.listdir(os.path.join(run_b, sub, "train", "images"))[0]), stale)
subprocess.run([sys.executable, os.path.join(DP, "create_patches.py"), "PP", "--output_root", run_b], capture_output=True, check=True)
subprocess.run([sys.executable, os.path.join(DP, "combine_patches.py"), "--data_root", run_b], capture_output=True, check=True)
for sub in ("patches_256_pp", "patches_256_combined"):
    check(not os.path.exists(os.path.join(run_b, sub, "train", "images", "stale_patch.png")), f"{sub}: stale file removed")

print("\nSplit sizes (patches):", {s: len(d["images"]) for s, d in new.items()},
      " annotations:", {s: len(d["annotations"]) for s, d in new.items()})
print(f"\n{'ALL TESTS PASSED' if not failures else f'{len(failures)} FAILED'}")
if opts.keep:
    print(f"Generated patches kept in {SCRATCH}")
else:
    shutil.rmtree(SCRATCH)
sys.exit(1 if failures else 0)
