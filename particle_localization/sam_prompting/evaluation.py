"""Shared datasets, ground truth, matching, metrics and outputs for every segmentation method.

Each run_<method>.py script builds a `predict(image_bgr, conf)` function that returns one
polygon per predicted particle, and hands it to `run()`. Everything else (finding images,
loading labels, scoring, writing CSVs and overlays) lives here, so all methods are
evaluated exactly the same way.
"""
from pathlib import Path
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from scipy.optimize import linear_sum_assignment


DATA_ROOT = Path('./data')
PATCHES = Path(__file__).resolve().parents[2] / 'data' / 'patches' / 'yolo_seg'  # common patch dataset (single class)
OUTPUT_ROOT = Path('./outputs')
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
LABEL_EXTENSIONS = ['.png', '.tif', '.tiff', '.jpg', '.jpeg', '.bmp']

# Datasets in DATA_ROOT. image_dir / label_dir default to <name>/image and <name>/label.
#
# label_format:
#   'yolo_polygon': one .txt per image, one particle per line: class x1 y1 x2 y2 ... (normalized)
#   'mask':         one label image per image, read with:
#     foreground_threshold: label pixels with value > threshold are particles
#     instances_per_value:  True:  each particle is a connected region of one label value. Needed for
#                                  label maps where values are reused like map colours (1..9), so
#                                  touching particles with different values stay separate
#                           False: binarize first, then split into connected regions
#     label_suffix:         label file name = <image stem><suffix><ext>, e.g. 010.tiff -> 010_label.tif
DATASETS = {
    # Held-out test split of the 256x256 patches the detector was trained on (260 patches)
    'MiNa_dataset': {
        'image_dir': PATCHES / 'images' / 'test',
        'label_dir': PATCHES / 'labels' / 'test',
        'label_format': 'yolo_polygon',
    },
    # Validation split of the same patches; used only to choose the detector confidence threshold
    'MiNa_val': {
        'image_dir': PATCHES / 'images' / 'val',
        'label_dir': PATCHES / 'labels' / 'val',
        'label_format': 'yolo_polygon',
    },
    # Label maps with values 0..9 reused across particles; 010.tiff -> 010_label.tif
    'dataset2_high_mag': {
        'label_format': 'mask',
        'foreground_threshold': 0,
        'instances_per_value': True,
        'label_suffix': '_label',
    },
    # Same label format as high_mag; Img1.png -> Img1_label.tif
    'dataset2_low_mag': {
        'label_format': 'mask',
        'foreground_threshold': 0,
        'instances_per_value': True,
        'label_suffix': '_label',
    },
    # Binary 0/255 masks with a few stray low-value pixels; same file name as the image
    'fragment_1': {
        'label_format': 'mask',
        'foreground_threshold': 127,
        'instances_per_value': False,
        'label_suffix': '',
    },
    # Binary 0/255 masks; same file name as the image
    'fragment_2': {
        'label_format': 'mask',
        'foreground_threshold': 127,
        'instances_per_value': False,
        'label_suffix': '',
    },
}

EVAL_IOU_THRESHOLD = 0.5  # polygon IoU needed for a true positive
MIN_POLYGON_AREA = 5      # smallest polygon (pixels) kept, for predictions
OVERLAY_MAX_SIDE = 2000   # overlays are downscaled so the longer side is at most this many pixels


# =========================
# Command line
# =========================

def base_argument_parser(description):
    """Arguments shared by every run_<method>.py script."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS),
                        help=f'Dataset in {DATA_ROOT} (see DATASETS in evaluation.py)')
    parser.add_argument('--no-eval', action='store_true',
                        help='Only predict and save results; do not load labels or compute metrics')
    return parser


# =========================
# Polygons and masks
# =========================

def mask_to_polygons(mask, min_area=MIN_POLYGON_AREA):
    """Outer contours of a binary mask as [[x, y], ...] polygons, smallest ones dropped."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in contours
            if cv2.contourArea(c) >= min_area and len(c) >= 3]


def instance_polygon(mask, min_area=MIN_POLYGON_AREA):
    """The largest outer contour of one particle's mask, or None if nothing is big enough.

    One prediction is one particle, so stray fragments of a mask are dropped instead of
    being counted as extra particles.
    """
    polygons = mask_to_polygons(mask, min_area)
    if not polygons:
        return None
    return max(polygons, key=lambda p: cv2.contourArea(np.array(p, np.int32)))


def label_map_to_polygons(label_map, min_area=MIN_POLYGON_AREA):
    """One polygon per instance id (> 0) in an instance label map."""
    polygons = []
    for value in np.unique(label_map):
        if value == 0:
            continue
        poly = instance_polygon(label_map == value, min_area)
        if poly is not None:
            polygons.append(poly)
    return polygons


def polygons_to_mask(polygons, shape):
    mask = np.zeros(shape[:2], dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 3:
            cv2.fillPoly(mask, [np.array(poly, np.int32).reshape(-1, 1, 2)], 1)
    return mask


def to_shapely(poly):
    """Valid shapely polygon, or None."""
    if len(poly) < 3:
        return None
    try:
        shape = Polygon(poly)
        if not shape.is_valid:
            shape = shape.buffer(0)
        return shape if shape.is_valid and shape.area > 0 else None
    except Exception:
        return None


# =========================
# Datasets and ground truth
# =========================

def resolve_dataset(name):
    """Dataset settings plus absolute image/label folders; exits if a folder is missing."""
    cfg = dict(DATASETS[name])
    cfg['name'] = name
    cfg['image_dir'] = DATA_ROOT / cfg.get('image_dir', f'{name}/image')
    cfg['label_dir'] = DATA_ROOT / cfg.get('label_dir', f'{name}/label')
    for key in ('image_dir', 'label_dir'):
        if not cfg[key].is_dir():
            raise SystemExit(f'Missing folder for dataset {name}: {cfg[key]}')
    return cfg


def list_images(image_dir):
    return sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_label_path(cfg, image_name):
    stem = Path(image_name).stem
    if cfg['label_format'] == 'yolo_polygon':
        path = cfg['label_dir'] / f'{stem}.txt'
        return path if path.exists() else None
    for name in (f"{stem}{cfg['label_suffix']}", stem):
        for ext in LABEL_EXTENSIONS:
            path = cfg['label_dir'] / f'{name}{ext}'
            if path.exists():
                return path
    return None


def load_ground_truth(cfg, image_name, shape):
    """Return (foreground mask, list of GT particle polygons) for one image.

    Returns (None, None) if the image has no label file.
    """
    label_path = find_label_path(cfg, image_name)
    if label_path is None:
        return None, None
    h, w = shape[:2]

    if cfg['label_format'] == 'yolo_polygon':
        polygons = []
        for line in label_path.read_text().splitlines():
            values = [float(v) for v in line.split()[1:]]
            if len(values) >= 6:
                polygons.append([[values[i] * w, values[i + 1] * h] for i in range(0, len(values) - 1, 2)])
        return polygons_to_mask(polygons, shape), polygons

    label = cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)
    if label.ndim == 3:
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    if label.shape[:2] != (h, w):
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

    foreground = (label > cfg['foreground_threshold']).astype(np.uint8)
    if cfg['instances_per_value']:
        polygons = []
        for value in np.unique(label[foreground > 0]):
            polygons.extend(mask_to_polygons(label == value, min_area=1))
    else:
        polygons = mask_to_polygons(foreground, min_area=1)
    return foreground, polygons


# =========================
# Matching and metrics
# =========================

def match_polygons(pred_polygons, gt_polygons, iou_threshold=EVAL_IOU_THRESHOLD):
    """One-to-one matching (Hungarian on polygon IoU).

    Returns (pred shapes, GT shapes, list of (pred index, GT index, IoU) with IoU >= threshold).
    """
    pred_shapes = [s for s in map(to_shapely, pred_polygons) if s is not None]
    gt_shapes = [s for s in map(to_shapely, gt_polygons) if s is not None]
    if not pred_shapes or not gt_shapes:
        return pred_shapes, gt_shapes, []

    iou = np.zeros((len(pred_shapes), len(gt_shapes)))
    for i, p in enumerate(pred_shapes):
        for j, g in enumerate(gt_shapes):
            if not p.intersects(g):
                continue
            inter = p.intersection(g).area
            iou[i, j] = inter / (p.area + g.area - inter)

    rows, cols = linear_sum_assignment(-iou)
    matches = [(i, j, iou[i, j]) for i, j in zip(rows, cols) if iou[i, j] >= iou_threshold]
    return pred_shapes, gt_shapes, matches


def score_image(pred_polygons, gt_foreground, gt_polygons, shape):
    """Instance and pixel metrics for one image."""
    pred_shapes, gt_shapes, matches = match_polygons(pred_polygons, gt_polygons)
    tp = len(matches)
    fp = len(pred_shapes) - tp
    fn = len(gt_shapes) - tp
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    ious = [m[2] for m in matches]
    area_errors = [abs(pred_shapes[i].area - gt_shapes[j].area) / gt_shapes[j].area * 100 for i, j, _ in matches]

    pred_mask = polygons_to_mask(pred_polygons, shape).astype(bool)
    gt_mask = gt_foreground.astype(bool)
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    total = pred_mask.sum() + gt_mask.sum()

    return {
        'pred_count': len(pred_shapes),
        'gt_count': len(gt_shapes),
        'true_positives': tp,
        'false_positives': fp,
        'false_negatives': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'matched_iou': round(float(np.mean(ious)), 4) if ious else 0.0,
        # Dice of a matched pair follows from its IoU: DSC = 2 IoU / (1 + IoU)
        'matched_dsc': round(float(np.mean([2 * x / (1 + x) for x in ious])), 4) if ious else 0.0,
        'area_mape': round(float(np.mean(area_errors)), 2) if area_errors else 0.0,
        'mask_iou': round(float(inter / union), 4) if union else 1.0,
        'mask_dice': round(float(2 * inter / total), 4) if total else 1.0,
    }


def sweep_ap50(rows):
    """AP50 from the precision/recall points of a confidence sweep (coarse: one point per threshold)."""
    rows = sorted(rows, key=lambda r: r['confidence'], reverse=True)
    mrec = np.concatenate(([0.0], [r['recall'] for r in rows], [1.0]))
    mpre = np.concatenate(([1.0], [r['precision'] for r in rows], [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def best_row_key(row):
    """Order used to pick an image's best threshold in a sweep."""
    return (row['f1_score'], row['matched_iou'], row['mask_iou'], -row['area_mape'], row['precision'], row['recall'])


# =========================
# Outputs
# =========================

def polygon_rows(image_name, polygons, confidence):
    rows = []
    for idx, poly in enumerate(polygons):
        shape = to_shapely(poly)
        if shape is None:
            continue
        x, y, w, h = cv2.boundingRect(np.array(poly, np.int32).reshape(-1, 1, 2))
        rows.append({
            'image_name': image_name,
            'confidence': confidence,
            'polygon_id': idx,
            'area_pixels': round(shape.area, 2),
            'bbox_x': x, 'bbox_y': y, 'bbox_width': w, 'bbox_height': h,
            'centroid_x': round(shape.centroid.x, 2),
            'centroid_y': round(shape.centroid.y, 2),
            'num_vertices': len(poly),
            'polygon_coords': json.dumps(poly),
        })
    return rows


def save_overlay(image_bgr, polygons, path, title):
    """Grey image, colour-filled predictions with red outlines, downscaled for viewing."""
    gray = cv2.cvtColor(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    scale = min(1.0, OVERLAY_MAX_SIDE / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    pts = [(np.array(p, np.float64) * scale).astype(np.int32).reshape(-1, 1, 2) for p in polygons if len(p) >= 3]

    rng = np.random.default_rng(42)
    fill = gray.copy()
    for p in pts:
        cv2.fillPoly(fill, [p], tuple(int(c) for c in rng.integers(0, 256, 3)))
    overlay = cv2.addWeighted(fill, 0.4, gray, 0.6, 0)
    thickness = max(1, round(max(gray.shape[:2]) / 500))
    cv2.polylines(overlay, pts, True, (0, 0, 255), thickness)
    cv2.putText(overlay, title, (10, 10 + 12 * thickness), cv2.FONT_HERSHEY_SIMPLEX,
                0.4 * thickness, (0, 255, 255), thickness)
    cv2.imwrite(str(path), overlay)


# =========================
# Main loop
# =========================

def run(method, predict, dataset, confidences=(None,), evaluate=True, settings=None):
    """Run `predict` on every image of `dataset`, save predictions/overlays, and score them.

    method:      name of the method; results go to OUTPUT_ROOT/<method>/<dataset>/
    predict:     predict(image_bgr, conf) -> list of [[x, y], ...] polygons, one per particle
    confidences: confidence thresholds to try. With more than one (a sweep), each image keeps its
                 best threshold by F1 (chosen with the labels, so an optimistic upper bound) and
                 a sweep AP50 is reported. Methods without a confidence use (None,).
    evaluate:    False skips labels and metrics
    settings:    method settings, saved to run_config.json
    """
    cfg = resolve_dataset(dataset)
    out_dir = OUTPUT_ROOT / method / dataset
    (out_dir / 'overlays').mkdir(parents=True, exist_ok=True)
    sweep = len(confidences) > 1

    image_files = list_images(cfg['image_dir'])
    if not image_files:
        raise SystemExit(f"No images found in {cfg['image_dir']}")

    print(f"Method: {method}")
    print(f"Dataset: {dataset} ({len(image_files)} images)")
    print(f"  images: {cfg['image_dir']}")
    if evaluate:
        print(f"  labels: {cfg['label_dir']} ({cfg['label_format']})")
    print(f"  confidence: {'sweep ' + ', '.join(map(str, confidences)) if sweep else confidences[0]}")
    print(f"  output: {out_dir}")

    run_config = {'method': method, 'dataset': {k: str(v) for k, v in cfg.items()},
                  'confidences': list(confidences), 'evaluate': evaluate,
                  'eval_iou_threshold': EVAL_IOU_THRESHOLD, 'settings': settings or {}}
    (out_dir / 'run_config.json').write_text(json.dumps(run_config, indent=2))

    prediction_rows, image_rows, sweep_rows = [], [], []
    start = time.time()

    for n, image_path in enumerate(image_files, 1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f'[{n}/{len(image_files)}] {image_path.name}: could not read, skipped')
            continue

        gt_foreground = gt_polygons = None
        if evaluate:
            gt_foreground, gt_polygons = load_ground_truth(cfg, image_path.name, image.shape)
            if gt_polygons is None:
                print(f'[{n}/{len(image_files)}] {image_path.name}: no label file, skipped')
                continue

        results = []
        for conf in confidences:
            t0 = time.time()
            polygons = predict(image, conf)
            row = {'image_name': image_path.name, 'confidence': conf, 'seconds': round(time.time() - t0, 3)}
            if evaluate:
                row.update(score_image(polygons, gt_foreground, gt_polygons, image.shape))
            else:
                row['pred_count'] = len(polygons)
            results.append((row, polygons))

        if sweep and evaluate:
            ap50 = sweep_ap50([r for r, _ in results])
            best_row, best_polygons = max(results, key=lambda rp: best_row_key(rp[0]))
            for row, _ in results:
                row['sweep_ap50'] = round(ap50, 4)
                row['is_best'] = row is best_row
                sweep_rows.append(row)
        else:
            best_row, best_polygons = results[0]

        image_rows.append(best_row)
        prediction_rows.extend(polygon_rows(image_path.name, best_polygons, best_row['confidence']))
        save_overlay(image, best_polygons, out_dir / 'overlays' / f'{image_path.stem}.png',
                     f"{method} | {image_path.name} | {len(best_polygons)} particles")

        summary = f"{best_row['pred_count']} predicted"
        if evaluate:
            summary += (f", {best_row['gt_count']} GT, P {best_row['precision']:.3f} R {best_row['recall']:.3f} "
                        f"F1 {best_row['f1_score']:.3f}")
        if sweep:
            summary += f" (best conf {best_row['confidence']})"
        print(f'[{n}/{len(image_files)}] {image_path.name}: {summary}')

    pd.DataFrame(prediction_rows).to_csv(out_dir / 'predictions.csv', index=False)
    df = pd.DataFrame(image_rows)
    df.to_csv(out_dir / 'per_image_metrics.csv', index=False)
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(out_dir / 'confidence_sweep.csv', index=False)

    summary = {'method': method, 'dataset': dataset, 'images': len(df),
               'confidence': 'per-image best (sweep)' if sweep else confidences[0],
               'mean_seconds_per_image': round(df['seconds'].sum() / len(df), 3) if len(df) else 0.0,
               'total_seconds': round(time.time() - start, 1)}
    if evaluate and len(df):
        tp, fp, fn = (int(df[c].sum()) for c in ('true_positives', 'false_positives', 'false_negatives'))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        # Per-image matched-pair averages, weighted by each image's matches = average over all matched particles
        weights = df['true_positives']
        def pooled(column):
            return round(float((df[column] * weights).sum() / weights.sum()), 4) if weights.sum() else 0.0

        summary.update({
            'total_true_positives': tp, 'total_false_positives': fp, 'total_false_negatives': fn,
            # pooled over all particles of the dataset
            'precision': round(p, 4), 'recall': round(r, 4),
            'f1_score': round(2 * p * r / (p + r), 4) if p + r else 0.0,
            'matched_iou': pooled('matched_iou'),
            'matched_dsc': pooled('matched_dsc'),
            'area_mape': pooled('area_mape'),
            # averages of the per-image values
            **{f'mean_{c}': round(float(df[c].mean()), 4)
               for c in ('precision', 'recall', 'f1_score', 'mask_iou', 'mask_dice')},
        })
        if sweep:
            summary['mean_sweep_ap50'] = round(float(df['sweep_ap50'].mean()), 4)
    pd.DataFrame([summary]).to_csv(out_dir / 'summary.csv', index=False)

    print(f"\n{'=' * 60}\n{method} on {dataset}\n{'=' * 60}")
    for key, value in summary.items():
        print(f'{key:>26}: {value}')
    print(f'\nResults saved to {out_dir}')
    return summary
