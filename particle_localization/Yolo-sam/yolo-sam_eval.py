from ultralytics import YOLO, SAM
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import argparse
from pathlib import Path
import pandas as pd
import json

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


image_dir = '../../data/patches/yolo_seg/images/test'

# Configuration
CONFIG = {
    'yolo_conf_threshold': 0.6,  # Fixed single threshold for fair comparison
    'yolo_model': './zoo/rt-detr.pt',
    'sam_model': './zoo/sam2.1_l.pt',
    'resize_dim': (260, 260),
    'eval_iou_threshold': 0.5,
    'min_contour_area': 5,
    'random_seed': 42,
    'output_dir': './outputs/yolo_sam'

}

def get_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return '0'  # Use first GPU
    elif torch.backends.mps.is_available():
        return 'mps'  # Use Metal Performance Shaders on macOS
    else:
        return 'cpu'

def load_models(yolo_path, sam_path, device):
    """Load YOLO and SAM models with error handling."""
    try:
        print(f"Loading models on device: {device}")
        yolo_model = YOLO(yolo_path)
        sam_model = SAM(sam_path)
        return yolo_model, sam_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def validate_image_path(image_dir, image_name):
    """Validate and return a working image path."""
    # First try the specified image
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        return image_path
    
    # Try to find any test image in the workspace
    test_dir = Path(image_dir)
    if test_dir.exists():
        image_files = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
        if image_files:
            return str(image_files[0])
    
    raise FileNotFoundError(f"No valid image found. Please check the path: {image_path}")

def load_ground_truth_labels(image_name, image_shape):
    """Load YOLO format ground truth labels and convert to polygons."""
    labels_dir = '../../data/patches/yolo_seg/labels/test'
    label_file = os.path.join(labels_dir, os.path.splitext(image_name)[0] + '.txt')
    
    gt_polygons = []
    if os.path.exists(label_file):
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            h, w = image_shape[:2]
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # class_id + at least 4 coordinates (x1,y1,x2,y2)
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                    
                    # Convert normalized coordinates to pixel coordinates
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            x = coords[i] * w
                            y = coords[i + 1] * h
                            pixel_coords.extend([x, y])
                    
                    # Reshape to polygon format
                    if len(pixel_coords) >= 6:  # At least 3 points (6 coordinates)
                        polygon = [[pixel_coords[i], pixel_coords[i+1]] 
                                 for i in range(0, len(pixel_coords), 2)]
                        gt_polygons.append(polygon)
                        
        except Exception as e:
            print(f"Error loading ground truth for {image_name}: {e}")
    
    return gt_polygons

def create_mask_from_polygons(polygons, image_shape):
    """Convert polygons to binary mask."""
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for poly in polygons:
        if len(poly) >= 3:
            pts = np.array(poly, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)
    return mask

def calculate_mask_iou(pred_mask, gt_mask):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 0.0
    return intersection / union

def calculate_polygon_metrics(pred_polygons, gt_polygons, image_shape, iou_threshold=0.5):
    """Calculate precision, recall, and IoU for polygon predictions."""
    if not gt_polygons and not pred_polygons:
        return 1.0, 1.0, 1.0, 0, 0, 0, [], [], []  # Perfect score when no objects to detect
    
    if not gt_polygons:
        return 0.0, 1.0, 0.0, 0, len(pred_polygons), 0, [], [], []
    
    if not pred_polygons:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_polygons), [], [], []
    
    # Convert polygons to Shapely objects for IoU calculation
    pred_shapes = []
    for poly in pred_polygons:
        try:
            if len(poly) >= 3:
                shape = Polygon(poly)
                if shape.is_valid and shape.area > 0:
                    pred_shapes.append(shape)
        except:
            continue
    
    gt_shapes = []
    for poly in gt_polygons:
        try:
            if len(poly) >= 3:
                shape = Polygon(poly)
                if shape.is_valid and shape.area > 0:
                    gt_shapes.append(shape)
        except:
            continue
    
    if not pred_shapes and not gt_shapes:
        return 1.0, 1.0, 1.0, 0, 0, 0, [], [], []
    if not pred_shapes:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_shapes), [], [], []
    if not gt_shapes:
        return 0.0, 1.0, 0.0, 0, len(pred_shapes), 0, [], [], []
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_shapes), len(gt_shapes)))
    for i, pred_shape in enumerate(pred_shapes):
        for j, gt_shape in enumerate(gt_shapes):
            try:
                intersection = pred_shape.intersection(gt_shape).area
                union = pred_shape.union(gt_shape).area
                if union > 0:
                    iou_matrix[i, j] = intersection / union
            except:
                iou_matrix[i, j] = 0.0
    
    # Find one-to-one matches using Hungarian assignment, then threshold by IoU.
    matched_gt = set()
    matched_pred = set()
    matched_pairs = []  # List of (pred_idx, gt_idx) tuples
    total_iou = 0.0
    match_count = 0

    if SCIPY_AVAILABLE:
        # Maximize IoU by minimizing (1 - IoU)
        cost_matrix = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for pred_idx, gt_idx in zip(row_ind, col_ind):
            iou_val = iou_matrix[pred_idx, gt_idx]
            if iou_val >= iou_threshold:
                matched_pred.add(pred_idx)
                matched_gt.add(gt_idx)
                matched_pairs.append((pred_idx, gt_idx))
                total_iou += iou_val
                match_count += 1
    else:
        # Fallback to greedy matching if scipy is not available.
        for i in range(len(pred_shapes)):
            best_gt_idx = -1
            best_iou = 0.0

            for j in range(len(gt_shapes)):
                if j not in matched_gt and iou_matrix[i, j] > best_iou and iou_matrix[i, j] >= iou_threshold:
                    best_gt_idx = j
                    best_iou = iou_matrix[i, j]

            if best_gt_idx != -1:
                matched_pred.add(i)
                matched_gt.add(best_gt_idx)
                matched_pairs.append((i, best_gt_idx))
                total_iou += best_iou
                match_count += 1
    
    # Calculate metrics
    true_positives = len(matched_pred)
    false_positives = len(pred_shapes) - true_positives
    false_negatives = len(gt_shapes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    mean_iou = total_iou / match_count if match_count > 0 else 0.0
    
    return precision, recall, mean_iou, true_positives, len(pred_shapes), len(gt_shapes), matched_pairs, pred_shapes, gt_shapes


def merge_overlapping_polygons(polygon_list, iou_threshold=0.3):
    """Merge overlapping polygons using IoU threshold."""
    if not polygon_list:
        return []

    shapely_polys = []
    for p in polygon_list:
        if len(p) >= 3:
            try:
                poly = Polygon(p)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_valid and poly.area > 0:
                    shapely_polys.append(poly)
            except Exception:
                continue

    if not shapely_polys:
        return []

    merged = True
    while merged and len(shapely_polys) > 1:
        merged = False
        new_polys = []
        skip_indices = set()

        for i in range(len(shapely_polys)):
            if i in skip_indices:
                continue

            current_poly = shapely_polys[i]
            for j in range(i + 1, len(shapely_polys)):
                if j in skip_indices:
                    continue

                try:
                    other_poly = shapely_polys[j]
                    intersection = current_poly.intersection(other_poly).area
                    union = current_poly.union(other_poly).area

                    if union > 0:
                        iou = intersection / union
                        if iou > iou_threshold:
                            current_poly = current_poly.union(other_poly)
                            skip_indices.add(j)
                            merged = True
                except Exception:
                    continue

            new_polys.append(current_poly)

        shapely_polys = new_polys

    final = []
    for poly in shapely_polys:
        try:
            if poly.geom_type == 'Polygon':
                coords = list(poly.exterior.coords)
                if len(coords) >= 4:
                    final.append(coords[:-1])
            elif poly.geom_type == 'MultiPolygon':
                for part in poly.geoms:
                    coords = list(part.exterior.coords)
                    if len(coords) >= 4:
                        final.append(coords[:-1])
        except Exception:
            continue

    return final


def build_polygon_data(image_name, confidence_threshold, merged_polygons):
    """Build per-polygon records for CSV export."""
    polygon_data = []

    for idx, poly in enumerate(merged_polygons):
        if len(poly) >= 3:
            try:
                shapely_poly = Polygon(poly)
                area = shapely_poly.area
                pts = np.array(poly, np.int32)
                x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
                centroid = shapely_poly.centroid

                polygon_data.append({
                    'image_name': image_name,
                    'polygon_id': idx,
                    'confidence_threshold': confidence_threshold,
                    'area_pixels': round(area, 2),
                    'bbox_x': x,
                    'bbox_y': y,
                    'bbox_width': w,
                    'bbox_height': h,
                    'centroid_x': round(centroid.x, 2),
                    'centroid_y': round(centroid.y, 2),
                    'num_vertices': len(poly),
                    'polygon_coords': json.dumps(poly)
                })
            except Exception as e:
                print(f"Error calculating metrics for polygon {idx}: {e}")
                continue

    return polygon_data


def evaluate_image_at_confidence(image_name, original_image, original_image_rgb, resized_image, gt_polygons,
                                 yolo_model, sam_model, device):
    """Run YOLO+SAM for one image and compute metrics."""
    print(f"Running YOLO detection (conf={CONFIG['yolo_conf_threshold']:.2f})...")
    try:
        yolo_results = yolo_model(resized_image, device=device, conf=CONFIG['yolo_conf_threshold'])
        if yolo_results[0].boxes is not None:
            bboxes = yolo_results[0].boxes.xyxy.tolist()
        else:
            bboxes = []
    except Exception as e:
        print(f"Error during YOLO detection: {e}")
        bboxes = []

    original_h, original_w = original_image.shape[:2]
    resized_h, resized_w = resized_image.shape[:2]
    scale_x = original_w / resized_w
    scale_y = original_h / resized_h

    # Build box prompts from YOLO detections and scale to original image size
    scaled_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        scaled_bboxes.append([
            x_min * scale_x,
            y_min * scale_y,
            x_max * scale_x,
            y_max * scale_y,
        ])

    all_polygons = []
    if len(scaled_bboxes) > 0:
        try:
            print("Running SAM segmentation...")
            sam_results = sam_model(original_image_rgb, bboxes=scaled_bboxes, device=device, retina_masks=True)

            if sam_results[0].masks is not None:
                masks = sam_results[0].masks.data
                print(f"SAM generated {len(masks)} masks")

                for i in range(len(masks)):
                    try:
                        mask_np = masks[i].cpu().numpy().astype(np.uint8)
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) > CONFIG['min_contour_area']:
                                all_polygons.append(cnt.reshape(-1, 2).tolist())
                    except Exception as e:
                        print(f"Error processing mask {i}: {e}")
                        continue
        except Exception as e:
            print(f"Error during SAM segmentation: {e}")

    print(f"Generated {len(all_polygons)} polygons")
    merged_polygons = all_polygons

    # Calculate metrics at primary threshold (0.5)
    precision, recall, mean_iou, true_positives, pred_count, gt_count, matched_pairs, pred_shapes, gt_shapes = calculate_polygon_metrics(
        merged_polygons,
        gt_polygons,
        original_image.shape,
        iou_threshold=CONFIG['eval_iou_threshold']
    )
    
    # Compute per-instance MAPE from matched polygon pairs only
    instance_mape_list = []
    for pred_idx, gt_idx in matched_pairs:
        if pred_idx < len(pred_shapes) and gt_idx < len(gt_shapes):
            pred_area = pred_shapes[pred_idx].area
            gt_area = gt_shapes[gt_idx].area
            if gt_area > 0:
                mape_instance = abs(pred_area - gt_area) / gt_area * 100.0
                instance_mape_list.append(mape_instance)
    
    mape = np.mean(instance_mape_list) if instance_mape_list else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    image_summary = {
        'image_name': image_name,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'area_mape': round(mape, 2),
        'pred_count': pred_count,
        'gt_count': gt_count,
        'true_positives': true_positives,
    }

    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, MAPE: {mape:.2f}%")

    return image_summary


def result_sort_key(result):
    """Choose best threshold primarily by F1, then IoU, then lower MAPE."""
    summary = result['image_summary']
    return (
        summary['f1_score'],
        summary['mean_iou'],
        summary['mask_iou'],
        -summary['area_mape'],
        summary['precision'],
        summary['recall'],
    )


def compute_ap50_from_threshold_rows(threshold_rows):
    """
    Compute AP@0.50 from precision-recall operating points generated by
    confidence sweep for one image.
    """
    if not threshold_rows:
        return 0.0

    # Sort by confidence descending (standard detector sweep direction)
    rows = sorted(threshold_rows, key=lambda r: r['confidence_threshold'], reverse=True)

    recalls = np.array([float(r['recall']) for r in rows], dtype=np.float64)
    precisions = np.array([float(r['precision']) for r in rows], dtype=np.float64)

    # Add sentinels
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # Integrate PR curve (VOC/COCO-style envelope integration)
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


# --- 1. SETUP & PATHS ---
# Load models once (outside the loop for efficiency)
device = get_device()
yolo_model, sam_model = load_models(CONFIG['yolo_model'], CONFIG['sam_model'], device)

# Get list of all PNG images in the directory
image_files = list(Path(image_dir).glob('*.png'))
if not image_files:
    print(f"No PNG images found in {image_dir}")
    exit(1)

print(f"Found {len(image_files)} images to process")

# Initialize list to store all results for CSV export
all_threshold_results = []

if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

# Process each image
for image_file in image_files:
    image_name = image_file.name
    image_path = str(image_file)
    
    print(f"\n[{len(all_threshold_results)+1}/{len(image_files)}] {image_name}")

    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"  Could not load image, skipping...")
            continue
        
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(original_image, CONFIG['resize_dim'], interpolation=cv2.INTER_AREA)
    except Exception as e:
        print(f"  Error loading image: {e}")
        continue

    gt_polygons = load_ground_truth_labels(image_name, original_image.shape)
    
    result = evaluate_image_at_confidence(
        image_name,
        original_image,
        original_image_rgb,
        resized_image,
        gt_polygons,
        yolo_model,
        sam_model,
        device,
    )
    all_threshold_results.append(result)

print("\n=== All images processed! ===")

# Export results
if all_threshold_results:
    df_results = pd.DataFrame(all_threshold_results)
    results_csv_path = os.path.join(CONFIG['output_dir'], 'yolo_sam_comparison.csv')
    df_results.to_csv(results_csv_path, index=False)
    print(f"Results saved to: {results_csv_path}")
    
    # Calculate overall metrics
    overall_precision = np.mean([s['precision'] for s in all_threshold_results])
    overall_recall = np.mean([s['recall'] for s in all_threshold_results])
    overall_f1 = np.mean([s['f1_score'] for s in all_threshold_results])
    overall_mape = np.mean([s['area_mape'] for s in all_threshold_results])
    
    total_tp = sum([s['true_positives'] for s in all_threshold_results])
    total_pred = sum([s['pred_count'] for s in all_threshold_results])
    total_gt = sum([s['gt_count'] for s in all_threshold_results])
    
    print(f"\n{'='*60}")
    print(f"YOLO-SAM (Confidence = {CONFIG['yolo_conf_threshold']:.1f})")
    print(f"{'='*60}")
    print(f"Precision:  {overall_precision:.4f}")
    print(f"Recall:     {overall_recall:.4f}")
    print(f"F1-Score:   {overall_f1:.4f}")
    print(f"Area MAPE:  {overall_mape:.2f}%")
    print(f"{'='*60}")
    print(f"Total Predictions: {total_pred}")
    print(f"Total Ground Truth: {total_gt}")
    print(f"Total True Positives: {total_tp}")
    print(f"{'='*60}\n")
    
    # Save summary
    summary = {
        'model': 'YOLO-SAM',
        'confidence_threshold': CONFIG['yolo_conf_threshold'],
        'precision': overall_precision,
        'recall': overall_recall,
        'f1_score': overall_f1,
        'area_mape': overall_mape,
        'num_images': len(all_threshold_results),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(CONFIG['output_dir'], 'yolo_sam_summary.csv'), index=False)
else:
    print("No results generated.")

# Clean up GPU memory if using CUDA
if device != 'cpu':
    torch.cuda.empty_cache()