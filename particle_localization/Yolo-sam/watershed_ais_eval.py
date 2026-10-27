"""
Evaluation metrics for pure watershed AIS predictions against ground truth.
Calculates: Precision, Recall, F1-Score, Area MAPE
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


def load_ground_truth(label_dir, img_height=256, img_width=256):
    """Load all ground truth labels from YOLO normalized format."""
    ground_truth = {}
    
    label_dir = Path(label_dir)
    if not label_dir.exists():
        print(f"Error: Label directory {label_dir} does not exist.")
        return ground_truth
    
    label_files = sorted(label_dir.glob("*.txt"))
    
    for label_file in label_files:
        image_name = label_file.stem + ".png"  # Assume images are .png
        polygons = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:  # Need at least class_id + 2 coords
                    continue
                
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # YOLO format: normalized coordinates [0-1]
                # Convert to pixel coordinates
                if len(coords) >= 6 and len(coords) % 2 == 0:
                    poly_coords = []
                    for i in range(0, len(coords), 2):
                        x_norm = coords[i]
                        y_norm = coords[i+1]
                        # Denormalize
                        x_pixel = x_norm * img_width
                        y_pixel = y_norm * img_height
                        poly_coords.append([x_pixel, y_pixel])
                    
                    if len(poly_coords) >= 3:  # Valid polygon
                        polygons.append({
                            'coords': poly_coords,
                            'class_id': class_id
                        })
        
        if polygons:
            ground_truth[image_name] = polygons
    
    return ground_truth


def load_watershed_predictions(predictions_csv):
    """Load watershed predictions from CSV."""
    predictions = {}
    
    csv_path = Path(predictions_csv)
    if not csv_path.exists():
        print(f"Error: CSV file {predictions_csv} does not exist.")
        return predictions
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        poly_coords = json.loads(row['polygon_coords'])
        
        if image_name not in predictions:
            predictions[image_name] = []
        
        predictions[image_name].append({
            'coords': poly_coords,
            'area': row['area_pixels']
        })
    
    return predictions


def polygon_to_shapely(coords):
    """Convert polygon coordinates to Shapely polygon."""
    try:
        return Polygon(coords)
    except:
        return None


def calculate_iou(poly1, poly2):
    """Calculate IoU between two polygons."""
    if poly1 is None or poly2 is None:
        return 0.0
    
    try:
        intersection = poly1.intersection(poly2).area
        union = poly1.union(poly2).area
        
        if union == 0:
            return 0.0
        
        return intersection / union
    except:
        return 0.0


def calculate_polygon_metrics(predicted_polygons, ground_truth_polygons, iou_threshold=0.5):
    """
    Calculate metrics for a single image using IoU-based matching.
    Uses Hungarian algorithm for optimal assignment.
    """
    
    if len(ground_truth_polygons) == 0 and len(predicted_polygons) == 0:
        return {'tp': 0, 'fp': 0, 'fn': 0, 'iou_scores': []}
    
    if len(ground_truth_polygons) == 0:
        return {'tp': 0, 'fp': len(predicted_polygons), 'fn': 0, 'iou_scores': []}
    
    if len(predicted_polygons) == 0:
        return {'tp': 0, 'fp': 0, 'fn': len(ground_truth_polygons), 'iou_scores': []}
    
    # Convert to Shapely polygons
    gt_polys = [polygon_to_shapely(p['coords']) for p in ground_truth_polygons]
    pred_polys = [polygon_to_shapely(p['coords']) for p in predicted_polygons]
    
    # Create IoU matrix
    iou_matrix = np.zeros((len(gt_polys), len(pred_polys)))
    for i, gt_poly in enumerate(gt_polys):
        for j, pred_poly in enumerate(pred_polys):
            iou_matrix[i, j] = calculate_iou(gt_poly, pred_poly)
    
    # Hungarian algorithm for optimal assignment
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    
    # Count matches and mismatches
    tp = 0
    iou_scores = []
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:
            tp += 1
            iou_scores.append(iou)
    
    fp = len(predicted_polygons) - tp
    fn = len(ground_truth_polygons) - tp
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'iou_scores': iou_scores
    }


def calculate_area_metrics(predicted_polygons, ground_truth_polygons, iou_threshold=0.5):
    """Calculate area-based metrics (MAPE)."""
    
    if len(ground_truth_polygons) == 0:
        return {'mape': 0.0, 'matched_pairs': 0}
    
    gt_polys = [polygon_to_shapely(p['coords']) for p in ground_truth_polygons]
    pred_polys = [polygon_to_shapely(p['coords']) for p in predicted_polygons]
    
    # Create IoU matrix
    iou_matrix = np.zeros((len(gt_polys), len(pred_polys)))
    for i, gt_poly in enumerate(gt_polys):
        for j, pred_poly in enumerate(pred_polys):
            iou_matrix[i, j] = calculate_iou(gt_poly, pred_poly)
    
    # Hungarian algorithm
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
    
    # Calculate MAPE only for matched pairs
    ape_values = []
    matched_pairs = 0
    
    for gt_idx, pred_idx in zip(gt_indices, pred_indices):
        iou = iou_matrix[gt_idx, pred_idx]
        if iou >= iou_threshold:
            gt_area = gt_polys[gt_idx].area
            pred_area = pred_polys[pred_idx].area
            
            if gt_area > 0:
                ape = abs(gt_area - pred_area) / gt_area
                ape_values.append(ape)
                matched_pairs += 1
    
    mape = np.mean(ape_values) if ape_values else 0.0
    
    return {'mape': mape, 'matched_pairs': matched_pairs}


def evaluate_watershed(predictions_csv, ground_truth_dir, iou_threshold=0.5):
    """Evaluate watershed predictions against ground truth."""
    
    print(f"\n{'='*70}")
    print(f"WATERSHED AIS EVALUATION")
    print(f"{'='*70}")
    print(f"Predictions: {predictions_csv}")
    print(f"Ground Truth: {ground_truth_dir}")
    print(f"IoU Threshold: {iou_threshold}")
    print(f"{'='*70}\n")
    
    # Load data
    predictions = load_watershed_predictions(predictions_csv)
    ground_truth = load_ground_truth(ground_truth_dir, img_height=256, img_width=256)
    
    if not predictions or not ground_truth:
        print("Error: Could not load predictions or ground truth.")
        return
    
    # Evaluate each image
    all_results = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_ape = []
    
    print(f"Evaluating {len(predictions)} images...\n")
    
    for image_name, pred_polys in predictions.items():
        gt_polys = ground_truth.get(image_name, [])
        
        # Calculate metrics
        match_metrics = calculate_polygon_metrics(pred_polys, gt_polys, iou_threshold)
        area_metrics = calculate_area_metrics(pred_polys, gt_polys, iou_threshold)
        
        tp = match_metrics['tp']
        fp = match_metrics['fp']
        fn = match_metrics['fn']
        mape = area_metrics['mape']
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        if len(match_metrics['iou_scores']) > 0:
            all_ape.extend(match_metrics['iou_scores'])
        
        # Calculate per-image metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result = {
            'image_name': image_name,
            'gt_count': len(gt_polys),
            'pred_count': len(pred_polys),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'mape': round(mape, 4)
        }
        all_results.append(result)
    
    # Calculate overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0.0
    overall_mape = np.mean(all_ape) if all_ape else 0.0
    
    # Create results dataframe
    df_results = pd.DataFrame(all_results)
    
    # Save detailed results
    output_dir = "watershed_evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    detailed_csv = os.path.join(output_dir, 'watershed_detailed_metrics.csv')
    df_results.to_csv(detailed_csv, index=False)
    print(f"✓ Detailed results saved: {detailed_csv}\n")
    
    # Print summary
    print(f"{'='*70}")
    print(f"OVERALL METRICS (IoU Threshold: {iou_threshold})")
    print(f"{'='*70}")
    print(f"Total Images: {len(predictions)}")
    print(f"Total Ground Truth Objects: {total_tp + total_fn}")
    print(f"Total Predicted Objects: {total_tp + total_fp}")
    print(f"\nMatching Results:")
    print(f"  True Positives (TP): {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print(f"\nMetrics:")
    print(f"  Precision: {overall_precision:.4f}")
    print(f"  Recall: {overall_recall:.4f}")
    print(f"  F1-Score: {overall_f1:.4f}")
    print(f"  Area MAPE: {overall_mape:.4f}")
    print(f"{'='*70}\n")
    
    # Per-image statistics
    print(f"Per-Image Statistics:")
    print(f"  Precision - Mean: {df_results['precision'].mean():.4f}, Std: {df_results['precision'].std():.4f}")
    print(f"  Recall    - Mean: {df_results['recall'].mean():.4f}, Std: {df_results['recall'].std():.4f}")
    print(f"  F1-Score  - Mean: {df_results['f1'].mean():.4f}, Std: {df_results['f1'].std():.4f}")
    print(f"  Area MAPE - Mean: {df_results['mape'].mean():.4f}, Std: {df_results['mape'].std():.4f}")
    print(f"{'='*70}\n")
    
    # Summary CSV
    summary_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score', 'Area MAPE'],
        'Value': [overall_precision, overall_recall, overall_f1, overall_mape]
    }
    df_summary = pd.DataFrame(summary_data)
    
    summary_csv = os.path.join(output_dir, 'watershed_summary_metrics.csv')
    df_summary.to_csv(summary_csv, index=False)
    print(f"✓ Summary results saved: {summary_csv}\n")


if __name__ == "__main__":
    # Evaluate watershed predictions
    evaluate_watershed(
        predictions_csv="watershed_ais_output/detailed_predictions_watershed.csv",
        ground_truth_dir="../../data/patches/yolo_seg/labels/test",
        iou_threshold=0.5
    )
