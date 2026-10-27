"""
Simple evaluation of μSAM predictions (Precision, Recall, F1, MAPE only).
Supports all segmentation modes: AIS, AMG, APG.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path
import pandas as pd
import json
from shapely.geometry import Polygon

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Configuration template
def get_config(segmentation_mode='amg'):
    """Get configuration based on segmentation mode."""
    mode = segmentation_mode.lower()
    
    if mode not in ['ais', 'amg', 'apg']:
        raise ValueError(f"Invalid segmentation mode: {mode}. Must be 'ais', 'amg', or 'apg'.")
    
    return {
        'segmentation_mode': mode,
        'predictions_csv': f'./{mode}_usam_output/detailed_predictions.csv',
        'image_dir': '../../data/patches/yolo_seg/images/test',
        'output_dir': './outputs_comparison',
        'eval_iou_threshold': 0.5,
    }

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
                if len(parts) >= 5:
                    coords = [float(x) for x in parts[1:]]
                    pixel_coords = []
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):
                            x = coords[i] * w
                            y = coords[i + 1] * h
                            pixel_coords.extend([x, y])
                    
                    if len(pixel_coords) >= 6:
                        polygon = [[pixel_coords[i], pixel_coords[i+1]] 
                                 for i in range(0, len(pixel_coords), 2)]
                        gt_polygons.append(polygon)
        except Exception as e:
            print(f"Error loading ground truth for {image_name}: {e}")
    
    return gt_polygons

def calculate_polygon_metrics(pred_polygons, gt_polygons, image_shape, iou_threshold=0.5):
    """Calculate precision, recall, and IoU for polygon predictions."""
    if not gt_polygons and not pred_polygons:
        return 1.0, 1.0, 1.0, 0, 0, 0, [], [], []
    
    if not gt_polygons:
        return 0.0, 1.0, 0.0, 0, len(pred_polygons), 0, [], [], []
    
    if not pred_polygons:
        return 1.0, 0.0, 0.0, 0, 0, len(gt_polygons), [], [], []
    
    # Convert polygons to Shapely objects
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
    
    # Hungarian assignment
    matched_pred = set()
    matched_gt = set()
    matched_pairs = []
    total_iou = 0.0
    match_count = 0

    if SCIPY_AVAILABLE:
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
    
    # Calculate metrics
    true_positives = len(matched_pred)
    false_positives = len(pred_shapes) - true_positives
    false_negatives = len(gt_shapes) - len(matched_gt)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    mean_iou = total_iou / match_count if match_count > 0 else 0.0
    
    return precision, recall, mean_iou, true_positives, len(pred_shapes), len(gt_shapes), matched_pairs, pred_shapes, gt_shapes

def load_usam_predictions(csv_path):
    """Load μSAM predictions from CSV."""
    df = pd.read_csv(csv_path)
    predictions_by_image = {}
    
    for _, row in df.iterrows():
        image_name = row['image_name']
        if image_name not in predictions_by_image:
            predictions_by_image[image_name] = []
        
        try:
            coords = json.loads(row['polygon_coords'])
            predictions_by_image[image_name].append({
                'polygon': coords,
                'area': row['area_pixels'],
            })
        except Exception as e:
            print(f"Error parsing polygon for {image_name}: {e}")
            continue
    
    return predictions_by_image

def evaluate_usam(config):
    """Evaluate μSAM predictions for given segmentation mode."""
    os.makedirs(config['output_dir'], exist_ok=True)
    
    mode = config['segmentation_mode'].upper()
    print(f"\n{'='*60}")
    print(f"Evaluating μSAM - {mode} Mode")
    print(f"{'='*60}\n")
    
    print(f"Loading μSAM predictions from {config['predictions_csv']}...")
    predictions_by_image = load_usam_predictions(config['predictions_csv'])
    print(f"Loaded predictions for {len(predictions_by_image)} images\n")
    
    image_files = sorted(Path(config['image_dir']).glob('*.png'))
    print(f"Found {len(image_files)} test images\n")
    
    image_summaries = []
    
    for idx, image_file in enumerate(image_files, 1):
        image_name = image_file.name
        image_path = str(image_file)
        
        print(f"[{idx}/{len(image_files)}] {image_name}")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Could not load image, skipping...")
                continue
        except Exception as e:
            print(f"  Error loading image: {e}")
            continue
        
        gt_polygons = load_ground_truth_labels(image_name, image.shape)
        pred_polygons = []
        
        if image_name in predictions_by_image:
            pred_polygons = [p['polygon'] for p in predictions_by_image[image_name]]
        
        # Calculate metrics
        precision, recall, mean_iou, true_positives, pred_count, gt_count, matched_pairs, pred_shapes, gt_shapes = calculate_polygon_metrics(
            pred_polygons,
            gt_polygons,
            image.shape,
            iou_threshold=config['eval_iou_threshold']
        )
        
        # Calculate MAPE
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
        
        summary = {
            'image_name': image_name,
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            'area_mape': round(mape, 2),
            'pred_count': pred_count,
            'gt_count': gt_count,
            'true_positives': true_positives,
        }
        
        image_summaries.append(summary)
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}, MAPE: {mape:.2f}%")
    
    if image_summaries:
        df = pd.DataFrame(image_summaries)
        csv_path = os.path.join(config['output_dir'], f'usam_{mode.lower()}_comparison.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        # Overall metrics
        overall_precision = np.mean([s['precision'] for s in image_summaries])
        overall_recall = np.mean([s['recall'] for s in image_summaries])
        overall_f1 = np.mean([s['f1_score'] for s in image_summaries])
        overall_mape = np.mean([s['area_mape'] for s in image_summaries])
        
        total_tp = sum([s['true_positives'] for s in image_summaries])
        total_pred = sum([s['pred_count'] for s in image_summaries])
        total_gt = sum([s['gt_count'] for s in image_summaries])
        
        print(f"\n{'='*60}")
        print(f"μSAM ({mode} Mode) - Final Results")
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
        
        summary_df = pd.DataFrame([{
            'model': f'μSAM ({mode})',
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'area_mape': overall_mape,
            'num_images': len(image_summaries),
        }])
        summary_df.to_csv(os.path.join(config['output_dir'], f'usam_{mode.lower()}_summary.csv'), index=False)
        
        return {
            'mode': mode,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'area_mape': overall_mape,
            'num_images': len(image_summaries),
        }

if __name__ == "__main__":
    # Determine which segmentation mode(s) to evaluate
    if len(sys.argv) > 1:
        # User provided mode(s) as command line argument(s)
        modes = sys.argv[1:]
    else:
        # Evaluate all available modes
        modes = ['ais', 'amg', 'apg']
    
    all_results = []
    
    for mode in modes:
        try:
            config = get_config(mode)
            
            # Check if predictions file exists
            if not os.path.exists(config['predictions_csv']):
                print(f"\n⚠ Skipping {mode.upper()}: Predictions file not found at {config['predictions_csv']}")
                continue
            
            result = evaluate_usam(config)
            if result:
                all_results.append(result)
        
        except Exception as e:
            print(f"\n✗ Error evaluating {mode.upper()}: {e}\n")
            continue
    
    # Comparison summary
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print(f"COMPARISON SUMMARY - All Segmentation Modes")
        print(f"{'='*70}")
        comparison_df = pd.DataFrame(all_results)
        print(comparison_df.to_string(index=False))
        print(f"{'='*70}\n")
        
        comparison_df.to_csv(os.path.join('./outputs_comparison', 'usam_all_modes_summary.csv'), index=False)
        print(f"✓ Comparison saved to: ./outputs_comparison/usam_all_modes_summary.csv\n")
    elif len(all_results) == 1:
        print(f"\nEvaluated: {all_results[0]['mode']} mode")
    else:
        print("\n✗ No predictions found for any segmentation mode.")
