#!/usr/bin/env python3
"""
Custom evaluation script to manually calculate APS/APM/APL metrics
since YOLO's built-in evaluation seems to have issues with size-specific APs
"""

import os
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def calculate_iou_mask(mask1, mask2):
    """Calculate IoU between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def get_mask_area(mask):
    """Calculate area of binary mask"""
    return np.sum(mask > 0.5)

def categorize_by_size(area):
    """Categorize by modified size standards (COCO thresholds / 10)"""
    # Original COCO: small < 32², medium 32²-96², large > 96²
    # Modified: small < 3.2², medium 3.2²-9.6², large > 9.6²
    # Rounded: small < 102.4, medium 102.4-921.6, large > 921.6
    if area < 102.4:  # ~10² pixels
        return 'small'
    elif area < 921.6:  # ~30² pixels  
        return 'medium'
    else:
        return 'large'

def parse_ground_truth_masks(label_file, img_width=256, img_height=256):
    """Parse YOLO labels and create ground truth masks"""
    gt_masks = []
    gt_areas = []
    
    if not os.path.exists(label_file):
        return gt_masks, gt_areas
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
            
        coords = [float(x) for x in parts[1:]]
        
        # Convert to pixel coordinates
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_width)
            y = int(coords[i+1] * img_height)
            points.append([x, y])
        
        # Create mask from polygon
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(points)], 1)
        
        area = np.sum(mask)
        if area > 0:
            gt_masks.append(mask)
            gt_areas.append(area)
    
    return gt_masks, gt_areas

def save_overlaid_predictions(img_path, pred_masks, pred_confs, pred_areas, gt_masks, gt_areas, output_dir):
    """Save image with overlaid predictions and bounding boxes"""
    
    # Load original image
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create single figure for predictions only
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Display original image
    ax.imshow(img_rgb)
    #ax.set_title(f'Predictions ({len(pred_masks)} objects)', fontsize=14)
    ax.axis('off')
    
    # Overlay predictions with bounding boxes
    colors_pred = plt.cm.Set2(np.linspace(0, 1, max(len(pred_masks), 1)))
    for j, (pred_mask, pred_conf, pred_area) in enumerate(zip(pred_masks, pred_confs, pred_areas)):
        size_cat = categorize_by_size(pred_area)
        color = colors_pred[j % len(colors_pred)]
        
        # Create colored overlay for mask
        colored_mask = np.zeros_like(img_rgb)
        colored_mask[pred_mask > 0.5] = [int(c*255) for c in color[:3]]
        
        # Blend mask with original image
        alpha = 0.3
        overlay = cv2.addWeighted(img_rgb, 1-alpha, colored_mask, alpha, 0)
        img_rgb[pred_mask > 0.5] = overlay[pred_mask > 0.5]
    
    # Update image display with overlays
    ax.imshow(img_rgb)
    
    # Add size legend
    # legend_elements = [
    #     patches.Patch(color='red', label='Small (<102.4 px²)'),
    #     patches.Patch(color='green', label='Medium (102.4-921.6 px²)'),
    #     patches.Patch(color='blue', label='Large (>921.6 px²)')
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save image
    output_path = output_dir / f"{img_path.stem}_overlay.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    
    return output_path

def evaluate_custom_aps(model, test_dir, conf_threshold=0.25, save_overlays=True):
    """Custom evaluation to calculate APS/APM/APL"""
    
    test_images = list(Path(test_dir).glob("*.png")) + list(Path(test_dir).glob("*.jpg"))
    label_dir = test_dir.replace("images", "labels")
    
    # Create output directory for overlaid images
    output_dir = Path("custom_ap_results")
    output_dir.mkdir(exist_ok=True)
    
    # Store results by size category
    results_by_size = {
        'small': {'tp': [], 'fp': [], 'fn': 0, 'total_gt': 0},
        'medium': {'tp': [], 'fp': [], 'fn': 0, 'total_gt': 0},
        'large': {'tp': [], 'fp': [], 'fn': 0, 'total_gt': 0}
    }
    
    iou_threshold = 0.5
    
    print(f"Evaluating {len(test_images)} test images...")
    if save_overlays:
        print(f"Saving overlay images to: {output_dir}")
    
    # Save overlaid images for all test images
    
    for i, img_path in enumerate(test_images):
        if i % 20 == 0:
            print(f"Processing {i+1}/{len(test_images)}")
        
        # Get ground truth
        label_file = os.path.join(label_dir, img_path.stem + ".txt")
        gt_masks, gt_areas = parse_ground_truth_masks(label_file)
        
        # Categorize GT by size
        gt_by_size = defaultdict(list)
        for mask, area in zip(gt_masks, gt_areas):
            size_cat = categorize_by_size(area)
            gt_by_size[size_cat].append((mask, area))
            results_by_size[size_cat]['total_gt'] += 1
        
        # Run inference
        results = model(str(img_path), conf=conf_threshold, verbose=False)
        
        if len(results) == 0 or results[0].masks is None:
            # All GT are false negatives
            for size_cat in results_by_size:
                results_by_size[size_cat]['fn'] += len(gt_by_size[size_cat])
            continue
        
        # Get predictions
        pred_masks = results[0].masks.data.cpu().numpy()
        pred_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
        
        # Resize prediction masks to image size
        img_shape = (256, 256)  # Your patch size
        pred_masks_resized = []
        pred_areas = []
        
        for pred_mask, conf in zip(pred_masks, pred_confs):
            resized_mask = cv2.resize(pred_mask, img_shape, interpolation=cv2.INTER_NEAREST)
            area = get_mask_area(resized_mask)
            if area > 0:
                pred_masks_resized.append((resized_mask, conf, area))
                pred_areas.append(area)
        
        # Match predictions to ground truth by size category
        for size_cat in ['small', 'medium', 'large']:
            gt_masks_size = [item[0] for item in gt_by_size[size_cat]]
            gt_areas_size = [item[1] for item in gt_by_size[size_cat]]
            
            # Get predictions of this size category
            pred_masks_size = [(mask, conf) for mask, conf, area in pred_masks_resized 
                             if categorize_by_size(area) == size_cat]
            
            # Match predictions to ground truth
            gt_matched = [False] * len(gt_masks_size)
            
            for pred_mask, pred_conf in pred_masks_size:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_mask in enumerate(gt_masks_size):
                    if gt_matched[gt_idx]:
                        continue
                        
                    iou = calculate_iou_mask(pred_mask > 0.5, gt_mask > 0.5)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # True positive
                    results_by_size[size_cat]['tp'].append((pred_conf, True))
                    gt_matched[best_gt_idx] = True
                else:
                    # False positive
                    results_by_size[size_cat]['fp'].append((pred_conf, False))
            
            # Count false negatives (unmatched ground truth)
            fn_count = sum(1 for matched in gt_matched if not matched)
            results_by_size[size_cat]['fn'] += fn_count
        
        # Save overlaid image for all images
        if save_overlays:
            try:
                saved_path = save_overlaid_predictions(
                    img_path, 
                    [item[0] for item in pred_masks_resized],  # masks
                    [item[1] for item in pred_masks_resized],  # confidences  
                    [item[2] for item in pred_masks_resized],  # areas
                    gt_masks, 
                    gt_areas,
                    output_dir
                )
                if i == 0:  # Print first saved image path
                    print(f"  ✓ Sample overlay saved: {saved_path}")
            except Exception as e:
                print(f"  ⚠ Failed to save overlay for {img_path.name}: {e}")
    
    # Calculate AP for each size category
    aps = {}
    
    for size_cat in ['small', 'medium', 'large']:
        tp_list = results_by_size[size_cat]['tp']
        fp_list = results_by_size[size_cat]['fp']
        total_gt = results_by_size[size_cat]['total_gt']
        
        if total_gt == 0:
            aps[size_cat] = 0.0
            continue
        
        # Combine and sort by confidence
        all_dets = [(conf, True) for conf, _ in tp_list] + [(conf, False) for conf, _ in fp_list]
        all_dets.sort(key=lambda x: x[0], reverse=True)
        
        if len(all_dets) == 0:
            aps[size_cat] = 0.0
            continue
        
        # Calculate precision-recall curve
        tp_count = 0
        fp_count = 0
        precisions = []
        recalls = []
        
        for conf, is_tp in all_dets:
            if is_tp:
                tp_count += 1
            else:
                fp_count += 1
                
            precision = tp_count / (tp_count + fp_count)
            recall = tp_count / total_gt
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate AP using trapezoidal rule
        if len(recalls) > 0:
            # Add points at (0,1) and (1,0) for proper AP calculation
            recalls = [0] + recalls + [1]
            precisions = [precisions[0] if len(precisions) > 0 else 0] + precisions + [0]
            
            # Compute AP
            ap = 0
            for i in range(len(recalls) - 1):
                ap += (recalls[i+1] - recalls[i]) * precisions[i+1]
            aps[size_cat] = ap
        else:
            aps[size_cat] = 0.0
    
    return aps, results_by_size

def main():
    """Run custom APS/APM/APL evaluation"""
    
    print("Custom APS/APM/APL Evaluation with Overlay Images")
    print("="*50)
    
    # Find the best trained model
    model_path = None
    
    # Check local runs directory first
    runs_dir = Path("runs/segment")
    if runs_dir.exists():
        # Look for YOLOv26 training runs first
        yolo26_runs = [d for d in runs_dir.iterdir() if d.is_dir() and ("yolo26" in d.name.lower() or "mnp_yolo26" in d.name.lower())]
        if yolo26_runs:
            latest_run = max(yolo26_runs, key=lambda x: x.stat().st_mtime)
            best_model = latest_run / "weights" / "best.pt"
            if best_model.exists():
                model_path = str(best_model)
    
    if not model_path:
        print("No YOLOv26 model found!")
        return
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run custom evaluation
    test_dir = "../../data/patches/yolo_seg/images/test"
    aps, detailed_results = evaluate_custom_aps(model, test_dir, save_overlays=True)
    
    print(f"\n" + "="*50)
    print("CUSTOM CALCULATED APS/APM/APL METRICS")
    print("Modified Size Thresholds (COCO/10):")
    print("Small: < 102.4 px² (~10²), Medium: 102.4-921.6 px² (~10²-30²), Large: > 921.6 px² (~30²)")
    print("="*50)
    
    for size_cat in ['small', 'medium', 'large']:
        ap = aps[size_cat]
        stats = detailed_results[size_cat]
        
        tp_count = len(stats['tp'])
        fp_count = len(stats['fp'])
        fn_count = stats['fn']
        total_gt = stats['total_gt']
        
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
        recall = tp_count / total_gt if total_gt > 0 else 0
        
        print(f"\n{size_cat.upper()} OBJECTS:")
        print(f"  AP: {ap:.4f}")
        print(f"  Ground Truth: {total_gt}")
        print(f"  True Positives: {tp_count}")
        print(f"  False Positives: {fp_count}")
        print(f"  False Negatives: {fn_count}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
    
    print(f"\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"APS (Small):  {aps['small']:.4f}")
    print(f"APM (Medium): {aps['medium']:.4f}")
    print(f"APL (Large):  {aps['large']:.4f}")
    
    print(f"\n💾 RESULTS SAVED:")
    print(f"📁 Overlay images: custom_ap_results/ (sample images with predictions)")
    print(f"📊 Use these images to visually inspect model performance")
    print(f"🎯 Colors indicate object sizes: Red=Small, Green=Medium, Blue=Large")

if __name__ == "__main__":
    main()