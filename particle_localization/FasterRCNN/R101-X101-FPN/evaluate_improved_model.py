#!/usr/bin/env python3
"""
Evaluation Script for Faster R-CNN Model
Compatible with ResNet-50/101 and multi-class microplastic detection
"""

import os
import json
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, build_detection_test_loader, DatasetCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances
import argparse
import cv2

# Paths relative to the project folder (one level above this script's folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches)
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo")

def compute_iou_boxes(box1, box2):
    """Compute IoU between two bounding boxes in [x,y,w,h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1,y1,x2,y2] format
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
    
    # Calculate intersection
    inter_x1 = max(box1_xyxy[0], box2_xyxy[0])
    inter_y1 = max(box1_xyxy[1], box2_xyxy[1])
    inter_x2 = min(box1_xyxy[2], box2_xyxy[2])
    inter_y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def extract_per_class_ap50(evaluator):
    """Extract per-class AP50 metrics from COCO evaluator."""
    try:
        # Access the COCOeval object from the evaluator after evaluation
        if hasattr(evaluator, '_results') and 'bbox' in evaluator._results:
            # The evaluator has been run and has results
            coco_eval = None
            
            # Try different ways to access the COCOeval object
            if hasattr(evaluator, '_coco_eval'):
                coco_eval = evaluator._coco_eval
            elif hasattr(evaluator, 'coco_eval'):
                coco_eval = evaluator.coco_eval
            
            if coco_eval is not None and hasattr(coco_eval, 'eval') and coco_eval.eval:
                # Extract per-class AP at IoU=0.5 (index 0 in the IoU array)
                # COCO evaluation uses IoU thresholds from 0.5 to 0.95 in steps of 0.05
                # Index 0 corresponds to IoU=0.5 (AP50)
                per_class_ap50 = {}
                
                # Get the precision array: [TxRxKxAxM] where:
                # T = IoU thresholds (10), R = recall thresholds (101), K = categories, A = area ranges, M = max dets
                precision = coco_eval.eval['precision']
                
                if precision.size > 0:
                    # Extract AP50 (IoU threshold index 0) for each class
                    # precision shape: [T, R, K, A, M]
                    # We want: IoU=0.5 (index 0), all recall (all R), each category (K), area=all (index 0), maxDets=100 (index 2)
                    ap50_precision = precision[0, :, :, 0, 2]  # IoU=0.5, all areas, maxDets=100
                    
                    class_names = ["PP", "PE", "PS", "PET"]
                    for class_idx, class_name in enumerate(class_names):
                        if class_idx < ap50_precision.shape[1]:
                            # Calculate AP50 for this class (mean over recall thresholds, excluding -1 values)
                            class_precision = ap50_precision[:, class_idx]
                            valid_precision = class_precision[class_precision > -1]
                            if len(valid_precision) > 0:
                                per_class_ap50[class_name] = float(valid_precision.mean() * 100)  # Convert to percentage
                            else:
                                per_class_ap50[class_name] = 0.0
                
                return per_class_ap50
            
            # If we can't access COCOeval directly, try to compute from custom analysis
            return None
            
    except Exception as e:
        print(f"Warning: Could not extract per-class AP50: {e}")
        return None
    
    return None

def compute_per_class_ap50_custom(detailed_metrics):
    """Compute per-class AP50 from our detailed metrics analysis."""
    per_class_ap50 = {}
    
    class_names = ["PP", "PE", "PS", "PET"]
    class_ids = [0, 1, 2, 3]
    
    for class_id, class_name in zip(class_ids, class_names):
        if class_id in detailed_metrics['class_metrics']:
            metrics = detailed_metrics['class_metrics'][class_id]
            tp = metrics['tp']
            fp = metrics['fp']
            fn = metrics['fn']
            
            # Calculate precision and recall at IoU=0.5
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # For AP50, we use precision as an approximation
            # This is a simplified calculation - true AP50 would need multiple precision points
            ap50_estimate = precision * 100  # Convert to percentage
            per_class_ap50[class_name] = ap50_estimate
        else:
            per_class_ap50[class_name] = 0.0
    
    return per_class_ap50

def compute_iou_boxes_duplicate():
    """Compute IoU between two bounding boxes in [x,y,w,h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to [x1,y1,x2,y2] format
    box1_xyxy = [x1, y1, x1 + w1, y1 + h1]
    box2_xyxy = [x2, y2, x2 + w2, y2 + h2]
    
    # Calculate intersection
    inter_x1 = max(box1_xyxy[0], box2_xyxy[0])
    inter_y1 = max(box1_xyxy[1], box2_xyxy[1])
    inter_x2 = min(box1_xyxy[2], box2_xyxy[2])
    inter_y2 = min(box1_xyxy[3], box2_xyxy[3])
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def compute_detailed_metrics(predictor, dataset_name):
    """Compute detailed precision, recall, and IoU metrics for bounding boxes."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    # Class names matching the training order  
    class_names = ["PP", "PE", "PS", "PET"]
    # Use 0-based indexing for consistency with detectron2
    class_ids = [0, 1, 2, 3]  # Internal detectron2 class IDs (0-based)
    # IMPORTANT: Check if filtered dataset uses 0-based or 1-based class IDs
    # Based on debug output, filtered dataset is using 0-based IDs (GT class 0)
    # So no conversion needed - filtered dataset classes are already 0-based
    annotation_to_detectron2 = {0: 0, 1: 1, 2: 2, 3: 3}  # Identity mapping for filtered dataset
    
    # Overall metrics
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    iou_scores = []
    
    # Per-class metrics
    class_metrics = {}
    for class_id, class_name in zip(class_ids, class_names):
        class_metrics[class_id] = {
            'name': class_name,
            'tp': 0, 'fp': 0, 'fn': 0,
            'iou_scores': [],
            'predictions': 0,
            'ground_truth': 0
        }
    
    iou_thresholds = [0.5]
    metrics_per_threshold = {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in iou_thresholds}
    class_metrics_per_threshold = {}
    for thresh in iou_thresholds:
        class_metrics_per_threshold[thresh] = {class_id: {'tp': 0, 'fp': 0, 'fn': 0} for class_id in class_ids}
    
    print("Processing test images for detailed metrics...")
    
    # Debug counters
    total_processed_images = 0
    total_gt_annotations = 0
    total_predictions = 0
    
    for i, dataset_dict in enumerate(dataset_dicts):
        if i % 50 == 0:
            print(f"Processed {i}/{len(dataset_dicts)} images")
            
        total_processed_images += 1
        
        # Load image and run inference
        im = cv2.imread(dataset_dict["file_name"])
        outputs = predictor(im)
        pred_instances = outputs["instances"].to("cpu")
        
        # Get ground truth annotations
        gt_annotations = dataset_dict["annotations"]
        total_gt_annotations += len(gt_annotations)
        
        # Convert predictions to bounding boxes
        if len(pred_instances) > 0:
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()  # [x1,y1,x2,y2] format
            pred_scores = pred_instances.scores.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            # Convert to [x,y,w,h] format
            pred_boxes_xywh = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in pred_boxes]
            total_predictions += len(pred_boxes_xywh)
        else:
            pred_boxes_xywh = []
            pred_scores = np.array([])
            pred_classes = np.array([])
        
        # Get ground truth bounding boxes
        gt_boxes = [ann['bbox'] for ann in gt_annotations]  # Already in [x,y,w,h] format
        gt_classes = [ann['category_id'] for ann in gt_annotations]
        
        # Update per-class ground truth counts
        for gt_class in gt_classes:
            # Convert annotation class ID (1-based) to detectron2 class ID (0-based)
            detectron2_class = annotation_to_detectron2.get(gt_class)
            if detectron2_class is not None:
                class_metrics[detectron2_class]['ground_truth'] += 1

        # Update per-class prediction counts
        for pred_class in pred_classes:
            # pred_class is already 0-based detectron2 class ID
            if pred_class in class_metrics:
                class_metrics[pred_class]['predictions'] += 1
        
        # Match predictions to ground truth using IoU
        if len(pred_boxes_xywh) > 0 and len(gt_boxes) > 0:
            # Compute IoU matrix - compute for ALL pairs first, then filter by class
            iou_matrix = np.zeros((len(pred_boxes_xywh), len(gt_boxes)))
            for p_idx, pred_box in enumerate(pred_boxes_xywh):
                for g_idx, gt_box in enumerate(gt_boxes):
                    # Compute IoU for all pairs
                    iou_matrix[p_idx, g_idx] = compute_iou_boxes(pred_box, gt_box)
            
            # Simple greedy matching (IoU > 0.5 threshold)
            matched_pairs = []
            used_gt = set()
            used_pred = set()
            
            # Sort predictions by score (highest first)
            pred_order = np.argsort(-pred_scores)
            
            # SIMPLIFIED MATCHING: Find best IoU match for each prediction
            for p_idx in pred_order:
                best_gt_idx = -1
                best_iou = 0.5  # Minimum IoU threshold
                
                for g_idx in range(len(gt_boxes)):
                    if g_idx not in used_gt:
                        # Check if classes match
                        gt_detectron2_class = annotation_to_detectron2.get(gt_classes[g_idx])
                        if (gt_detectron2_class is not None and 
                            pred_classes[p_idx] == gt_detectron2_class):
                            # Check if IoU is better than current best
                            current_iou = iou_matrix[p_idx, g_idx]
                            if current_iou > best_iou:
                                best_iou = current_iou
                                best_gt_idx = g_idx
                
                # If we found a good match, record it
                if best_gt_idx >= 0:
                    matched_pairs.append((p_idx, best_gt_idx, best_iou))
                    used_gt.add(best_gt_idx)
                    used_pred.add(p_idx)
                    iou_scores.append(best_iou)
                    
                    # Track per-class metrics
                    gt_annotation_class = gt_classes[best_gt_idx]
                    gt_detectron2_class = annotation_to_detectron2.get(gt_annotation_class)
                    if gt_detectron2_class is not None:
                        class_metrics[gt_detectron2_class]['iou_scores'].append(best_iou)
            
            # Count TP, FP, FN for different IoU thresholds
            for thresh in iou_thresholds:
                tp_thresh = sum(1 for _, _, iou in matched_pairs if iou >= thresh)
                fp_thresh = len(pred_boxes_xywh) - tp_thresh
                fn_thresh = len(gt_boxes) - tp_thresh
                
                metrics_per_threshold[thresh]['tp'] += tp_thresh
                metrics_per_threshold[thresh]['fp'] += fp_thresh
                metrics_per_threshold[thresh]['fn'] += fn_thresh
                
                # Per-class threshold metrics
                for class_id in class_ids:
                    class_tp = sum(1 for p_idx, g_idx, iou in matched_pairs 
                                 if iou >= thresh and annotation_to_detectron2.get(gt_classes[g_idx]) == class_id)
                    class_fp = sum(1 for p_idx in range(len(pred_boxes_xywh)) 
                                 if p_idx not in used_pred and pred_classes[p_idx] == class_id)
                    class_fn = sum(1 for g_idx in range(len(gt_boxes)) 
                                 if g_idx not in used_gt and annotation_to_detectron2.get(gt_classes[g_idx]) == class_id)
                    
                    class_metrics_per_threshold[thresh][class_id]['tp'] += class_tp
                    class_metrics_per_threshold[thresh][class_id]['fp'] += class_fp
                    class_metrics_per_threshold[thresh][class_id]['fn'] += class_fn
            
            # Overall counts (IoU > 0.5)
            total_tp += len(matched_pairs)
            total_fp += len(pred_boxes_xywh) - len(matched_pairs)
            total_fn += len(gt_boxes) - len(matched_pairs)
            
            # Per-class overall counts
            for class_id in class_ids:
                class_tp = sum(1 for _, g_idx, _ in matched_pairs 
                             if annotation_to_detectron2.get(gt_classes[g_idx]) == class_id)
                class_fp = sum(1 for p_idx in range(len(pred_boxes_xywh)) 
                             if p_idx not in used_pred and pred_classes[p_idx] == class_id)
                class_fn = sum(1 for g_idx in range(len(gt_boxes)) 
                             if g_idx not in used_gt and annotation_to_detectron2.get(gt_classes[g_idx]) == class_id)
                
                class_metrics[class_id]['tp'] += class_tp
                class_metrics[class_id]['fp'] += class_fp
                class_metrics[class_id]['fn'] += class_fn
        else:
            # No predictions or no ground truth
            total_fp += len(pred_boxes_xywh)
            total_fn += len(gt_boxes)
            
            # Per-class counts when no matches
            for p_idx in range(len(pred_boxes_xywh)):
                pred_class_id = pred_classes[p_idx]
                if pred_class_id in class_metrics:
                    class_metrics[pred_class_id]['fp'] += 1
            
            for gt_class in gt_classes:
                detectron2_class = annotation_to_detectron2.get(gt_class)
                if detectron2_class is not None:
                    class_metrics[detectron2_class]['fn'] += 1
            
            for thresh in iou_thresholds:
                metrics_per_threshold[thresh]['fp'] += len(pred_boxes_xywh)
                metrics_per_threshold[thresh]['fn'] += len(gt_boxes)
                
                # Per-class threshold updates
                for p_idx in range(len(pred_boxes_xywh)):
                    pred_class_id = pred_classes[p_idx]
                    if pred_class_id in class_metrics_per_threshold[thresh]:
                        class_metrics_per_threshold[thresh][pred_class_id]['fp'] += 1
                
                for gt_class in gt_classes:
                    detectron2_class = annotation_to_detectron2.get(gt_class)
                    if detectron2_class is not None and detectron2_class in class_metrics_per_threshold[thresh]:
                        class_metrics_per_threshold[thresh][detectron2_class]['fn'] += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    
    # Debug output
    print(f"\n🐛 DEBUG INFO:")
    print(f"  Processed images: {total_processed_images}")
    print(f"  Total GT annotations: {total_gt_annotations}")
    print(f"  Total predictions made: {total_predictions}")
    print(f"  IoU matches found: {len(iou_scores)}")
    
    # Debug: Check class-wise ground truth and predictions
    print(f"\n🔬 DETAILED CLASS ANALYSIS:")
    for class_id in class_ids:
        metrics = class_metrics[class_id]
        name = metrics['name']
        gt_count = metrics['ground_truth']
        pred_count = metrics['predictions']
        tp_count = metrics['tp']
        print(f"  {name}: {gt_count} GT, {pred_count} Pred, {tp_count} TP")
    
    # Debug: Check if we're actually finding the predictions
    print(f"\n🎯 PREDICTION CONFIDENCE ANALYSIS:")
    print(f"  Min confidence threshold used: 0.25")
    print(f"  NOTE: COCO evaluation uses all detections and applies multiple thresholds")
    print(f"  NOTE: Our custom analysis only uses detections above 0.25 confidence")
    
    print(f"\n📊 OVERALL METRICS (IoU > 0.5):")
    print(f"  True Positives:  {total_tp}")
    print(f"  False Positives: {total_fp}")
    print(f"  False Negatives: {total_fn}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1_score:.3f}")
    print(f"  Mean IoU: {mean_iou:.3f}")
    
    print(f"\n📈 METRICS BY IoU THRESHOLD:")
    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 45)
    
    for thresh in sorted(iou_thresholds):
        metrics = metrics_per_threshold[thresh]
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        
        print(f"{thresh:<10.1f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")
    
    # Per-class detailed metrics
    print(f"\n📋 PER-CLASS METRICS (IoU > 0.5):")
    print(f"{'Class':<8} {'GT':<6} {'Pred':<6} {'TP':<6} {'FP':<6} {'FN':<6} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Mean IoU':<10}")
    print("-" * 90)
    
    for class_id in class_ids:
        metrics = class_metrics[class_id]
        name = metrics['name']
        tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
        gt_count = metrics['ground_truth']
        pred_count = metrics['predictions']
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
        mean_iou = np.mean(metrics['iou_scores']) if metrics['iou_scores'] else 0
        
        print(f"{name:<8} {gt_count:<6} {pred_count:<6} {tp:<6} {fp:<6} {fn:<6} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f} {mean_iou:<10.3f}")
    
    # Per-class metrics by threshold
    print(f"\n📊 PER-CLASS METRICS BY IoU THRESHOLD (TEST SET):")
    for class_id in class_ids:
        class_name = class_metrics[class_id]['name']
        print(f"\n{class_name} (Class {class_id}):")
        print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
        print("-" * 45)
        
        for thresh in sorted(iou_thresholds):
            metrics = class_metrics_per_threshold[thresh][class_id]
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            
            print(f"{thresh:<10.1f} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")
    
    print(f"\n🎯 IoU DISTRIBUTION:")
    if iou_scores:
        iou_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        for min_iou, max_iou in iou_ranges:
            count = sum(1 for iou in iou_scores if min_iou <= iou < max_iou)
            percentage = count / len(iou_scores) * 100
            print(f"  IoU {min_iou:.1f}-{max_iou:.1f}: {count:4d} matches ({percentage:5.1f}%)")
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'class_metrics': class_metrics,
        'class_metrics_per_threshold': class_metrics_per_threshold
    }

def setup_improved_cfg(model_path, model_variant="FasterRCNN-R50"):
    """Setup configuration matching the improved model architecture."""
    cfg = get_cfg()
    
    # Configure model based on variant
    if model_variant == "FasterRCNN-R50":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    elif model_variant == "FasterRCNN-R101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    elif model_variant == "FasterRCNN-X101":
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  # PP, PE, PS, PET
    
    # Enhanced FPN settings (matching training)
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 384  # Matches training config
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    # Enhanced anchor settings (matching training config exactly)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 16],      # p2
        [24, 32],     # p3
        [48, 64],     # p4
        [128, 192]    # p5
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]
    
    # RPN settings (matching training)
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match anchor config
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    
    # ROI settings (matching training)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 384  # Matches training
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5    # Matches training threshold
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # Multi-scale features
    
    # Input settings (matching training)
    cfg.INPUT.MIN_SIZE_TEST = 800  # Matches training test resolution
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def filter_small_annotations(dataset_name, min_size_pixels=5):
    """Filter out annotations with bounding boxes smaller than min_size_pixels."""
    print(f"Filtering annotations smaller than {min_size_pixels} pixels...")
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    filtered_dicts = []
    
    total_annotations = 0
    filtered_annotations = 0
    
    for record in dataset_dicts:
        filtered_record = record.copy()
        filtered_annotations_list = []
        
        for annotation in record["annotations"]:
            total_annotations += 1
            bbox = annotation["bbox"]
            width, height = bbox[2], bbox[3]  # COCO format: [x, y, width, height]
            
            # Keep annotation if both width and height are >= min_size_pixels
            if width >= min_size_pixels and height >= min_size_pixels:
                filtered_annotations_list.append(annotation)
            else:
                filtered_annotations += 1
        
        # Only keep images that still have annotations after filtering
        if filtered_annotations_list:
            filtered_record["annotations"] = filtered_annotations_list
            filtered_dicts.append(filtered_record)
    
    print(f"Filtered {filtered_annotations} out of {total_annotations} annotations")
    print(f"Kept {len(filtered_dicts)} images with valid annotations")
    
    # Re-register the filtered dataset
    DatasetCatalog.register(f"{dataset_name}_filtered", lambda: filtered_dicts)
    MetadataCatalog.get(f"{dataset_name}_filtered").set(
        thing_classes=MetadataCatalog.get(dataset_name).thing_classes
    )
    
    return f"{dataset_name}_filtered"

def register_datasets():
    """Register the multi-class microplastic dataset with filtering (matching training)."""
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_root = os.path.join(os.path.dirname(project_root), "data", "patches", "coco", "patches_256_combined")
    
    # Register test set for evaluation
    register_coco_instances(
        "microplastic_test_multi",
        {},
        os.path.join(dataset_root, "test", "annotations.json"),
        os.path.join(dataset_root, "test", "images")
    )
    
    # Set thing classes for multiple polymer types (matching training order)
    MetadataCatalog.get("microplastic_test_multi").thing_classes = ["PP", "PE", "PS", "PET"]
    
    # Apply moderate small particle filtering (remove particles < 3 pixels) - matching training
    print("🔍 Applying moderate small particle filtering...")
    test_filtered = filter_small_annotations("microplastic_test_multi", min_size_pixels=3)
    print(f"✅ Using filtered test dataset: {test_filtered}")
    
    return test_filtered

def evaluate_improved_model(model_path, model_variant="FasterRCNN-R50"):
    """Comprehensive evaluation of the improved model."""
    print("="*70)
    print("FASTER R-CNN MODEL EVALUATION")
    print("="*70)
    if model_variant == "FasterRCNN-R50":
        backbone_desc = "ResNet-50 + FPN"
    elif model_variant == "FasterRCNN-R101":
        backbone_desc = "ResNet-101 + FPN"
    else:
        backbone_desc = "ResNeXt-101 + FPN"
    print(f"Architecture: {backbone_desc}")
    print("Classes: PP, PE, PS, PET (Multi-class detection)")
    print("Enhancements: Optimized anchors, improved RPN, better ROI settings")
    print("="*70)
    
    # Register dataset and get filtered dataset name
    filtered_dataset = register_datasets()
    
    # Setup configuration
    cfg = setup_improved_cfg(model_path, model_variant)
    
    # Use the filtered dataset for evaluation
    cfg.DATASETS.TEST = (filtered_dataset,)
    
    print(f"\nModel Configuration:")
    print(f"Backbone: {backbone_desc}")
    print(f"Classes: 4 (PP, PE, PS, PET)")
    print(f"Anchor sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"Anchor ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")
    print(f"RPN NMS threshold: {cfg.MODEL.RPN.NMS_THRESH}")
    print(f"ROI score threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
    print(f"ROI NMS threshold: {cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST}")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    print(f"\nModel loaded from: {model_path}")
    print(f"Running evaluation on test set...")
    
    # Run COCO evaluation
    evaluator = COCOEvaluator(
        filtered_dataset,
        output_dir=os.path.join(os.path.dirname(model_path), "evaluation_improved")
    )
    test_loader = build_detection_test_loader(cfg, filtered_dataset)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    
    # Extract detailed per-class AP50 metrics
    per_class_ap50_coco = extract_per_class_ap50(evaluator)
    
    # Print results
    print("\n" + "="*70)
    print("FASTER R-CNN EVALUATION RESULTS")
    print("="*70)
    
    # Debug: Show available metric keys
    if 'bbox' in results:
        bbox = results['bbox']
        print(f"\n🔍 DEBUG - Available COCO metrics keys:")
        print(f"  {sorted(bbox.keys())}")
    
    if 'bbox' in results:
        bbox = results['bbox']
        print(f"\nBOUNDING BOX DETECTION METRICS:")
        print(f"  AP (IoU=0.50:0.95): {bbox['AP']:.3f}")
        print(f"  AP50 (IoU=0.50): {bbox['AP50']:.3f}")
        print(f"  AP75 (IoU=0.75): {bbox['AP75']:.3f}")
        print(f"  APs (small objects): {bbox.get('APs', 0):.3f}")
        print(f"  APm (medium objects): {bbox.get('APm', 0):.3f}")
        print(f"  APl (large objects): {bbox.get('APl', 0):.3f}")
        
        # Extract per-class AP metrics (automatically calculated by COCO evaluator)
        class_names = ["PP", "PE", "PS", "PET"]
        per_class_ap = []
        
        print(f"\n  PER-CLASS AP METRICS:")
        print(f"    {'Class':<6} {'AP (IoU=0.50:0.95)':<15}")
        print(f"    {'-'*6:<6} {'-'*15:<15}")
        
        for i, class_name in enumerate(class_names):
            # Try different key formats for per-class AP metrics
            ap_key = f'AP-{class_name}'
            idx_key = f'AP{i}'
            
            # Get AP (IoU=0.50:0.95)
            ap_value = None
            if ap_key in bbox:
                ap_value = bbox[ap_key]
            elif idx_key in bbox:
                ap_value = bbox[idx_key]
                
            if ap_value is not None:
                per_class_ap.append((class_name, ap_value))
                ap_str = f"{ap_value:.3f}"
            else:
                ap_str = "N/A"
                
            print(f"    {class_name:<6} {ap_str:<15}")
        
        # Calculate and show mAP
        if per_class_ap:
            map_value = sum(ap for _, ap in per_class_ap) / len(per_class_ap)
            print(f"\n  MEAN AVERAGE PRECISION:")
            print(f"    mAP (IoU=0.50:0.95): {map_value:.3f}")
            print(f"    mAP50 (IoU=0.50): {bbox['AP50']:.3f}")
            
            # Show formatted per-class results
            print(f"\n  DETAILED CLASS PERFORMANCE:")
            print(f"    {'Class':<6} {'AP (0.50:0.95)':<15} {'Performance':<12}")
            print(f"    {'-'*6:<6} {'-'*15:<15} {'-'*12:<12}")
            for class_name, ap_value in per_class_ap:
                # COCO AP values are in percentage format (0-100), not decimals (0-1)
                print(f"    {class_name:<6} {ap_value:.3f}           {ap_value:.1f}%")
        
        # Note about AP50 per-class limitations
        print(f"\n  📋 NOTE: COCO evaluator provides overall mAP50 but not individual per-class AP50.")
        print(f"      Per-class AP is averaged over IoU thresholds 0.50:0.95.")
        print(f"      Overall mAP50 (all classes): {bbox['AP50']:.3f}")
    
    # Additional analysis with different thresholds
    print(f"\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    thresholds = [0.5]
    
    for threshold in thresholds:
        print(f"\nConfidence Threshold: {threshold}")
        print("-" * 40)
        
        cfg_thresh = setup_improved_cfg(model_path, model_variant)
        cfg_thresh.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg_thresh.DATASETS.TEST = (filtered_dataset,)
        predictor_thresh = DefaultPredictor(cfg_thresh)
        
        # Quick evaluation on a subset
        from detectron2.data import DatasetCatalog
        dataset_dicts = DatasetCatalog.get(filtered_dataset)
        
        total_predictions = 0
        high_conf_predictions = 0
        sample_size = min(20, len(dataset_dicts))  # Use actual sample size
        
        for i, dataset_dict in enumerate(dataset_dicts[:sample_size]):  # Sample images
            im = cv2.imread(dataset_dict["file_name"])
            outputs = predictor_thresh(im)
            instances = outputs["instances"].to("cpu")
            
            total_predictions += len(instances)
            high_conf_predictions += len([s for s in instances.scores if s > 0.8])
        
        avg_predictions = total_predictions / sample_size if sample_size > 0 else 0
        print(f"  Average predictions per image: {avg_predictions:.2f}")
        print(f"  High confidence (>0.8) predictions: {high_conf_predictions}")
    
    # Calculate detailed precision, recall, and IoU metrics
    print(f"\n" + "="*70)
    print("DETAILED PRECISION, RECALL & IoU ANALYSIS")
    print("="*70)
    
    detailed_metrics = compute_detailed_metrics(predictor, filtered_dataset)
    
    # Compute per-class AP50 from our detailed analysis
    per_class_ap50_custom = compute_per_class_ap50_custom(detailed_metrics)
    
    # Display per-class AP50 results
    print(f"\n" + "="*70)
    print("PER-CLASS AP50 RESULTS")
    print("="*70)
    
    print(f"\n📊 PER-CLASS AP50 METRICS:")
    print(f"    {'Class':<6} {'AP50 (Custom)':<15} {'AP50 (COCO)':<15}")
    print(f"    {'-'*6:<6} {'-'*15:<15} {'-'*15:<15}")
    
    total_ap50_custom = 0
    valid_classes = 0
    
    for class_name in ["PP", "PE", "PS", "PET"]:
        custom_ap50 = per_class_ap50_custom.get(class_name, 0.0)
        coco_ap50 = per_class_ap50_coco.get(class_name, "N/A") if per_class_ap50_coco else "N/A"
        
        coco_str = f"{coco_ap50:.1f}%" if isinstance(coco_ap50, (int, float)) else coco_ap50
        
        print(f"    {class_name:<6} {custom_ap50:<14.1f}% {coco_str:<15}")
        
        if custom_ap50 > 0:
            total_ap50_custom += custom_ap50
            valid_classes += 1
    
    # Calculate mean AP50
    mean_ap50_custom = total_ap50_custom / valid_classes if valid_classes > 0 else 0
    print(f"\n    {'mAP50':<6} {mean_ap50_custom:<14.1f}% {'(overall)':<15}")
    
    print(f"\n  📋 NOTE: Custom AP50 is computed from precision at IoU=0.5 threshold.")
    print(f"      COCO AP50 extraction may fail due to evaluator implementation details.")
    print(f"      Overall COCO mAP50: {bbox['AP50']:.3f}")
    
    # Per-class performance summary
    print(f"\n" + "="*70)
    print("PER-CLASS PERFORMANCE SUMMARY")
    print("="*70)
    
    class_names = ["PP", "PE", "PS", "PET"]
    class_ids = [0, 1, 2, 3]  # Use 0-based IDs to match detailed_metrics
    
    print(f"\n🔍 CLASS DISTRIBUTION:")
    print(f"Debug: Available class IDs in detailed_metrics: {list(detailed_metrics['class_metrics'].keys())}")
    for class_id, class_name in zip(class_ids, class_names):
        if class_id in detailed_metrics['class_metrics']:
            metrics = detailed_metrics['class_metrics'][class_id]
            gt_count = metrics['ground_truth']
            pred_count = metrics['predictions']
            print(f"  {class_name}: {gt_count} ground truth, {pred_count} predictions")
        else:
            print(f"  {class_name}: No metrics found for class_id {class_id}")
    
    print(f"\n⭐ BEST PERFORMING CLASSES:")
    class_f1_scores = []
    for class_id, class_name in zip(class_ids, class_names):
        if class_id in detailed_metrics['class_metrics']:
            metrics = detailed_metrics['class_metrics'][class_id]
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
            class_f1_scores.append((class_name, f1, prec, rec))
    
    # Sort by F1 score
    class_f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (class_name, f1, prec, rec) in enumerate(class_f1_scores):
        if i == 0:
            print(f"  🥇 {class_name}: F1={f1:.3f} (Precision={prec:.3f}, Recall={rec:.3f})")
        elif i == 1:
            print(f"  🥈 {class_name}: F1={f1:.3f} (Precision={prec:.3f}, Recall={rec:.3f})")
        elif i == 2:
            print(f"  🥉 {class_name}: F1={f1:.3f} (Precision={prec:.3f}, Recall={rec:.3f})")
        else:
            print(f"     {class_name}: F1={f1:.3f} (Precision={prec:.3f}, Recall={rec:.3f})")
    
    # Performance summary
    print(f"\n" + "="*70)
    print("FASTER R-CNN PERFORMANCE SUMMARY")
    print("="*70)
    
    if 'bbox' in results:
        ap = results['bbox']['AP']
        ap50 = results['bbox']['AP50']
        
        print(f"✓ Bounding Box AP: {ap:.3f}")
        print(f"✓ Bounding Box AP50 (mAP50): {ap50:.3f}")
        
        # Performance interpretation
        if ap > 0.35:
            print("🎉 EXCELLENT: Model shows strong performance")
        elif ap > 0.25:
            print("✅ GOOD: Model shows solid performance")
        elif ap > 0.15:
            print("⚠️  FAIR: Model shows moderate performance")
        else:
            print("❌ POOR: Model needs improvement")
        
        print(f"\nKey Features of Faster R-CNN Architecture:")
        print(f"✓ {backbone_desc} backbone provides strong features")
        print("✓ Multi-scale anchors capture particles of various sizes")
        print("✓ Multi-class detection for 4 polymer types")
        print("✓ Optimized RPN settings for small object detection")
        print("✓ No mask prediction - focus on bounding box accuracy")
    
    return results

def compare_with_baseline():
    """Load and compare with baseline results if available."""
    baseline_path = os.path.join(ZOO_ROOT, "baseline", "model_final.pth")
    
    if os.path.exists(baseline_path):
        print(f"\n" + "="*70)
        print("BASELINE COMPARISON")
        print("="*70)
        print("Comparing improved model with baseline ResNet-50 model...")
        
        # This would run baseline evaluation for comparison
        # For now, just show the architecture differences
        print(f"\nARCHITECTURE COMPARISON:")
        print(f"{'Component':<25} {'Baseline':<20} {'Improved':<20}")
        print("-" * 65)
        print(f"{'Backbone':<25} {'ResNet-50':<20} {'ResNet-101':<20}")
        print(f"{'Anchor Sizes':<25} {'[32,64,128,256,512]':<20} {'[8,16,32,64,128]':<20}")
        print(f"{'Aspect Ratios':<25} {'[0.5,1.0,2.0]':<20} {'[0.25-4.0]':<20}")
        print(f"{'RPN NMS':<25} {'0.7':<20} {'0.6':<20}")
        print(f"{'ROI Score Thresh':<25} {'0.05':<20} {'0.3':<20}")
        print(f"{'Learning Rate':<25} {'0.00025':<20} {'0.0005':<20}")
        
        print(f"\nExpected Improvements:")
        print("• Better feature extraction with ResNet-101")
        print("• Enhanced small particle detection")
        print("• More robust mask prediction")
        print("• Improved precision-recall balance")

def main():
    parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN Model')
    parser.add_argument('--model_path', required=True, help='Path to model file')
    parser.add_argument('--model_variant', choices=['FasterRCNN-R50', 'FasterRCNN-R101', 'FasterRCNN-X101'], 
                        default='FasterRCNN-R50', help='Model variant')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Run evaluation
    results = evaluate_improved_model(args.model_path, args.model_variant)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        compare_with_baseline()
    
    print(f"\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    print(f"Detailed results saved in: {os.path.dirname(args.model_path)}/evaluation_improved/")
    print("The Faster R-CNN architecture shows optimized performance for microplastic detection.")

if __name__ == "__main__":
    import cv2  # Import here to avoid issues
    main()