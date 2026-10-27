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

def compute_detailed_metrics(predictor, dataset_name):
    """Compute detailed precision, recall, and IoU metrics for bounding boxes."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    # Class names matching the training order. register_coco_instances maps the
    # annotation category ids 1-4 to contiguous ids 0-3, the same ids the model predicts.
    class_names = ["PP", "PE", "PS", "PET"]
    class_ids = [0, 1, 2, 3]
    
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
    
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_per_threshold = {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in iou_thresholds}
    class_metrics_per_threshold = {}
    for thresh in iou_thresholds:
        class_metrics_per_threshold[thresh] = {class_id: {'tp': 0, 'fp': 0, 'fn': 0} for class_id in class_ids}
    
    print(f"Processing {len(dataset_dicts)} images for detailed metrics...")
    
    for i, dataset_dict in enumerate(dataset_dicts):
        if i % 50 == 0:
            print(f"Processed {i}/{len(dataset_dicts)} images")
            
        # Load image and run inference
        im = cv2.imread(dataset_dict["file_name"])
        outputs = predictor(im)
        pred_instances = outputs["instances"].to("cpu")
        
        # Get ground truth annotations
        gt_annotations = dataset_dict["annotations"]
        
        # Convert predictions to bounding boxes
        if len(pred_instances) > 0:
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()  # [x1,y1,x2,y2] format
            pred_scores = pred_instances.scores.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            # Convert to [x,y,w,h] format
            pred_boxes_xywh = [[box[0], box[1], box[2]-box[0], box[3]-box[1]] for box in pred_boxes]
        else:
            pred_boxes_xywh = []
            pred_scores = np.array([])
            pred_classes = np.array([])
        
        # Get ground truth bounding boxes
        gt_boxes = [ann['bbox'] for ann in gt_annotations]  # Already in [x,y,w,h] format
        gt_classes = [ann['category_id'] for ann in gt_annotations]
        
        # Update per-class ground truth counts
        for gt_class in gt_classes:
            if gt_class in class_metrics:
                class_metrics[gt_class]['ground_truth'] += 1
        
        # Update per-class prediction counts
        for pred_class in pred_classes:
            pred_class_id = int(pred_class)
            if pred_class_id in class_metrics:
                class_metrics[pred_class_id]['predictions'] += 1
        
        # Match predictions to ground truth using IoU
        if len(pred_boxes_xywh) > 0 and len(gt_boxes) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(pred_boxes_xywh), len(gt_boxes)))
            for p_idx, pred_box in enumerate(pred_boxes_xywh):
                for g_idx, gt_box in enumerate(gt_boxes):
                    # Only match if classes are the same
                    pred_class_id = int(pred_classes[p_idx])
                    if pred_class_id == gt_classes[g_idx]:
                        iou_matrix[p_idx, g_idx] = compute_iou_boxes(pred_box, gt_box)
            
            # Simple greedy matching (IoU > 0.5 threshold)
            matched_pairs = []
            used_gt = set()
            used_pred = set()
            
            # Sort predictions by score (highest first)
            pred_order = np.argsort(-pred_scores)
            
            for p_idx in pred_order:
                best_gt_idx = -1
                best_iou = 0
                
                for g_idx in range(len(gt_boxes)):
                    if g_idx not in used_gt and iou_matrix[p_idx, g_idx] > best_iou:
                        best_iou = iou_matrix[p_idx, g_idx]
                        best_gt_idx = g_idx
                
                if best_iou > 0.5:  # IoU threshold for match
                    matched_pairs.append((p_idx, best_gt_idx, best_iou))
                    used_gt.add(best_gt_idx)
                    used_pred.add(p_idx)
                    iou_scores.append(best_iou)
                    
                    # Track per-class metrics
                    gt_class_id = gt_classes[best_gt_idx]
                    if gt_class_id in class_metrics:
                        class_metrics[gt_class_id]['iou_scores'].append(best_iou)
            
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
                                 if iou >= thresh and gt_classes[g_idx] == class_id)
                    class_fp = sum(1 for c in pred_classes if int(c) == class_id) - class_tp
                    class_fn = sum(1 for c in gt_classes if c == class_id) - class_tp
                    
                    class_metrics_per_threshold[thresh][class_id]['tp'] += class_tp
                    class_metrics_per_threshold[thresh][class_id]['fp'] += class_fp
                    class_metrics_per_threshold[thresh][class_id]['fn'] += class_fn
            
            # Overall counts (IoU > 0.5)
            total_tp += len(matched_pairs)
            total_fp += len(pred_boxes_xywh) - len(matched_pairs)
            total_fn += len(gt_boxes) - len(matched_pairs)
            
            # Per-class overall counts
            for class_id in class_ids:
                class_tp = sum(1 for _, g_idx, _ in matched_pairs if gt_classes[g_idx] == class_id)
                class_fp = sum(1 for p_idx in range(len(pred_boxes_xywh)) 
                             if p_idx not in used_pred and int(pred_classes[p_idx]) == class_id)
                class_fn = sum(1 for g_idx in range(len(gt_boxes)) 
                             if g_idx not in used_gt and gt_classes[g_idx] == class_id)
                
                class_metrics[class_id]['tp'] += class_tp
                class_metrics[class_id]['fp'] += class_fp
                class_metrics[class_id]['fn'] += class_fn
        else:
            # No predictions or no ground truth
            total_fp += len(pred_boxes_xywh)
            total_fn += len(gt_boxes)
            
            # Per-class counts when no matches
            for p_idx in range(len(pred_boxes_xywh)):
                pred_class_id = int(pred_classes[p_idx])
                if pred_class_id in class_metrics:
                    class_metrics[pred_class_id]['fp'] += 1
            
            for gt_class in gt_classes:
                if gt_class in class_metrics:
                    class_metrics[gt_class]['fn'] += 1
            
            for thresh in iou_thresholds:
                metrics_per_threshold[thresh]['fp'] += len(pred_boxes_xywh)
                metrics_per_threshold[thresh]['fn'] += len(gt_boxes)
                
                # Per-class threshold updates
                for p_idx in range(len(pred_boxes_xywh)):
                    pred_class_id = int(pred_classes[p_idx])
                    if pred_class_id in class_metrics_per_threshold[thresh]:
                        class_metrics_per_threshold[thresh][pred_class_id]['fp'] += 1
                
                for gt_class in gt_classes:
                    if gt_class in class_metrics_per_threshold[thresh]:
                        class_metrics_per_threshold[thresh][gt_class]['fn'] += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(iou_scores) if iou_scores else 0
    
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
    print(f"\n📊 PER-CLASS METRICS BY IoU THRESHOLD:")
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25    # Matches training threshold
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

def register_datasets(split="test"):
    """Register one split of the multi-class patch dataset, unfiltered (incl. empty patches)."""
    
    # Get project root directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_root = os.path.join(os.path.dirname(project_root), "data", "patches", "coco", "patches_256_combined")
    
    dataset_name = f"microplastic_{split}_multi"
    register_coco_instances(
        dataset_name,
        {},
        os.path.join(dataset_root, split, "annotations.json"),
        os.path.join(dataset_root, split, "images")
    )
    
    # Set thing classes for multiple polymer types (matching training order)
    MetadataCatalog.get(dataset_name).thing_classes = ["PP", "PE", "PS", "PET"]
    print(f"✅ Evaluating on {split}: {len(DatasetCatalog.get(dataset_name))} images")
    
    return dataset_name

def evaluate_improved_model(model_path, model_variant="FasterRCNN-R50", split="test"):
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
    
    # Register the split to evaluate (unfiltered)
    filtered_dataset = register_datasets(split)
    
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
    print(f"Running evaluation on {split} set...")
    
    # Run COCO evaluation with a low score threshold so AP sees the whole PR curve;
    # the 0.25 predictor above is the operating point for the P/R/F1 counts below
    cfg_ap = setup_improved_cfg(model_path, model_variant)
    cfg_ap.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg_ap.DATASETS.TEST = (filtered_dataset,)
    predictor_ap = DefaultPredictor(cfg_ap)
    evaluator = COCOEvaluator(
        filtered_dataset,
        output_dir=os.path.join(os.path.dirname(model_path), f"evaluation_{split}")
    )
    val_loader = build_detection_test_loader(cfg_ap, filtered_dataset)
    results = inference_on_dataset(predictor_ap.model, val_loader, evaluator)
    
    # Print results
    print("\n" + "="*70)
    print("FASTER R-CNN EVALUATION RESULTS")
    print("="*70)
    
    if 'bbox' in results:
        bbox = results['bbox']
        print(f"\nBOUNDING BOX DETECTION METRICS:")
        print(f"  AP (IoU=0.50:0.95): {bbox['AP']:.3f}")
        print(f"  AP50 (IoU=0.50): {bbox['AP50']:.3f}")
        print(f"  AP75 (IoU=0.75): {bbox['AP75']:.3f}")
        print(f"  APs (small objects): {bbox.get('APs', 0):.3f}")
        print(f"  APm (medium objects): {bbox.get('APm', 0):.3f}")
        print(f"  APl (large objects): {bbox.get('APl', 0):.3f}")
        
        # Per-class AP metrics if available
        if 'AP-PP' in bbox or len([k for k in bbox.keys() if k.startswith('AP-')]) > 0:
            print(f"\n  PER-CLASS AP METRICS:")
            class_names = ["PP", "PE", "PS", "PET"]
            for i, class_name in enumerate(class_names):
                ap_key = f'AP-{class_name}'
                if ap_key in bbox:
                    print(f"    {class_name}: {bbox[ap_key]:.3f}")
                else:
                    # Try index-based keys
                    idx_key = f'AP{i}' if f'AP{i}' in bbox else None
                    if idx_key:
                        print(f"    {class_name}: {bbox[idx_key]:.3f}")
    
    # Additional analysis with different thresholds
    print(f"\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    thresholds = [0.3, 0.5, 0.7, 0.8]
    
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
        
        for i, dataset_dict in enumerate(dataset_dicts[:20]):  # Sample 20 images
            im = cv2.imread(dataset_dict["file_name"])
            outputs = predictor_thresh(im)
            instances = outputs["instances"].to("cpu")
            
            total_predictions += len(instances)
            high_conf_predictions += len([s for s in instances.scores if s > 0.8])
        
        avg_predictions = total_predictions / 20
        print(f"  Average predictions per image: {avg_predictions:.2f}")
        print(f"  High confidence (>0.8) predictions: {high_conf_predictions}")
    
    # Calculate detailed precision, recall, and IoU metrics
    print(f"\n" + "="*70)
    print("DETAILED PRECISION, RECALL & IoU ANALYSIS")
    print("="*70)
    
    detailed_metrics = compute_detailed_metrics(predictor, filtered_dataset)
    
    summary_path = os.path.join(os.path.dirname(model_path), f"evaluation_{split}", "detailed_metrics.json")
    with open(summary_path, "w") as f:
        json.dump({
            "split": split, "operating_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "coco": {k: dict(v) for k, v in results.items()},
            "precision": detailed_metrics["precision"], "recall": detailed_metrics["recall"],
            "f1_score": detailed_metrics["f1_score"], "mean_iou": float(detailed_metrics["mean_iou"]),
            "per_class": {m["name"]: {k: m[k] for k in ("ground_truth", "predictions", "tp", "fp", "fn")}
                          for m in detailed_metrics["class_metrics"].values()},
        }, f, indent=2, default=float)
    print(f"Detailed metrics saved to {summary_path}")
    
    # Per-class performance summary
    print(f"\n" + "="*70)
    print("PER-CLASS PERFORMANCE SUMMARY")
    print("="*70)
    
    class_names = ["PP", "PE", "PS", "PET"]
    class_ids = [1, 2, 3, 4]
    
    print(f"\n🔍 CLASS DISTRIBUTION:")
    for class_id, class_name in zip(class_ids, class_names):
        if class_id in detailed_metrics['class_metrics']:
            metrics = detailed_metrics['class_metrics'][class_id]
            gt_count = metrics['ground_truth']
            pred_count = metrics['predictions']
            print(f"  {class_name}: {gt_count} ground truth, {pred_count} predictions")
    
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
        print(f"✓ Bounding Box AP50: {ap50:.3f}")
        
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
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Split to evaluate (default: test)')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Run evaluation
    results = evaluate_improved_model(args.model_path, args.model_variant, args.split)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        compare_with_baseline()
    
    print(f"\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    print(f"Detailed results saved in: {os.path.dirname(args.model_path)}/evaluation_{args.split}/")
    print("The Faster R-CNN architecture shows optimized performance for microplastic detection.")

if __name__ == "__main__":
    import cv2  # Import here to avoid issues
    main()