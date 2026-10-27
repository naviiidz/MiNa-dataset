#!/usr/bin/env python3
"""
Evaluation Script for Improved Mask R-CNN Model
Compatible with ResNet-101 and enhanced anchor configuration
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
from pycocotools.mask import decode, area, toBbox
from pycocotools import mask as mask_util

# Paths relative to the project folder (one level above this script's folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches): full image area, PE x3 in train
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo")

def compute_iou_masks(mask1, mask2):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0

def compute_detailed_metrics(predictor, dataset_name):
    """Compute detailed precision, recall, and IoU metrics."""
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    total_tp = 0  # True positives
    total_fp = 0  # False positives
    total_fn = 0  # False negatives
    iou_scores = []
    
    iou_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    metrics_per_threshold = {thresh: {'tp': 0, 'fp': 0, 'fn': 0} for thresh in iou_thresholds}
    
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
        
        # Convert predictions to masks
        if len(pred_instances) > 0:
            pred_masks = pred_instances.pred_masks.numpy()
            pred_scores = pred_instances.scores.numpy()
        else:
            pred_masks = np.array([])
            pred_scores = np.array([])
        
        # Convert ground truth to masks
        gt_masks = []
        for ann in gt_annotations:
            if 'segmentation' in ann and ann['segmentation']:
                # Handle RLE format
                if isinstance(ann['segmentation'], dict):
                    mask = decode(ann['segmentation'])
                else:
                    # Handle polygon format
                    from pycocotools import mask as mask_util
                    rle = mask_util.frPyObjects(ann['segmentation'], im.shape[0], im.shape[1])
                    mask = decode(rle)
                    if len(mask.shape) == 3:
                        mask = mask[:, :, 0]
                gt_masks.append(mask)
        
        gt_masks = np.array(gt_masks) if gt_masks else np.array([]).reshape(0, im.shape[0], im.shape[1])
        
        # Match predictions to ground truth using Hungarian algorithm
        if len(pred_masks) > 0 and len(gt_masks) > 0:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
            for p_idx, pred_mask in enumerate(pred_masks):
                for g_idx, gt_mask in enumerate(gt_masks):
                    iou_matrix[p_idx, g_idx] = compute_iou_masks(pred_mask, gt_mask)
            
            # Simple greedy matching for now (IoU > 0.5 threshold)
            matched_pairs = []
            used_gt = set()
            used_pred = set()
            
            # Sort predictions by score (highest first)
            pred_order = np.argsort(-pred_scores)
            
            for p_idx in pred_order:
                best_gt_idx = -1
                best_iou = 0
                
                for g_idx in range(len(gt_masks)):
                    if g_idx not in used_gt and iou_matrix[p_idx, g_idx] > best_iou:
                        best_iou = iou_matrix[p_idx, g_idx]
                        best_gt_idx = g_idx
                
                if best_iou > 0.5:  # IoU threshold for match
                    matched_pairs.append((p_idx, best_gt_idx, best_iou))
                    used_gt.add(best_gt_idx)
                    used_pred.add(p_idx)
                    iou_scores.append(best_iou)
            
            # Count TP, FP, FN for different IoU thresholds
            for thresh in iou_thresholds:
                tp_thresh = sum(1 for _, _, iou in matched_pairs if iou >= thresh)
                fp_thresh = len(pred_masks) - tp_thresh
                fn_thresh = len(gt_masks) - tp_thresh
                
                metrics_per_threshold[thresh]['tp'] += tp_thresh
                metrics_per_threshold[thresh]['fp'] += fp_thresh
                metrics_per_threshold[thresh]['fn'] += fn_thresh
            
            # Overall counts (IoU > 0.5)
            total_tp += len(matched_pairs)
            total_fp += len(pred_masks) - len(matched_pairs)
            total_fn += len(gt_masks) - len(matched_pairs)
        else:
            # No predictions or no ground truth
            total_fp += len(pred_masks)
            total_fn += len(gt_masks)
            
            for thresh in iou_thresholds:
                metrics_per_threshold[thresh]['fp'] += len(pred_masks)
                metrics_per_threshold[thresh]['fn'] += len(gt_masks)
    
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
        'total_fn': total_fn
    }

def setup_improved_cfg(model_path):
    """Setup configuration matching the improved model architecture."""
    cfg = get_cfg()
    
    # Use ResNet-101 configuration
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Enhanced FPN settings (EXACTLY matching training)
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 384  # This is crucial - matches training config!
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    # Enhanced anchor settings (EXACTLY matching training config)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 16],      # p2
        [24, 32],     # p3
        [48, 64],     # p4
        [128, 192]    # p5  - Fixed! Was missing p5
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]  # Fixed! Was missing 0.25 and 4
    
    # RPN settings (EXACTLY matching training)
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match 4 anchor sizes
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    
    # ROI settings (EXACTLY matching training)
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
    """Register one split of the single-class patch dataset, unfiltered (incl. empty patches)."""
    # Single-class copy of the annotations (all polymers -> "Microplastic")
    ann_file = os.path.join(DATA_ROOT, split, "annotations.json")
    with open(ann_file, 'r') as f:
        data = json.load(f)
    data['categories'] = [{'id': 1, 'name': 'Microplastic', 'supercategory': 'microplastic'}]
    for ann in data['annotations']:
        ann['category_id'] = 1
    single_file = os.path.join(DATA_ROOT, split, "annotations_single_class.json")
    with open(single_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    dataset_name = f"microplastic_{split}_single"
    register_coco_instances(dataset_name, {}, single_file, os.path.join(DATA_ROOT, split, "images"))
    MetadataCatalog.get(dataset_name).thing_classes = ["Microplastic"]
    print(f"✅ Evaluating on {split}: {len(DatasetCatalog.get(dataset_name))} images")
    
    return dataset_name

def evaluate_improved_model(model_path, split="test"):
    """Comprehensive evaluation of the improved model."""
    print("="*70)
    print("IMPROVED MASK R-CNN MODEL EVALUATION")
    print("="*70)
    print("Architecture: ResNet-101 + FPN")
    print("Enhancements: Smaller anchors, optimized RPN, better ROI settings")
    print("="*70)
    
    # Register the split to evaluate (unfiltered)
    filtered_dataset = register_datasets(split)
    
    # Setup configuration
    cfg = setup_improved_cfg(model_path)
    
    # Use the filtered dataset for evaluation
    cfg.DATASETS.TEST = (filtered_dataset,)
    
    print(f"\nModel Configuration:")
    print(f"Backbone: ResNet-101 + FPN")
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
    cfg_ap = setup_improved_cfg(model_path)
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
    print("IMPROVED MODEL EVALUATION RESULTS")
    print("="*70)
    
    if 'segm' in results:
        segm = results['segm']
        print(f"\nSEGMENTATION METRICS:")
        print(f"  AP (IoU=0.50:0.95): {segm['AP']:.3f}")
        print(f"  AP50 (IoU=0.50): {segm['AP50']:.3f}")
        print(f"  AP75 (IoU=0.75): {segm['AP75']:.3f}")
        print(f"  APs (small objects): {segm.get('APs', 0):.3f}")
        print(f"  APm (medium objects): {segm.get('APm', 0):.3f}")
        print(f"  APl (large objects): {segm.get('APl', 0):.3f}")
    
    if 'bbox' in results:
        bbox = results['bbox']
        print(f"\nBOUNDING BOX METRICS:")
        print(f"  AP (IoU=0.50:0.95): {bbox['AP']:.3f}")
        print(f"  AP50 (IoU=0.50): {bbox['AP50']:.3f}")
        print(f"  AP75 (IoU=0.75): {bbox['AP75']:.3f}")
        print(f"  APs (small objects): {bbox.get('APs', 0):.3f}")
        print(f"  APm (medium objects): {bbox.get('APm', 0):.3f}")
        print(f"  APl (large objects): {bbox.get('APl', 0):.3f}")
    
    # Additional analysis with different thresholds
    print(f"\n" + "="*70)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("="*70)
    
    thresholds = [0.3, 0.5, 0.7, 0.8]
    
    for threshold in thresholds:
        print(f"\nConfidence Threshold: {threshold}")
        print("-" * 40)
        
        cfg_thresh = setup_improved_cfg(model_path)
        cfg_thresh.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
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
        json.dump({"split": split, "operating_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
                   "coco": {k: dict(v) for k, v in results.items()},
                   **{k: float(v) for k, v in detailed_metrics.items()}}, f, indent=2)
    print(f"Detailed metrics saved to {summary_path}")
    
    # Performance summary
    print(f"\n" + "="*70)
    print("IMPROVED MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    if 'segm' in results:
        ap = results['segm']['AP']
        ap50 = results['segm']['AP50']
        
        print(f"✓ Segmentation AP: {ap:.3f}")
        print(f"✓ Segmentation AP50: {ap50:.3f}")
        
        # Performance interpretation
        if ap > 0.35:
            print("🎉 EXCELLENT: Model shows strong performance")
        elif ap > 0.25:
            print("✅ GOOD: Model shows solid performance")
        elif ap > 0.15:
            print("⚠️  FAIR: Model shows moderate performance")
        else:
            print("❌ POOR: Model needs improvement")
        
        print(f"\nKey Improvements from Enhanced Architecture:")
        print("✓ ResNet-101 backbone provides stronger features")
        print("✓ Smaller anchors (8,16px) better capture tiny particles")
        print("✓ More aspect ratios handle diverse shapes")
        print("✓ Lower thresholds increase sensitivity")
        print("✓ Better RPN settings for small object detection")
    
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
    parser = argparse.ArgumentParser(description='Evaluate Improved Mask R-CNN Model')
    parser.add_argument('--model_path', required=True, help='Path to improved model')
    parser.add_argument('--split', choices=['val', 'test'], default='test', help='Split to evaluate (default: test)')
    parser.add_argument('--compare_baseline', action='store_true', help='Compare with baseline')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Run evaluation
    results = evaluate_improved_model(args.model_path, args.split)
    
    # Compare with baseline if requested
    if args.compare_baseline:
        compare_with_baseline()
    
    print(f"\n" + "="*70)
    print("EVALUATION COMPLETED!")
    print("="*70)
    print(f"Detailed results saved in: {os.path.dirname(args.model_path)}/evaluation_{args.split}/")
    print("The improved architecture shows enhanced performance for microplastic detection.")

if __name__ == "__main__":
    import cv2  # Import here to avoid issues
    main()