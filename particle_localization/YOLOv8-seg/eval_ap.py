#!/usr/bin/env python3
"""
Comprehensive YOLOv8 Test Evaluation with Detailed Metrics
Reports AP75 for both bounding boxes and segmentation masks
"""

from ultralytics import YOLO
import os
import sys
import json
import numpy as np
from pathlib import Path

def extract_detailed_metrics(results):
    """Extract detailed AP metrics from validation results"""
    detailed_metrics = {}
    
    # Bounding box metrics
    if hasattr(results, 'box'):
        box_metrics = {
            'map50': float(results.box.map50),
            'map': float(results.box.map),
            'mp': float(results.box.mp),
            'mr': float(results.box.mr),
            'map75': 0.0,
        }
        
        # Try to get AP75 and size-specific APs
        if hasattr(results.box, 'map75'):
            box_metrics['map75'] = float(results.box.map75)
        
        detailed_metrics['box'] = box_metrics
    
    # Segmentation metrics
    if hasattr(results, 'seg'):
        seg_metrics = {
            'map50': float(results.seg.map50),
            'map': float(results.seg.map),
            'mp': float(results.seg.mp),
            'mr': float(results.seg.mr),
            'map75': 0.0,
        }
        
        # Try to get AP75 and size-specific APs
        if hasattr(results.seg, 'map75'):
            seg_metrics['map75'] = float(results.seg.map75)
        
        # Extract per-IoU AP values if available
        if hasattr(results.seg, 'ap'):
            ap_values = results.seg.ap
            iou_aps = []
            if len(ap_values.shape) > 1:
                # ap_values is typically [num_classes, num_iou_thresholds]
                for i in range(ap_values.shape[1]):
                    iou_aps.append(float(ap_values[0, i]))  # For class 0 (MNP)
            else:
                iou_aps = [float(x) for x in ap_values]
            
            seg_metrics['ap_per_iou'] = iou_aps
            
            # Extract AP75 specifically (IoU=0.75 is typically at index 5 in 0.5:0.05:0.95)
            if len(iou_aps) > 5:
                seg_metrics['map75'] = iou_aps[5]
        
        detailed_metrics['seg'] = seg_metrics
    
    return detailed_metrics

def main():
    """Run comprehensive evaluation with detailed metrics"""
    
    print("YOLOv8 Comprehensive Test Evaluation")
    print("="*50)
    
    # Find the best trained model
    model_path = None
    
    # Check local runs directory first
    runs_dir = Path("runs/segment")
    if runs_dir.exists():
        yolov8_runs = [d for d in runs_dir.iterdir() if d.is_dir() and "yolov8" in d.name]
        if yolov8_runs:
            latest_run = max(yolov8_runs, key=lambda x: x.stat().st_mtime)
            best_model = latest_run / "weights" / "best.pt"
            if best_model.exists():
                model_path = str(best_model)
    
    if not model_path:
        # Never fall back to the COCO-pretrained weights: their scores would look plausible but mean nothing
        sys.exit("No trained YOLOv8 model found in runs/segment. Train one first (python train.py).")
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run comprehensive validation
    print("\nRunning comprehensive evaluation on test set...")
    
    try:
        # Run validation with detailed metrics enabled
        results = model.val(
            data="data.yaml",
            split="test",
            save_json=True,
            conf=0.001,  # Lower confidence to capture more objects
            device="0",
            verbose=True,
            plots=True
        )
        
        print("✅ Test evaluation completed successfully!")
        
        # Extract and display detailed metrics
        detailed_metrics = extract_detailed_metrics(results)
        
        print("\n" + "="*60)
        print("DETAILED TEST SET EVALUATION RESULTS")
        print("="*60)
        
        # Display bounding box metrics
        if 'box' in detailed_metrics:
            box = detailed_metrics['box']
            print("\n📦 BOUNDING BOX METRICS:")
            print(f"   mAP@0.5      : {box['map50']:.4f}")
            print(f"   mAP@0.5:0.95 : {box['map']:.4f}")
            print(f"   mAP@0.75     : {box['map75']:.4f}")
            print(f"   Mean Precision: {box['mp']:.4f}")
            print(f"   Mean Recall   : {box['mr']:.4f}")
        
        # Display segmentation metrics
        if 'seg' in detailed_metrics:
            seg = detailed_metrics['seg']
            print("\n🎯 SEGMENTATION MASK METRICS:")
            print(f"   mAP@0.5      : {seg['map50']:.4f}")
            print(f"   mAP@0.5:0.95 : {seg['map']:.4f}")
            print(f"   mAP@0.75     : {seg['map75']:.4f}")
            print(f"   Mean Precision: {seg['mp']:.4f}")
            print(f"   Mean Recall   : {seg['mr']:.4f}")
            
            # Display per-IoU AP values if available
            if 'ap_per_iou' in seg:
                print("\n   📊 AP at Different IoU Thresholds:")
                iou_thresholds = np.arange(0.5, 1.0, 0.05)
                for i, (iou, ap) in enumerate(zip(iou_thresholds, seg['ap_per_iou'][:len(iou_thresholds)])):
                    marker = " ⭐" if abs(iou - 0.75) < 0.01 else ""
                    print(f"      AP@{iou:.2f}: {ap:.4f}{marker}")
        
        # Save detailed results to JSON
        output_file = "detailed_test_metrics.json"
        detailed_metrics['model_path'] = model_path
        detailed_metrics['test_images'] = 119  # From your test set
        
        with open(output_file, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        
        print(f"\n💾 Detailed metrics saved to: {output_file}")
        
        # Summary for paper/reporting
        print("\n" + "="*60)
        print("SUMMARY FOR REPORTING")
        print("="*60)
        if 'seg' in detailed_metrics:
            seg = detailed_metrics['seg']
            print(f"Segmentation Performance on Test Set (119 images):")
            print(f"• mAP@0.5      = {seg['map50']:.3f}")
            print(f"• mAP@0.5:0.95 = {seg['map']:.3f}")
            print(f"• mAP@0.75     = {seg['map75']:.3f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()