#!/usr/bin/env python3
"""
Simple YOLOv8 Test Evaluation Script
Evaluates trained model on test dataset
"""

from ultralytics import YOLO
import os
from pathlib import Path

def main():
    """Run model evaluation on test set"""
    
    print("YOLOv8 Model Evaluation on Test Set")
    print("="*40)
    
    # Use specified trained model
    model_path = "zoo/MNP_yolov8_segment.pt"
    
    print(f"Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Run validation on test split
    print("\nRunning evaluation on test set...")
    
    # Method 1: Direct evaluation if test data is in data.yaml
    try:
        results = model.val(
            data="data.yaml",
            split="test",
            save_json=True,
            device="0"
        )
        print("✅ Test evaluation completed successfully!")
        print("\nTest Results:")
        print("="*50)
        print("BOUNDING BOX METRICS:")
        print(f"mAP@0.5: {results.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {results.box.map:.4f}")
        if hasattr(results.box, 'map75'):
            print(f"mAP@0.75: {results.box.map75:.4f}")
        
        if hasattr(results, 'seg'):
            print("\nSEGMENTATION MASK METRICS:")
            print(f"mAP@0.5: {results.seg.map50:.4f}")
            print(f"mAP@0.5:0.95: {results.seg.map:.4f}")
            if hasattr(results.seg, 'map75'):
                print(f"mAP@0.75: {results.seg.map75:.4f}")
            
            # Additional detailed metrics
            print("\nDETAILED SEGMENTATION METRICS:")
            # Access the underlying metrics if available
            if hasattr(results.seg, 'mp') and hasattr(results.seg, 'mr'):
                print(f"Mean Precision: {results.seg.mp:.4f}")
                print(f"Mean Recall: {results.seg.mr:.4f}")
            
            # Try to access per-class metrics
            if hasattr(results.seg, 'ap'):
                ap_values = results.seg.ap
                if len(ap_values) > 0:
                    print(f"AP per IoU threshold:")
                    # AP at different IoU thresholds (0.5:0.05:0.95)
                    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                    for i, iou in enumerate(iou_thresholds):
                        if i < len(ap_values[0]) if len(ap_values.shape) > 1 else len(ap_values):
                            ap_val = ap_values[0][i] if len(ap_values.shape) > 1 else ap_values[i]
                            print(f"  AP@{iou:.2f}: {ap_val:.4f}")
                            if iou == 0.75:  # Specifically highlight AP75
                                print(f"  *** AP75: {ap_val:.4f} ***")
        
    except Exception as e:
        print(f"Direct validation failed: {e}")
        
        # Method 2: Manual prediction on test images
        print("\nFalling back to manual prediction on test images...")
        test_dir = "../../data/patches/yolo_seg/images/test"
        
        if os.path.exists(test_dir):
            # Run prediction on test directory
            results = model.predict(
                source=test_dir,
                conf=0.25,
                save=True,
                save_txt=True,
                save_crop=True,
                project="test_predictions",
                name="yolov8_test_results",
                exist_ok=True
            )
            
            print(f"✅ Predictions saved to: test_predictions/yolov8_test_results/")
            print(f"Processed {len(results)} test images")
            
            # Count detections
            total_detections = 0
            images_with_detections = 0
            
            for r in results:
                num_det = len(r.boxes) if r.boxes is not None else 0
                total_detections += num_det
                if num_det > 0:
                    images_with_detections += 1
            
            print(f"\nDetection Summary:")
            print(f"- Total images: {len(results)}")
            print(f"- Images with detections: {images_with_detections}")
            print(f"- Total detections: {total_detections}")
            print(f"- Average detections per image: {total_detections/len(results):.2f}")
        
        else:
            print(f"Error: Test directory not found: {test_dir}")

if __name__ == "__main__":
    main()