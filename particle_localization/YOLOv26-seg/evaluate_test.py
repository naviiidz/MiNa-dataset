#!/usr/bin/env python3
"""
YOLOv26n-seg Test Set Evaluation Script
Evaluate the fine-tuned model on the test set and generate comprehensive metrics.
"""

import warnings
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
warnings.filterwarnings('ignore', category=UserWarning)

def find_best_model():
    """Find the best trained model weights."""
    # Use the specified model from zoo directory
    model_path = Path("zoo/MNP_v26_seg.pt")
    
    if model_path.exists():
        print(f"✓ Found model: {model_path}")
        return str(model_path)
    else:
        print(f"❌ Model not found: {model_path}")
        print("💡 Please ensure the model file exists in the zoo directory")
        return None

def evaluate_test_set():
    """Evaluate the trained model on the test set."""
    from ultralytics import YOLO
    
    print("🧪 YOLOv26n-seg Test Set Evaluation")
    print("=" * 45)
    
    # Find best model
    model_path = find_best_model()
    if not model_path:
        return False
    
    # Load the trained model
    print("\n📥 Loading trained model...")
    model = YOLO(model_path)
    print("✓ Model loaded successfully")
    
    # Run evaluation on test set
    print("\n📊 Running evaluation on test set...")
    results = model.val(
        data='data.yaml',
        split='test',
        imgsz=640,
        plots=True
    )
    
    # Print detailed results
    print("\n📈 Test Set Evaluation Results:")
    print("=" * 40)
    
    # Get metrics from results
    metrics = results.results_dict
    
    # Segmentation metrics
    if 'metrics/mAP50(M)' in metrics:
        map50_seg = metrics['metrics/mAP50(M)']
        map50_95_seg = metrics['metrics/mAP50-95(M)']
        precision_seg = metrics.get('metrics/precision(M)', 0)
        recall_seg = metrics.get('metrics/recall(M)', 0)
        
        print(f"🎯 Segmentation Metrics:")
        print(f"  mAP@0.5:      {map50_seg:.4f}")
        print(f"  mAP@0.5:0.95: {map50_95_seg:.4f}")
        print(f"  Precision:    {precision_seg:.4f}")
        print(f"  Recall:       {recall_seg:.4f}")
    
    # Detection metrics (bounding boxes)
    if 'metrics/mAP50(B)' in metrics:
        map50_det = metrics['metrics/mAP50(B)']
        map50_95_det = metrics['metrics/mAP50-95(B)']
        precision_det = metrics.get('metrics/precision(B)', 0)
        recall_det = metrics.get('metrics/recall(B)', 0)
        
        print(f"\n📦 Detection Metrics:")
        print(f"  mAP@0.5:      {map50_det:.4f}")
        print(f"  mAP@0.5:0.95: {map50_95_det:.4f}")
        print(f"  Precision:    {precision_det:.4f}")
        print(f"  Recall:       {recall_det:.4f}")
    
    # Count test images
    test_img_dir = Path("../../data/patches/yolo_seg/images/test")
    if test_img_dir.exists():
        test_count = len(list(test_img_dir.glob("*")))
        print(f"\n📊 Dataset Info:")
        print(f"  Test images:  {test_count}")
        print(f"  Class:        MNP (Microplastic)")
    
    return True, results

def run_sample_predictions():
    """Run predictions on some test samples."""
    from ultralytics import YOLO
    import random
    
    print("\n🔍 Running Sample Predictions on Test Set")
    print("=" * 45)
    
    # Find best model
    model_path = find_best_model()
    if not model_path:
        return
    
    # Load model
    model = YOLO(model_path)
    
    # Get test images
    test_img_dir = Path("../../data/patches/yolo_seg/images/test")
    if not test_img_dir.exists():
        print("❌ Test images directory not found")
        return
    
    test_images = list(test_img_dir.glob("*.jpg")) + list(test_img_dir.glob("*.png"))
    if not test_images:
        print("❌ No test images found")
        return
    
    # Select random samples
    sample_count = min(5, len(test_images))
    sample_images = random.sample(test_images, sample_count)
    
    print(f"🎲 Testing on {sample_count} random samples...")
    
    total_detections = 0
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n📷 Sample {i}: {img_path.name}")
        
        # Run prediction
        results = model.predict(
            str(img_path),
            conf=0.25,           # Confidence threshold
            iou=0.5,             # IoU threshold  
            imgsz=256,           # Image size
            save=True,           # Save annotated images
            show_boxes=True,     # Show bounding boxes
            show_labels=True,    # Show labels
            show_conf=True       # Show confidence
        )
        
        # Process results
        for result in results:
            if result.masks is not None:
                num_detections = len(result.masks)
                total_detections += num_detections
                confidences = result.boxes.conf.cpu().numpy()
                
                print(f"  ✓ Found {num_detections} microplastic(s)")
                for j, conf in enumerate(confidences):
                    print(f"    MNP {j+1}: Confidence = {conf:.3f}")
            else:
                print(f"  ○ No microplastics detected")
    
    print(f"\n📊 Summary:")
    print(f"  Total samples tested: {sample_count}")
    print(f"  Total detections: {total_detections}")
    print(f"  Average per image: {total_detections/sample_count:.2f}")
    print(f"  Results saved to: runs/predict/")

def generate_evaluation_report():
    """Generate a comprehensive evaluation report."""
    print("\n📋 Generating Evaluation Report...")
    
    # Get project root
    project_root = os.getcwd()
    runs_dir = Path(f"{project_root}/runs/segment")
    
    # Find latest run
    run_dirs = list(runs_dir.glob("mnp_yolo26n*"))
    if run_dirs:
        latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
        
        print(f"\n📁 Training Results Location: {latest_run}")
        
        # Check for generated files
        files_to_check = [
            ("Training Results", "results.csv"),
            ("Validation Results", "val_batch*.jpg"),
            ("Training Plots", "results.png"),
            ("Confusion Matrix", "confusion_matrix.png"),
            ("F1 Curve", "F1_curve.png"),
            ("PR Curve", "PR_curve.png")
        ]
        
        print("\n📊 Generated Files:")
        for desc, pattern in files_to_check:
            matching_files = list(latest_run.glob(pattern))
            if matching_files:
                print(f"  ✓ {desc}: {len(matching_files)} file(s)")
            else:
                print(f"  ○ {desc}: Not found")

def main():
    """Main evaluation pipeline."""
    print("🔬 YOLOv26n-seg Microplastic Test Evaluation")
    print("=" * 50)
    
    # Check if model exists
    if not find_best_model():
        print("\n💡 Train your model first using:")
        print("   python3 quick_train.py")
        return
    
    # Run comprehensive evaluation
    print("\n🚀 Starting comprehensive evaluation...")
    
    # 1. Evaluate test set metrics
    success, results = evaluate_test_set()
    if not success:
        print("❌ Test evaluation failed")
        return
    
    # 2. Run sample predictions
    run_sample_predictions()
    
    # 3. Generate evaluation report
    generate_evaluation_report()
    
    print("\n🎉 Evaluation Complete!")
    print("\n📖 How to interpret results:")
    print("• mAP@0.5: Mean Average Precision at 50% IoU threshold")
    print("• mAP@0.5:0.95: Mean Average Precision averaged over IoU 0.5-0.95")
    print("• Precision: True positives / (True positives + False positives)")
    print("• Recall: True positives / (True positives + False negatives)")
    print("\n💡 For microplastic segmentation:")
    print("• Good mAP@0.5 > 0.7 for practical applications")
    print("• High precision is important to reduce false alarms")
    print("• High recall ensures we don't miss microplastics")

if __name__ == "__main__":
    main()