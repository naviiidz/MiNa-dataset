#!/usr/bin/env python3
"""
Quantification Analysis Script
Compares ground truth vs predicted counts for each microplastic class.
"""

import os
import glob
import warnings
warnings.filterwarnings('ignore')

def find_best_weights(model_name):
    """Latest best.pt written by quick_train.py --model <model_name> (runs/<model>/mnp_<model>*)."""
    from pathlib import Path
    runs = sorted(Path("runs", model_name).glob(f"mnp_{model_name}*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    return str(runs[-1]) if runs else None


def load_model(path):
    """RT-DETR checkpoints need the RTDETR class; everything else loads with YOLO."""
    from ultralytics import YOLO, RTDETR
    return (RTDETR if "rtdetr" in path else YOLO)(path)


def analyze_quantification(argparse_model='yolo26s'):
    """Analyze quantification accuracy by comparing GT vs predictions."""
    from ultralytics import YOLO
    
    print("📊 MICROPLASTIC QUANTIFICATION ANALYSIS")
    print("=" * 60)
    
    # Find the latest trained model
    model_path = find_best_weights(argparse_model)
    if not model_path:
        print(f"❌ No trained model found in runs/{argparse_model}")
        return
    
    print(f"🤖 Model: {model_path}")
    model = load_model(model_path)
    
    # Class names
    class_names = ['PP', 'PE', 'PS', 'PET']
    
    # Initialize counters
    gt_counts = {cls: 0 for cls in class_names}
    pred_counts = {cls: 0 for cls in class_names}
    
    # Directories
    test_images_dir = "../../data/patches/yolo_det/images/test"
    test_labels_dir = "../../data/patches/yolo_det/labels/test"
    
    # Get test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    print(f"📁 Processing {len(test_images)} test images...")
    
    # Process each image
    for img_path in test_images:
        # Count ground truth
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(test_labels_dir, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        cls_id = int(parts[0])
                        if cls_id < len(class_names):
                            gt_counts[class_names[cls_id]] += 1
        
        # Count predictions
        results = model.predict(img_path, conf=0.25, device='0', verbose=False)
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id < len(class_names):
                    pred_counts[class_names[cls_id]] += 1
    
    # Print results
    print(f"\n📊 QUANTIFICATION RESULTS")
    print("=" * 60)
    
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    
    print(f"{'Class':<6} {'Ground Truth':<12} {'Predicted':<10} {'Difference':<10} {'Accuracy':<8}")
    print("-" * 60)
    
    for class_name in class_names:
        gt_count = gt_counts[class_name]
        pred_count = pred_counts[class_name]
        difference = pred_count - gt_count
        
        # Calculate accuracy
        if gt_count > 0:
            accuracy = (1 - abs(difference) / gt_count) * 100
        else:
            accuracy = 100 if pred_count == 0 else 0
            
        diff_str = f"{difference:+d}"
        print(f"{class_name:<6} {gt_count:<12} {pred_count:<10} {diff_str:<10} {accuracy:.1f}%")
    
    print("-" * 60)
    total_accuracy = ((1 - abs(total_pred - total_gt) / total_gt) * 100 if total_gt > 0 else 0)
    total_diff_str = f"{total_pred - total_gt:+d}"
    print(f"{'TOTAL':<6} {total_gt:<12} {total_pred:<10} {total_diff_str:<10} {total_accuracy:.1f}%")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Total objects in ground truth: {total_gt}")
    print(f"   Total objects predicted: {total_pred}")
    print(f"   Overall count difference: {total_pred - total_gt:+d}")
    print(f"   Overall quantification accuracy: {total_accuracy:.1f}%")
    
    print(f"\n💡 CLASS-SPECIFIC ANALYSIS:")
    for class_name in class_names:
        gt_count = gt_counts[class_name]
        pred_count = pred_counts[class_name]
        difference = pred_count - gt_count
        
        if gt_count > 0:
            percentage = abs(difference) / gt_count * 100
            if difference > 0:
                print(f"   {class_name}: Over-predicted by {difference} objects ({percentage:.1f}% above GT)")
            elif difference < 0:
                print(f"   {class_name}: Under-predicted by {abs(difference)} objects ({percentage:.1f}% below GT)")
            else:
                print(f"   {class_name}: Perfect count match! 🎯")
        else:
            print(f"   {class_name}: No ground truth objects, predicted {pred_count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s", help="model name used with quick_train.py --model")
    analyze_quantification(argparse_model=parser.parse_args().model)
