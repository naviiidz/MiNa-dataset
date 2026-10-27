#!/usr/bin/env python3
"""
Comprehensive Test Visualization Script
Shows bounding boxes with labels on all test images using the trained model.
"""

import os
import cv2
import glob
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

def visualize_test_predictions():
    """Visualize predictions on test set with bounding boxes and labels."""
    from ultralytics import YOLO
    
    print("🔍 Test Set Visualization with Bounding Boxes")
    print("=" * 50)
    
    # Find the latest trained model (YOLOv8s - the model we actually trained)
    yolo8s_runs = "runs/yolo8s"
    model_patterns = [
        f"{yolo8s_runs}/mnp_yolo8s3/weights/best.pt",
        f"{yolo8s_runs}/mnp_yolo8s2/weights/best.pt",  
        f"{yolo8s_runs}/mnp_yolo8s/weights/best.pt"
    ]
    
    model_path = None
    for pattern in model_patterns:
        if os.path.exists(pattern):
            model_path = pattern
            break
    
    if not model_path:
        print(f"❌ No trained model found in {yolo8s_runs}")
        return
    
    print(f"📁 Loading model: {model_path}")
    model = YOLO(model_path)
    
    # Class names and colors
    class_names = ['PP', 'PE', 'PS', 'PET']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # BGR format for OpenCV
    
    # Initialize counters for quantification analysis
    gt_counts = {cls: 0 for cls in class_names}
    pred_counts = {cls: 0 for cls in class_names}
    
    # Find test images and labels
    test_images_dir = "../../data/patches/yolo_det/images/test"
    test_labels_dir = "../../data/patches/yolo_det/labels/test"
    
    if not os.path.exists(test_images_dir):
        print(f"❌ Test images directory not found: {test_images_dir}")
        return
    
    if not os.path.exists(test_labels_dir):
        print(f"❌ Test labels directory not found: {test_labels_dir}")
        return
    
    # Get all test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    test_images = []
    for ext in image_extensions:
        test_images.extend(glob.glob(os.path.join(test_images_dir, ext)))
    
    if not test_images:
        print(f"❌ No test images found in {test_images_dir}")
        return
    
    print(f"🖼️  Found {len(test_images)} test images")
    
    # Create output directory
    output_dir = "test_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    print("\n🚀 Processing images...")
    
    for idx, img_path in enumerate(test_images):
        try:
            # Get corresponding label file
            img_name = os.path.basename(img_path)
            base_name = os.path.splitext(img_name)[0]
            label_path = os.path.join(test_labels_dir, f"{base_name}.txt")
            
            # Count ground truth labels
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            cls_id = int(parts[0])
                            if cls_id < len(class_names):
                                gt_counts[class_names[cls_id]] += 1
            
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"⚠️  Could not read image: {img_path}")
                continue
            
            # Make prediction
            results = model.predict(img_path, conf=0.25, device='0', verbose=False)
            
            # Draw predictions on image
            annotated_image = image.copy()
            
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Draw each detection
                for i, box in enumerate(boxes):
                    # Get coordinates (xyxy format)
                    coords = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, coords)
                    
                    # Get confidence and class
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    
                    if cls_id < len(class_names):
                        class_name = class_names[cls_id]
                        color = colors[cls_id]
                        
                        # Count predictions
                        pred_counts[class_name] += 1
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                        
                        # Create label with class only (no confidence)
                        label = f"{class_name}"
                        
                        # Get text size for background
                        (text_width, text_height), baseline = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                        
                        # Draw label background
                        cv2.rectangle(annotated_image, 
                                    (x1, y1 - text_height - baseline - 5), 
                                    (x1 + text_width, y1), 
                                    color, -1)
                        
                        # Draw label text
                        cv2.putText(annotated_image, label, 
                                  (x1, y1 - baseline - 2), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                  (255, 255, 255), 1)
            
            # Save annotated image
            output_path = os.path.join(output_dir, f"pred_{os.path.basename(img_path)}")
            cv2.imwrite(output_path, annotated_image)
            
            # Progress update
            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_images):
                print(f"   Processed {idx + 1}/{len(test_images)} images")
                
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {str(e)}")
            continue
    
    print(f"\n✅ Visualization completed!")
    print(f"📁 All annotated images saved in: {output_dir}/")
    print(f"🎯 Images show bounding boxes with class labels and confidence scores")
    
    # Print quantification analysis
    print_quantification_analysis(gt_counts, pred_counts)
    
    # Create a summary montage of first 9 results
    create_summary_montage(output_dir)

def print_quantification_analysis(gt_counts, pred_counts):
    """Print detailed quantification analysis comparing ground truth vs predictions."""
    print(f"\n📊 QUANTIFICATION ANALYSIS")
    print("=" * 60)
    
    total_gt = sum(gt_counts.values())
    total_pred = sum(pred_counts.values())
    
    print(f"{'Class':<6} {'Ground Truth':<12} {'Predicted':<10} {'Difference':<10} {'Accuracy':<8}")
    print("-" * 60)
    
    for class_name in ['PP', 'PE', 'PS', 'PET']:
        gt_count = gt_counts[class_name]
        pred_count = pred_counts[class_name]
        difference = pred_count - gt_count
        
        # Calculate quantification accuracy (how close predicted count is to GT)
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
    
    # Class-specific insights
    print(f"\n💡 INSIGHTS:")
    for class_name in ['PP', 'PE', 'PS', 'PET']:
        gt_count = gt_counts[class_name]
        pred_count = pred_counts[class_name]
        difference = pred_count - gt_count
        
        if difference > 0:
            print(f"   {class_name}: Over-predicted by {difference} objects ({difference/gt_count*100:.1f}% more than GT)")
        elif difference < 0:
            print(f"   {class_name}: Under-predicted by {abs(difference)} objects ({abs(difference)/gt_count*100:.1f}% less than GT)")
        else:
            print(f"   {class_name}: Perfect count match! 🎯")
    
def create_summary_montage(output_dir):
    """Create a summary montage showing first 9 predictions."""
    try:
        # Get first 9 annotated images
        annotated_images = sorted(glob.glob(os.path.join(output_dir, "pred_*.jpg")))[:9]
        
        if len(annotated_images) < 1:
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        fig.suptitle('Test Set Predictions - Sample Results', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for i, img_path in enumerate(annotated_images):
            if i >= 9:
                break
                
            # Read and display image
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i].imshow(img_rgb)
            axes[i].set_title(os.path.basename(img_path).replace('pred_', ''), fontsize=10)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(annotated_images), 9):
            axes[i].axis('off')
        
        plt.tight_layout()
        summary_path = os.path.join(output_dir, "summary_montage.jpg")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Summary montage saved: {summary_path}")
        
    except Exception as e:
        print(f"⚠️  Could not create summary montage: {str(e)}")

if __name__ == "__main__":
    visualize_test_predictions()