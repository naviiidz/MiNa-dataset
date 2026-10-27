#!/usr/bin/env python3
"""
YOLOv8 Segmentation Inference Script for Microplastics Test Set
Uses trained model to perform inference on test images
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import json

def load_trained_model(model_path=None):
    """Load the best trained YOLOv8 model"""
    if model_path is None:
        # Use the specific MNP trained model
        model_path = "zoo/MNP_yolov8_segment.pt"
    
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    return model

def run_inference_on_test_set(model, test_dir, output_dir="inference_results", conf_threshold=0.25):
    """Run inference on all test images"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
    os.makedirs(f"{output_dir}/predictions", exist_ok=True)
    
    # Get all test images
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        test_images.extend(glob.glob(os.path.join(test_dir, ext)))
    
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print(f"No images found in {test_dir}")
        return
    
    results_summary = []
    
    # Process each test image
    for i, img_path in enumerate(test_images):
        print(f"Processing {i+1}/{len(test_images)}: {os.path.basename(img_path)}")
        
        # Run inference
        results = model(img_path, conf=conf_threshold, save=False, verbose=False)
        
        # Process results
        for r in results:
            img_name = os.path.basename(img_path)
            img_name_no_ext = os.path.splitext(img_name)[0]
            
            # Get detection information
            num_detections = len(r.boxes) if r.boxes is not None else 0
            
            detection_info = {
                'image_name': img_name,
                'num_detections': num_detections,
                'detections': []
            }
            
            # Load original image for visualization
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if num_detections > 0:
                # Extract detection details
                boxes = r.boxes.xyxy.cpu().numpy()
                confidences = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.cpu().numpy()
                
                # Extract masks if available
                if r.masks is not None:
                    masks = r.masks.data.cpu().numpy()
                    
                    # Create visualization
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    
                    # Image with detections only
                    ax.imshow(img_rgb)
                    
                    # Overlay masks and boxes
                    colors = plt.cm.tab10(np.linspace(0, 1, 10))
                    
                    for j, (box, conf, mask) in enumerate(zip(boxes, confidences, masks)):
                        color = colors[j % len(colors)]
                        
                        # Draw bounding box (hidden)
                        # x1, y1, x2, y2 = box
                        # rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        #                    linewidth=2, edgecolor=color[:3], 
                        #                    facecolor='none')
                        # ax.add_patch(rect)
                        
                        # Fill mask
                        x1, y1, x2, y2 = box
                        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))
                        # Create transparent overlay only on particles
                        img_with_overlay = img_rgb.copy().astype(float)
                        mask_area = mask_resized > 0.5
                        for c in range(3):  # RGB channels
                            img_with_overlay[mask_area, c] = img_with_overlay[mask_area, c] * 0.5 + color[c] * 255 * 0.5
                        img_rgb = img_with_overlay.astype(np.uint8)
                        
                        # Store detection info
                        detection_info['detections'].append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': 'MNP',
                            'class_id': int(classes[j])
                        })
                    
                    # Display the image with overlays
                    ax.imshow(img_rgb)
                    #ax.set_title(f'Detections: {num_detections} microplastics found')
                    ax.axis('off')
                    
                else:
                    # No masks, just boxes (but boxes are hidden)
                    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                    
                    ax.imshow(img_rgb)
                    
                    for j, (box, conf) in enumerate(zip(boxes, confidences)):
                        # x1, y1, x2, y2 = box
                        # rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                        #                    linewidth=2, edgecolor='red', 
                        #                    facecolor='none')
                        # ax.add_patch(rect)
                        
                        # ax.text(x1, y1-5, f'MNP: {conf:.2f}', 
                        #         bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                        #         fontsize=10, color='white')
                        
                        x1, y1, x2, y2 = box
                        detection_info['detections'].append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(conf),
                            'class': 'MNP',
                            'class_id': int(classes[j])
                        })
                    
                    ax.set_title(f'Detections: {num_detections} microplastics found')
                    ax.axis('off')
            
            else:
                # No detections
                fig, ax = plt.subplots(1, 1, figsize=(10, 7))
                ax.imshow(img_rgb)
                ax.set_title(f'No detections: {img_name}')
                ax.axis('off')
            
            # Save visualization
            plt.tight_layout()
            plt.savefig(f"{output_dir}/visualizations/{img_name_no_ext}_result.png", 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save individual prediction file
            with open(f"{output_dir}/predictions/{img_name_no_ext}.json", 'w') as f:
                json.dump(detection_info, f, indent=2)
            
            results_summary.append(detection_info)
    
    # Save summary results
    with open(f"{output_dir}/inference_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Print summary statistics
    print("\n" + "="*50)
    print("INFERENCE SUMMARY")
    print("="*50)
    total_images = len(results_summary)
    images_with_detections = sum(1 for r in results_summary if r['num_detections'] > 0)
    total_detections = sum(r['num_detections'] for r in results_summary)
    
    print(f"Total images processed: {total_images}")
    print(f"Images with detections: {images_with_detections} ({images_with_detections/total_images*100:.1f}%)")
    print(f"Images without detections: {total_images - images_with_detections}")
    print(f"Total microplastic detections: {total_detections}")
    print(f"Average detections per image: {total_detections/total_images:.2f}")
    print(f"Average detections per positive image: {total_detections/max(1, images_with_detections):.2f}")
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"- Visualizations: {output_dir}/visualizations/")
    print(f"- Predictions: {output_dir}/predictions/")
    print(f"- Summary: {output_dir}/inference_summary.json")

def main():
    """Main function to run inference on test set"""
    
    print("YOLOv8 Segmentation Inference on Test Set")
    print("="*50)
    
    # Configuration
    test_dir = "../../data/patches/yolo_seg/images/test"
    conf_threshold = 0.5  # Confidence threshold for detections
    
    # Check if test directory exists
    if not os.path.exists(test_dir):
        print(f"Error: Test directory not found: {test_dir}")
        return
    
    # Load trained model
    try:
        model = load_trained_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Run inference
    print(f"\nRunning inference on test set: {test_dir}")
    print(f"Confidence threshold: {conf_threshold}")
    
    try:
        run_inference_on_test_set(model, test_dir, conf_threshold=conf_threshold)
        print("\nInference completed successfully! ✅")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    main()