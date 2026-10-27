from ultralytics import YOLO, SAM
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
import os
import time
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import argparse
from pathlib import Path
import pandas as pd
import json


image_dir = '../../data/patches/yolo_seg/images/test'


# Configuration
CONFIG = {
    'yolo_conf': 0.15,
    'yolo_model': './zoo/rt-detr.pt',
    'sam_model': './zoo/sam2.1_l.pt',
    'resize_dim': (260, 260),
    'iou_threshold': 0.3,
    'min_contour_area': 5,
    'visualization_alpha': 0.5,
    'random_seed': 42,
    'output_dir': './outputs/yolo_sam',
    'use_amp': True,  # Automatic Mixed Precision for faster GPU inference
    'use_half': True,  # Use FP16 (half precision) for inference
    'gpu_memory_fraction': 0.9,  # Use 90% of available GPU memory
}

def get_device():
    """Automatically detect the best available device."""
    if torch.cuda.is_available():
        return '0'  # Use first GPU
    elif torch.backends.mps.is_available():
        return 'mps'  # Use Metal Performance Shaders on macOS
    else:
        return 'cpu'

def load_models(yolo_path, sam_path, device):
    """Load YOLO and SAM models with error handling and GPU optimization."""
    try:
        print(f"Loading models on device: {device}")
        print(f"GPU Acceleration: {'Enabled' if device != 'cpu' else 'Disabled'}")
        
        yolo_model = YOLO(yolo_path)
        sam_model = SAM(sam_path)
        
        # GPU optimization settings
        if device != 'cpu' and torch.cuda.is_available():
            # Enable CUDA optimizations
            torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'])
            torch.cuda.empty_cache()
            
            # Enable cuDNN auto-tuning for better performance
            torch.backends.cudnn.benchmark = True
            
            # Set matmul precision for faster computation
            torch.set_float32_matmul_precision('medium')
            
            print("✓ CUDA optimizations enabled:")
            print(f"  - CuDNN auto-tuning: ON")
            print(f"  - Float32 matmul precision: MEDIUM")
            print(f"  - GPU memory fraction: {CONFIG['gpu_memory_fraction']*100:.0f}%")
        
        return yolo_model, sam_model
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

def validate_image_path(image_dir, image_name):
    """Validate and return a working image path."""
    # First try the specified image
    image_path = os.path.join(image_dir, image_name)
    if os.path.exists(image_path):
        return image_path
    
    # Try to find any test image in the workspace
    test_dir = Path(image_dir)
    if test_dir.exists():
        image_files = list(test_dir.glob('*.png')) + list(test_dir.glob('*.jpg'))
        if image_files:
            return str(image_files[0])
    
    raise FileNotFoundError(f"No valid image found. Please check the path: {image_path}")


# --- 1. SETUP & PATHS ---
# Load models once (outside the loop for efficiency)
device = get_device()
print(f"\n{'='*70}")
print(f"GPU ACCELERATION SETUP")
print(f"{'='*70}")
print(f"Device: {device}")
if device == '0':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print(f"{'='*70}\n")

yolo_model, sam_model = load_models(CONFIG['yolo_model'], CONFIG['sam_model'], device)

# Get list of all PNG images in the directory
image_files = list(Path(image_dir).glob('*.png'))
if not image_files:
    print(f"No PNG images found in {image_dir}")
    exit(1)

print(f"Found {len(image_files)} images to process")

# Initialize list to store all results for CSV export
all_results = []
inference_times = {
    'yolo': [],
    'sam': [],
    'total': []
}

# Track overall start time
overall_start_time = time.time()

# Process each image
for image_file in image_files:
    image_name = image_file.name
    image_path = str(image_file)
    
    print(f"\n=== Processing: {image_name} ===")
    print(f"Using image: {image_path}")

    # Load and validate image
    try:
        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Could not load image from {image_path}, skipping...")
            continue
        
        print(f"Image loaded successfully: {original_image.shape}")
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        resized_image = cv2.resize(original_image, CONFIG['resize_dim'], interpolation=cv2.INTER_AREA)
        
    except Exception as e:
        print(f"Error loading image {image_name}: {e}")
        continue

    # --- 2. DETECTION & SEGMENTATION ---
    print(f"Running YOLO detection (conf={CONFIG['yolo_conf']})...")
    yolo_start_time = time.time()
    
    # Use GPU acceleration context
    with torch.no_grad():  # Disable gradient computation for faster inference
        if CONFIG['use_amp'] and device != 'cpu':
            # Use Automatic Mixed Precision (AMP) for faster inference
            with torch.cuda.amp.autocast():
                try:
                    yolo_results = yolo_model(resized_image, device=device, conf=CONFIG['yolo_conf'])
                    if yolo_results[0].boxes is not None:
                        bboxes = yolo_results[0].boxes.xyxy.tolist()
                    else:
                        bboxes = []
                except Exception as e:
                    print(f"Error during YOLO detection: {e}")
                    bboxes = []
        else:
            try:
                yolo_results = yolo_model(resized_image, device=device, conf=CONFIG['yolo_conf'])
                if yolo_results[0].boxes is not None:
                    bboxes = yolo_results[0].boxes.xyxy.tolist()
                else:
                    bboxes = []
            except Exception as e:
                print(f"Error during YOLO detection: {e}")
                bboxes = []
    
    yolo_inference_time = time.time() - yolo_start_time
    inference_times['yolo'].append(yolo_inference_time)

    # Scale bounding boxes back to original image dimensions
    # Get original and resized dimensions
    original_h, original_w = original_image.shape[:2]
    resized_h, resized_w = resized_image.shape[:2]

    scale_x = original_w / resized_w
    scale_y = original_h / resized_h

    scaled_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        scaled_x_min = x_min * scale_x
        scaled_y_min = y_min * scale_y
        scaled_x_max = x_max * scale_x
        scaled_y_max = y_max * scale_y
        scaled_bboxes.append([scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max])

    all_polygons = []

    if len(scaled_bboxes) == 0:
        print("YOLO didn't find any plastics!")
        inference_times['sam'].append(0.0)
    else:
        try:
            points = [[(b[0]+b[2])/2, (b[1]+b[3])/2] for b in scaled_bboxes]
            labels = [1] * len(scaled_bboxes)

            print("Running SAM segmentation...")
            sam_start_time = time.time()
            
            # Use GPU acceleration for SAM
            with torch.no_grad():
                if CONFIG['use_amp'] and device != 'cpu':
                    with torch.cuda.amp.autocast():
                        sam_results = sam_model(original_image_rgb, points=points, labels=labels, device=device, retina_masks=True)
                else:
                    sam_results = sam_model(original_image_rgb, points=points, labels=labels, device=device, retina_masks=True)
            
            sam_inference_time = time.time() - sam_start_time
            inference_times['sam'].append(sam_inference_time)

            if sam_results[0].masks is None:
                print("SAM did not generate any masks.")
            else:
                # Extract every contour from SAM masks
                masks = sam_results[0].masks.data
                print(f"SAM generated {len(masks)} masks")
                
                for i in range(len(masks)):
                    try:
                        mask_np = masks[i].cpu().numpy().astype(np.uint8)
                        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for cnt in contours:
                            if cv2.contourArea(cnt) > CONFIG['min_contour_area']:
                                all_polygons.append(cnt.reshape(-1, 2).tolist())
                    except Exception as e:
                        print(f"Error processing mask {i}: {e}")
                        continue
                        
        except Exception as e:
            print(f"Error during SAM segmentation: {e}")
            all_polygons = []

    def merge_overlapping_polygons(polygon_list, iou_threshold=0.3):
        """Merge overlapping polygons using IoU threshold."""
        if not polygon_list:
            return []
        
        shapely_polys = []
        for p in polygon_list:
            if len(p) >= 3:
                try:
                    poly = Polygon(p)
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    if poly.is_valid and poly.area > 0:
                        shapely_polys.append(poly)
                except Exception:
                    continue  # Skip invalid polygons
        
        if not shapely_polys:
            return []
        
        merged = True
        while merged and len(shapely_polys) > 1:
            merged = False
            new_polys = []
            skip_indices = set()
            
            for i in range(len(shapely_polys)):
                if i in skip_indices:
                    continue
                
                current_poly = shapely_polys[i]
                for j in range(i + 1, len(shapely_polys)):
                    if j in skip_indices:
                        continue
                    
                    try:
                        other_poly = shapely_polys[j]
                        intersection = current_poly.intersection(other_poly).area
                        union = current_poly.union(other_poly).area
                        
                        if union > 0:
                            iou = intersection / union
                            if iou > iou_threshold:
                                current_poly = current_poly.union(other_poly)
                                skip_indices.add(j)
                                merged = True
                    except Exception:
                        continue  # Skip problematic polygon operations
                
                new_polys.append(current_poly)
            
            shapely_polys = new_polys
        
        # Convert back to coordinate lists
        final = []
        for poly in shapely_polys:
            try:
                if poly.geom_type == 'Polygon':
                    coords = list(poly.exterior.coords)
                    if len(coords) >= 4:  # Minimum for a valid polygon
                        final.append(coords[:-1])  # Remove duplicate last point
                elif poly.geom_type == 'MultiPolygon':
                    for part in poly.geoms:
                        coords = list(part.exterior.coords)
                        if len(coords) >= 4:
                            final.append(coords[:-1])
            except Exception:
                continue  # Skip problematic polygons
        
        return final

    # --- 3. ROBUST POLYGON MERGING ---

    print(f"Original count: {len(all_polygons)}")
    merged_polygons = merge_overlapping_polygons(all_polygons, iou_threshold=CONFIG['iou_threshold'])
    print(f"Final count after robust merging: {len(merged_polygons)}")
    
    # Calculate metrics for each polygon
    polygon_data = []
    for idx, poly in enumerate(merged_polygons):
        if len(poly) >= 3:
            try:
                # Calculate area using Shapely for accuracy
                shapely_poly = Polygon(poly)
                area = shapely_poly.area
                
                # Calculate bounding box
                pts = np.array(poly, np.int32)
                x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
                
                # Calculate centroid
                centroid = shapely_poly.centroid
                cx, cy = centroid.x, centroid.y
                
                polygon_info = {
                    'image_name': image_name,
                    'polygon_id': idx,
                    'area_pixels': round(area, 2),
                    'bbox_x': x,
                    'bbox_y': y,
                    'bbox_width': w,
                    'bbox_height': h,
                    'centroid_x': round(cx, 2),
                    'centroid_y': round(cy, 2),
                    'num_vertices': len(poly),
                    'polygon_coords': json.dumps(poly)  # Store as JSON string
                }
                polygon_data.append(polygon_info)
                
            except Exception as e:
                print(f"Error calculating metrics for polygon {idx}: {e}")
                continue
    
    # Add summary data for this image
    image_summary = {
        'image_name': image_name,
        'total_detections': len(merged_polygons),
        'total_area_pixels': sum([p['area_pixels'] for p in polygon_data]),
        'yolo_detections': len(scaled_bboxes),
        'sam_masks': len(masks) if 'masks' in locals() and masks is not None else 0,
        'original_polygons': len(all_polygons),
        'merged_polygons': len(merged_polygons)
    }
    
    # Store results
    all_results.extend(polygon_data)
    
    # Also store image summary (we'll create a separate summary CSV)
    if 'image_summaries' not in locals():
        image_summaries = []
    image_summaries.append(image_summary)

    # --- 4. VISUALIZATION ---
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image for visualization: {image_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        overlay = img.copy()
        random.seed(CONFIG['random_seed'])
        
        # Generate consistent colors for all polygons
        colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                for _ in range(len(merged_polygons))]
        
        # Draw merged polygons (filled)
        for i, poly in enumerate(merged_polygons):
            if len(poly) >= 3:
                try:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(overlay, [pts], color=colors[i])
                except Exception as e:
                    print(f"Error drawing polygon {i}: {e}")
                    continue
        
        # Draw merged polygons outlines and their bounding boxes (on top)
        for i, poly in enumerate(merged_polygons):
            if len(poly) >= 3:
                try:
                    pts = np.array(poly, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(overlay, [pts], True, colors[i], thickness=2)
                    
                    # Calculate and draw bounding box for the current merged polygon
                    x, y, w, h = cv2.boundingRect(pts)
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                except Exception as e:
                    print(f"Error drawing outline for polygon {i}: {e}")
                    continue
        
        final_img = cv2.addWeighted(overlay, CONFIG['visualization_alpha'], img, 1-CONFIG['visualization_alpha'], 0)
        
        if not os.path.exists(CONFIG['output_dir']):
            os.makedirs(CONFIG['output_dir'])
        
        plt.figure(figsize=(12, 12))
        plt.imshow(final_img)
        plt.title(f"Final Processed Instances: {len(merged_polygons)} (Image: {image_name})")
        plt.axis('off')
        
        # Save the figure
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(CONFIG['output_dir'], f"output_{base_name}_conf{CONFIG['yolo_conf']}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()  # Close the figure to free memory
            
    except Exception as e:
        print(f"Error during visualization for {image_name}: {e}")
        
    print(f"Processing completed for {image_name}!")
    print(f"Final results: {len(merged_polygons)} plastic particles detected and segmented.")

print("\n=== All images processed! ===")

# Calculate and report inference times
overall_time = time.time() - overall_start_time
yolo_times = inference_times['yolo']
sam_times = inference_times['sam']
total_yolo_time = sum(yolo_times)
total_sam_time = sum(sam_times)
total_inference_time = total_yolo_time + total_sam_time

print(f"\n{'='*70}")
print(f"INFERENCE TIME SUMMARY")
print(f"{'='*70}")
print(f"Total Processing Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
print(f"Total Inference Time: {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
print(f"\nBREAKDOWN:")
print(f"  YOLO Detection Time: {total_yolo_time:.2f}s ({total_yolo_time/total_inference_time*100:.1f}%)")
print(f"  SAM Segmentation Time: {total_sam_time:.2f}s ({total_sam_time/total_inference_time*100:.1f}%)")
print(f"\nPER-IMAGE AVERAGES:")
print(f"  Avg YOLO Time: {np.mean(yolo_times):.3f}s")
print(f"  Avg SAM Time: {np.mean(sam_times):.3f}s")
print(f"  Avg Total Time: {np.mean([y+s for y,s in zip(yolo_times, sam_times)]):.3f}s")
if len(yolo_times) > 1:
    print(f"\nYOLO STATISTICS:")
    print(f"  Min: {np.min(yolo_times):.3f}s, Max: {np.max(yolo_times):.3f}s, Std: {np.std(yolo_times):.3f}s")
if len(sam_times) > 1:
    print(f"SAM STATISTICS:")
    print(f"  Min: {np.min(sam_times):.3f}s, Max: {np.max(sam_times):.3f}s, Std: {np.std(sam_times):.3f}s")
print(f"{'='*70}")

# Export results to CSV
if all_results:
    # Create detailed predictions CSV
    df_detailed = pd.DataFrame(all_results)
    detailed_csv_path = os.path.join(CONFIG['output_dir'], 'detailed_predictions.csv')
    df_detailed.to_csv(detailed_csv_path, index=False)
    print(f"Detailed predictions saved to: {detailed_csv_path}")
    
    # Create image summary CSV
    if 'image_summaries' in locals():
        df_summary = pd.DataFrame(image_summaries)
        summary_csv_path = os.path.join(CONFIG['output_dir'], f'image_summaries_{CONFIG["yolo_conf"]}.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"Image summaries saved to: {summary_csv_path}")
    
    print(f"\nTotal results: {len(all_results)} polygon detections across {len(image_summaries)} images")
else:
    print("No detections found across all images.")

print(f"{'='*70}\n")

# Clean up GPU memory if using CUDA
if device != 'cpu' and torch.cuda.is_available():
    torch.cuda.empty_cache()
    print("✓ GPU memory cleaned up")

torch.cuda.empty_cache() if torch.cuda.is_available() else None