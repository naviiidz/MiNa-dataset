"""
Pure Watershed-based AIS (Automatic Instance Segmentation) for microplastics detection.
Direct image processing without SAM - pure morphological operations + watershed.
Fast baseline for instance segmentation.
"""

import os
import numpy as np
import cv2
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import random

import imageio.v3 as imageio
from shapely.geometry import Polygon


def extract_polygons_from_mask(segmentation_mask, min_contour_area=5):
    """Extract polygon coordinates from a segmentation mask."""
    polygons = []
    
    # Get unique labels (excluding background 0)
    unique_labels = np.unique(segmentation_mask)
    unique_labels = unique_labels[unique_labels != 0]
    
    for label in unique_labels:
        # Create binary mask for this instance
        binary_mask = (segmentation_mask == label).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by minimum area
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue
            
            # Simplify contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            simplified = cv2.approxPolyDP(contour, epsilon, True)
            
            # Convert to list of [x, y] coordinates
            poly_coords = simplified.reshape(-1, 2).tolist()
            
            if len(poly_coords) >= 3:  # Valid polygon needs at least 3 points
                polygons.append({
                    'coords': poly_coords,
                    'area': area,
                    'contour': contour
                })
    
    return polygons


def pure_watershed_segmentation(image, 
                                 threshold_method='otsu',
                                 threshold_value=0,
                                 morph_kernel_size=5,
                                 distance_transform_type=cv2.DIST_L2,
                                 min_size=15):
    """
    Pure watershed-based instance segmentation.
    
    Parameters:
    -----------
    image : ndarray
        Input image (grayscale or color)
    threshold_method : str
        'otsu', 'adaptive', or 'fixed' (use threshold_value)
    threshold_value : int
        Fixed threshold value (used when threshold_method='fixed')
    morph_kernel_size : int
        Kernel size for morphological operations
    distance_transform_type : int
        cv2.DIST_L2 or cv2.DIST_L1
    min_size : int
        Minimum particle size in pixels (filtering)
    
    Returns:
    --------
    segmentation : ndarray
        Instance segmentation map with unique labels per particle
    """
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] == 3 else image[:, :, 0]
    else:
        gray = image.copy()
    
    # Normalize to 0-255 range if needed
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    
    # Step 1: Thresholding
    if threshold_method == 'otsu':
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif threshold_method == 'adaptive':
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
    else:  # fixed
        _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    # Step 2: Morphological operations (noise removal)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size))
    
    # Opening: removes small noise
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Closing: fills small holes
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Step 3: Distance transform
    dist_transform = cv2.distanceTransform(binary, distance_transform_type, cv2.DIST_MASK_PRECISE)
    
    # Normalize distance transform to 0-255
    dist_normalized = cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Step 4: Find sure foreground (local maxima)
    # Use threshold on distance transform to find strong peaks
    _, sure_fg = cv2.threshold(dist_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Label connected components in sure foreground
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Step 5: Watershed segmentation
    # Apply watershed algorithm
    markers = cv2.watershed(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), markers)
    
    # Step 6: Filter by size and remove small particles
    segmentation = np.zeros_like(markers)
    unique_labels = np.unique(markers)
    
    new_label = 1
    for label in unique_labels:
        if label <= 1:  # Skip background and borders
            continue
        
        mask = (markers == label).astype(np.uint8)
        size = np.sum(mask)
        
        # Only keep particles larger than min_size
        if size >= min_size:
            segmentation[mask == 1] = new_label
            new_label += 1
    
    return segmentation


def process_test_set_watershed(threshold_method='otsu',
                                threshold_value=0,
                                morph_kernel_size=5,
                                min_size=15):
    """Process all test set images using pure watershed AIS."""
    test_dir = Path("../../data/patches/yolo_seg/images/test")
    
    if not test_dir.exists():
        print(f"Error: Test directory {test_dir} does not exist.")
        return
    
    # Create output directory
    output_dir = "watershed_ais_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = sorted(test_dir.glob("*.png")) + sorted(test_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"No images found in {test_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"PURE WATERSHED AIS SEGMENTATION")
    print(f"{'='*70}")
    print(f"Threshold Method: {threshold_method}")
    print(f"Morphological Kernel Size: {morph_kernel_size}")
    print(f"Minimum Particle Size: {min_size} pixels")
    print(f"{'='*70}\n")
    
    print(f"Processing {len(image_files)} images from test set...\n")
    
    # Lists to store results for CSV export
    all_polygon_results = []
    image_summaries = []
    inference_times = []
    
    overall_start_time = time.time()
    
    for idx, image_path in enumerate(image_files, 1):
        try:
            print(f"[{idx}/{len(image_files)}] Processing {image_path.name}...")
            image_name = image_path.stem
            
            # Read image
            image = imageio.imread(image_path)
            
            # Segment using watershed
            segmentation_start_time = time.time()
            segmentation = pure_watershed_segmentation(
                image,
                threshold_method=threshold_method,
                threshold_value=threshold_value,
                morph_kernel_size=morph_kernel_size,
                min_size=min_size
            )
            segmentation_time = time.time() - segmentation_start_time
            inference_times.append(segmentation_time)
            
            # Extract polygons from segmentation mask
            polygons = extract_polygons_from_mask(segmentation, min_contour_area=5)
            
            # Prepare polygon data for CSV
            for poly_idx, poly_info in enumerate(polygons):
                poly_coords = poly_info['coords']
                area = poly_info['area']
                
                # Calculate bounding box
                pts = np.array(poly_coords, np.int32)
                x, y, w, h = cv2.boundingRect(pts.reshape((-1, 1, 2)))
                
                # Calculate centroid using shapely
                try:
                    shapely_poly = Polygon(poly_coords)
                    centroid = shapely_poly.centroid
                    cx, cy = centroid.x, centroid.y
                except:
                    cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
                
                polygon_result = {
                    'image_name': image_path.name,
                    'polygon_id': poly_idx,
                    'area_pixels': round(area, 2),
                    'bbox_x': int(x),
                    'bbox_y': int(y),
                    'bbox_width': int(w),
                    'bbox_height': int(h),
                    'centroid_x': round(cx, 2),
                    'centroid_y': round(cy, 2),
                    'num_vertices': len(poly_coords),
                    'polygon_coords': json.dumps(poly_coords)
                }
                all_polygon_results.append(polygon_result)
            
            # Create image summary
            image_summary = {
                'image_name': image_path.name,
                'total_detections': len(polygons),
                'total_area_pixels': sum([p['area'] for p in polygons]),
                'avg_particle_area': round(np.mean([p['area'] for p in polygons]), 2) if polygons else 0,
            }
            image_summaries.append(image_summary)
            
            # Create visualization
            if image.dtype != np.uint8:
                if image.max() <= 1:
                    image_uint8 = (image * 255).astype(np.uint8)
                else:
                    image_uint8 = image.astype(np.uint8)
            else:
                image_uint8 = image
            
            # Convert to RGB if grayscale
            if len(image_uint8.shape) == 2:
                image_rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image_uint8
            
            # Create overlay visualization
            overlay = image_rgb.copy()
            random.seed(42)
            colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) 
                     for _ in range(len(polygons))]
            
            # Draw filled polygons
            for i, (poly_info, color) in enumerate(zip(polygons, colors)):
                pts = np.array(poly_info['coords'], np.int32)
                cv2.fillPoly(overlay, [pts], color)
            
            # Draw polygon outlines
            for i, (poly_info, color) in enumerate(zip(polygons, colors)):
                pts = np.array(poly_info['coords'], np.int32)
                cv2.polylines(overlay, [pts], True, (255, 255, 255), 2)
            
            # Create final composite image
            final_img = cv2.addWeighted(overlay, 0.5, image_rgb, 0.5, 0)
            
            # Save visualization
            plt.figure(figsize=(12, 12))
            plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
            plt.title(f"Pure Watershed AIS: {len(polygons)} particles detected (Image: {image_path.name})")
            plt.axis('off')
            
            output_path = os.path.join(output_dir, f"output_{image_name}_watershed.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"  ✓ Visualization saved: {output_path}")
            plt.close()
            
        except Exception as e:
            print(f"  ✗ Error processing {image_path.name}: {str(e)}\n")
            continue
    
    # Export results to CSV
    if all_polygon_results:
        df_detailed = pd.DataFrame(all_polygon_results)
        detailed_csv_path = os.path.join(output_dir, 'detailed_predictions_watershed.csv')
        df_detailed.to_csv(detailed_csv_path, index=False)
        print(f"\n✓ Detailed predictions saved: {detailed_csv_path}")
    
    if image_summaries:
        df_summary = pd.DataFrame(image_summaries)
        summary_csv_path = os.path.join(output_dir, 'image_summaries_watershed.csv')
        df_summary.to_csv(summary_csv_path, index=False)
        print(f"✓ Image summaries saved: {summary_csv_path}")
    
    # Calculate and report inference time
    overall_time = time.time() - overall_start_time
    avg_inference_time = np.mean(inference_times) if inference_times else 0.0
    total_inference_time = sum(inference_times)
    
    print(f"\n{'='*70}")
    print(f"INFERENCE TIME SUMMARY")
    print(f"{'='*70}")
    print(f"Total Processing Time: {overall_time:.2f} seconds ({overall_time/60:.2f} minutes)")
    print(f"Total Segmentation Time: {total_inference_time:.2f} seconds ({total_inference_time/60:.2f} minutes)")
    print(f"Average Time per Image: {avg_inference_time:.4f} seconds")
    print(f"Images Processed: {len(inference_times)}")
    if len(inference_times) > 1:
        print(f"Segmentation Time Statistics:")
        print(f"  - Min: {np.min(inference_times):.4f}s")
        print(f"  - Max: {np.max(inference_times):.4f}s")
        print(f"  - Std: {np.std(inference_times):.4f}s")
    print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"Total Detections: {len(all_polygon_results)} across {len(image_summaries)} images")
    print(f"Average Detections per Image: {len(all_polygon_results)/len(image_summaries):.1f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Pure watershed AIS parameters
    # You can tune these for better results:
    # - threshold_method: 'otsu' (automatic), 'adaptive' (local), 'fixed' (manual)
    # - morph_kernel_size: larger = more smoothing, typical range 3-11
    # - min_size: minimum particle size in pixels, filters noise
    
    process_test_set_watershed(
        threshold_method='otsu',      # Automatic Otsu thresholding
        morph_kernel_size=5,          # Morphological kernel size
        min_size=15                   # Minimum particle size (15 pixels)
    )
