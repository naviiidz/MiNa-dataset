#!/usr/bin/env python3
"""
Patch Visualization Script for Microplastic Datasets
"""

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
import argparse

# Common data folder at the repository root (next to data_processing/)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'patches', 'coco')

# Use non-interactive backend to avoid display issues
plt.switch_backend('Agg')

def visualize_patches(data_dir, output_dir='patch_visualizations', max_patches=None, skip_empty=False):
    """
    Visualize patch annotations and save to disk.
    
    Args:
        data_dir: Directory containing patches and annotations.json
        output_dir: Directory to save visualizations
        max_patches: Maximum number of patches to process (None for all)
        skip_empty: Skip patches with no annotations
    """
    # Load annotations
    ann_path = os.path.join(data_dir, 'annotations.json')
    if not os.path.exists(ann_path):
        print(f"Error: Annotations file not found: {ann_path}")
        return
    
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")
    
    # Group annotations by image
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Process images
    images_to_process = coco_data['images']
    if max_patches:
        images_to_process = images_to_process[:max_patches]
    
    print(f"Processing {len(images_to_process)} patches...")
    
    processed_count = 0
    with_annotations_count = 0
    skipped_count = 0
    
    # Category mapping for colors
    category_colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'yellow'}  # PP, PE, PS, PET
    category_names = {1: 'PP', 2: 'PE', 3: 'PS', 4: 'PET'}
    
    for img_info in images_to_process:
        img_id = img_info['id']
        target_annotations = annotations_by_image.get(img_id, [])
        
        # Skip empty patches if requested
        if skip_empty and len(target_annotations) == 0:
            skipped_count += 1
            continue
        
        # Load image
        img_path = os.path.join(data_dir, 'images', img_info['file_name'])
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        image = cv2.imread(img_path)
        if image is None:
            print(f"Warning: Could not load: {img_path}")
            continue
            
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title('Original Patch')
        ax1.axis('off')
        
        # Image with annotations
        ax2.imshow(image_rgb)
        polymer_type = img_info.get('polymer_type', 'Unknown')
        ax2.set_title(f'{polymer_type} - {len(target_annotations)} annotations')
        
        if target_annotations:
            with_annotations_count += 1
            
            # Draw annotations
            for idx, ann in enumerate(target_annotations):
                category_id = ann.get('category_id', 1)
                color = category_colors.get(category_id, 'cyan')
                
                # Draw bounding box
                if 'bbox' in ann and len(ann['bbox']) == 4:
                    x, y, w, h = ann['bbox']
                    if w > 0 and h > 0:
                        rect = mpatches.Rectangle((x, y), w, h, 
                                                linewidth=1, 
                                                edgecolor=color, 
                                                facecolor='none',
                                                alpha=0.8)
                        ax2.add_patch(rect)
                
                # Draw segmentation polygon
                if 'segmentation' in ann and ann['segmentation']:
                    for seg in ann['segmentation']:
                        if len(seg) >= 6 and len(seg) % 2 == 0:
                            # Convert to coordinate pairs
                            coords = np.array(seg).reshape(-1, 2)
                            
                            # Validate coordinates are within bounds
                            valid_coords = (coords[:, 0] >= 0) & (coords[:, 0] <= 256) & \
                                         (coords[:, 1] >= 0) & (coords[:, 1] <= 256)
                            
                            if np.all(valid_coords):
                                # Create polygon
                                polygon = Polygon(coords, 
                                                alpha=0.3, 
                                                facecolor=color,
                                                edgecolor=color, 
                                                linewidth=1)
                                ax2.add_patch(polygon)
        
        # Set limits and remove ticks
        ax2.set_xlim(0, 256)
        ax2.set_ylim(256, 0)  # Flip y-axis
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Create filename
        base_name = os.path.splitext(img_info['file_name'])[0]
        output_filename = f"{base_name}_annotated.png"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save figure
        try:
            plt.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close()  # Important: close to free memory
            
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"  Processed {processed_count}/{len(images_to_process)} patches...")
                
        except Exception as e:
            print(f"Warning: Failed to save {output_path}: {e}")
            plt.close()
    
    print(f"\nVisualization complete!")
    print(f"Processed: {processed_count} patches")
    print(f"With annotations: {with_annotations_count}")
    print(f"Empty patches: {processed_count - with_annotations_count}")
    if skip_empty:
        print(f"Skipped empty: {skipped_count}")
    print(f"Saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Visualize patch annotations')
    parser.add_argument('--data_dir', 
                       default=os.path.join(DATA_ROOT, 'patches_256_combined', 'train'),
                       help='Directory containing patches and annotations')
    parser.add_argument('--output_dir', default='patch_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--max_patches', type=int, default=None,
                       help='Maximum number of patches to process')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train',
                       help='Dataset split to visualize')
    parser.add_argument('--skip_empty', action='store_true',
                       help='Skip patches with no annotations')
    
    args = parser.parse_args()
    
    # Construct data directory path
    if args.split not in args.data_dir:
        base_dir = os.path.join(DATA_ROOT, 'patches_256_combined')
        data_dir = os.path.join(base_dir, args.split)
    else:
        data_dir = args.data_dir
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    visualize_patches(data_dir, args.output_dir, args.max_patches, args.skip_empty)

if __name__ == "__main__":
    main()