#!/usr/bin/env python3
"""
Analyze object sizes in the dataset to understand APS/APM/APL metrics
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict

def parse_yolo_label(label_file, img_width, img_height):
    """Parse YOLO segmentation label file and return object areas"""
    objects = []
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:  # Need at least class + 3 points (6 coordinates)
            continue
            
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i+1] * img_height
            points.append((x, y))
        
        # Calculate area using shoelace formula
        area = 0
        n = len(points)
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        area = abs(area) / 2
        
        # Calculate bounding box area for comparison
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox_area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        
        objects.append({
            'class_id': class_id,
            'mask_area': area,
            'bbox_area': bbox_area,
            'num_points': len(points)
        })
    
    return objects

def categorize_by_coco_size(area):
    """Categorize object by modified size standards (COCO/10)"""
    # Modified thresholds: divide COCO standards by 10
    # Small: < 102.4 px², Medium: 102.4-921.6 px², Large: > 921.6 px²
    if area < 102.4:  # ~10² pixels
        return 'small'
    elif area < 921.6:  # ~30² pixels
        return 'medium'
    else:
        return 'large'

def analyze_dataset_sizes():
    """Analyze object sizes across the entire dataset"""
    
    # Check all splits
    splits = ['train', 'val', 'test']
    all_stats = defaultdict(list)
    size_distribution = defaultdict(int)
    
    for split in splits:
        image_dir = f"../../data/patches/yolo_seg/images/{split}"
        label_dir = f"../../data/patches/yolo_seg/labels/{split}"
        
        if not os.path.exists(image_dir) or not os.path.exists(label_dir):
            print(f"Skipping {split} - directory not found")
            continue
            
        print(f"\nAnalyzing {split} split...")
        
        label_files = list(Path(label_dir).glob("*.txt"))
        split_stats = []
        
        for label_file in label_files:
            # Get corresponding image dimensions (assuming 256x256 patches)
            img_width, img_height = 256, 256
            
            objects = parse_yolo_label(label_file, img_width, img_height)
            
            for obj in objects:
                mask_area = obj['mask_area']
                bbox_area = obj['bbox_area']
                
                # Store statistics
                split_stats.append({
                    'file': label_file.name,
                    'mask_area': mask_area,
                    'bbox_area': bbox_area,
                    'size_category': categorize_by_coco_size(mask_area)
                })
                
                # Count size distribution
                size_category = categorize_by_coco_size(mask_area)
                size_distribution[size_category] += 1
                all_stats[split].extend([mask_area])
        
        if split_stats:
            areas = [s['mask_area'] for s in split_stats]
            print(f"  Objects: {len(split_stats)}")
            print(f"  Area range: {min(areas):.1f} - {max(areas):.1f} pixels")
            print(f"  Mean area: {np.mean(areas):.1f} pixels")
            print(f"  Median area: {np.median(areas):.1f} pixels")
    
    # Overall statistics
    print(f"\n" + "="*50)
    print("OVERALL DATASET STATISTICS")
    print("="*50)
    
    all_areas = []
    for split_areas in all_stats.values():
        all_areas.extend(split_areas)
    
    if all_areas:
        print(f"Total objects: {len(all_areas)}")
        print(f"Area range: {min(all_areas):.1f} - {max(all_areas):.1f} pixels")
        print(f"Mean area: {np.mean(all_areas):.1f} pixels")
        print(f"Median area: {np.median(all_areas):.1f} pixels")
        print(f"Std area: {np.std(all_areas):.1f} pixels")
        
        # COCO size category breakdown
        print(f"\nModified Size Category Distribution (COCO/10):")
        print(f"Small (< 102.4 px²): {size_distribution['small']} objects")
        print(f"Medium (102.4-921.6 px²): {size_distribution['medium']} objects") 
        print(f"Large (> 921.6 px²): {size_distribution['large']} objects")
        
        # Calculate percentages
        total = sum(size_distribution.values())
        if total > 0:
            print(f"\nPercentage breakdown:")
            for category, count in size_distribution.items():
                percentage = (count / total) * 100
                print(f"{category.capitalize()}: {percentage:.1f}%")
        
        # Analyze why APS/APM/APL might be zero
        print(f"\n" + "="*50)
        print("ANALYSIS FOR APS/APM/APL METRICS")
        print("="*50)
        
        if size_distribution['small'] == 0 and size_distribution['medium'] == 0 and size_distribution['large'] == 0:
            print("❌ ERROR: No objects found in any size category!")
            print("This suggests an issue with label parsing or area calculation.")
        elif size_distribution['small'] == 0:
            print("ℹ️  APS = 0 because no small objects (< 102.4 px²) found")
        elif size_distribution['medium'] == 0:
            print("ℹ️  APM = 0 because no medium objects (102.4-921.6 px²) found")
        elif size_distribution['large'] == 0:
            print("ℹ️  APL = 0 because no large objects (> 921.6 px²) found")
        
        # Create visualization
        create_size_distribution_plot(all_areas, size_distribution)
        
        # Save detailed analysis
        analysis_results = {
            'total_objects': len(all_areas),
            'area_stats': {
                'min': float(min(all_areas)),
                'max': float(max(all_areas)),
                'mean': float(np.mean(all_areas)),
                'median': float(np.median(all_areas)),
                'std': float(np.std(all_areas))
            },
            'size_distribution': dict(size_distribution),
            'coco_thresholds': {
                'small_threshold': 102.4,
                'medium_threshold': 921.6
            }
        }
        
        with open('object_size_analysis.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        print(f"\n💾 Detailed analysis saved to: object_size_analysis.json")

def create_size_distribution_plot(areas, size_distribution):
    """Create visualization of object size distribution"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histogram of areas
    ax1.hist(areas, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(102.4, color='red', linestyle='--', label='Small/Medium threshold (102.4)')
    ax1.axvline(921.6, color='orange', linestyle='--', label='Medium/Large threshold (921.6)')
    ax1.set_xlabel('Object Area (pixels²)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Object Areas (Modified Thresholds)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of size categories
    categories = list(size_distribution.keys())
    counts = list(size_distribution.values())
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    bars = ax2.bar(categories, counts, color=colors[:len(categories)])
    ax2.set_xlabel('Modified Size Category')
    ax2.set_ylabel('Number of Objects')
    ax2.set_title('Object Count by Modified Size Category (COCO/10)')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('object_size_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Size distribution plot saved to: object_size_analysis.png")

if __name__ == "__main__":
    analyze_dataset_sizes()