#!/usr/bin/env python3
"""
Script to analyze and visualize polymer annotations with their corresponding images.
Supports PP, PE, PS, and PET datasets.
Checks one-to-one relation between annotations and images and displays them.
"""

import json
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.patches as mpatches
from collections import defaultdict

def get_polymer_paths(polymer_type):
    """Get the correct paths for different polymer types."""
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "full_images")
    
    polymer_configs = {
        'pp': {
            'data_dir': f"{base_path}/pp",
            'json_file': f"{base_path}/pp/pp.json"
        },
        'pe': {
            'data_dir': f"{base_path}/PE", 
            'json_file': f"{base_path}/PE/pe.json"
        },
        'ps': {
            'data_dir': f"{base_path}/PS",
            'json_file': f"{base_path}/PS/PS.json"
        },
        'pet': {
            'data_dir': f"{base_path}/raw_data",
            'json_file': f"{base_path}/raw_data/pet.json"
        }
    }
    
    if polymer_type.lower() not in polymer_configs:
        raise ValueError(f"Unsupported polymer type: {polymer_type}. Supported types: {list(polymer_configs.keys())}")
    
    return polymer_configs[polymer_type.lower()]

def load_annotations(json_path, polymer_type):
    """Load and parse the COCO format annotations."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {polymer_type.upper()} annotation file with:")
    print(f"  - {len(data['images'])} images")
    print(f"  - {len(data['annotations'])} annotations")
    print(f"  - {len(data['categories'])} categories")
    
    return data

def analyze_image_annotation_relation(data, data_dir):
    """Analyze the relationship between images and annotations."""
    print("\n=== Image-Annotation Relationship Analysis ===")
    
    # Create mappings
    images_dict = {img['id']: img for img in data['images']}
    categories_dict = {cat['id']: cat for cat in data['categories']}
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    print(f"\nImage files found in directory: {len([f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])}")
    print(f"Images in annotations: {len(images_dict)}")
    print(f"Images with annotations: {len(annotations_by_image)}")
    
    # Check each image
    for image_id, image_info in images_dict.items():
        image_path = os.path.join(data_dir, image_info['file_name'])
        annotations_count = len(annotations_by_image[image_id])
        file_exists = os.path.exists(image_path)
        
        print(f"\nImage: {image_info['file_name']}")
        print(f"  - ID: {image_id}")
        print(f"  - File exists: {file_exists}")
        print(f"  - Dimensions: {image_info['width']}x{image_info['height']}")
        print(f"  - Annotations: {annotations_count}")
        
        if annotations_count > 0:
            for ann in annotations_by_image[image_id][:5]:  # Show first 5 annotations
                category = categories_dict[ann['category_id']]['name']
                print(f"    * Annotation {ann['id']}: {category}, Area: {ann['area']:.2f}")
            if annotations_count > 5:
                print(f"    ... and {annotations_count - 5} more annotations")
    
    return images_dict, annotations_by_image, categories_dict

def visualize_image_with_annotations(image_path, image_info, annotations, categories_dict):
    """Visualize a single image with its annotations."""
    # Load image
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image: {image_path}")
        return None
    
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"{image_info['file_name']} - {len(annotations)} annotations", fontsize=14)
    ax.axis('off')  # Hide axes for cleaner view
    
    # Colors for different categories
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink']
    category_colors = {}
    
    # Draw annotations
    for i, ann in enumerate(annotations):
        category_id = ann['category_id']
        category_name = categories_dict[category_id]['name']
        
        if category_id not in category_colors:
            category_colors[category_id] = colors[len(category_colors) % len(colors)]
        
        color = category_colors[category_id]
        
        # Draw segmentation if available
        if 'segmentation' in ann and ann['segmentation']:
            for seg in ann['segmentation']:
                if len(seg) >= 6 and len(seg) % 2 == 0:  # Need at least 3 points (6 coordinates) and even number
                    try:
                        # Convert to polygon points
                        points = np.array(seg).reshape(-1, 2)
                        polygon = Polygon(points, closed=True, fill=False, 
                                        edgecolor=color, linewidth=2, alpha=0.8)
                        ax.add_patch(polygon)
                    except ValueError as e:
                        print(f"Warning: Could not draw segmentation for annotation {ann['id']}: {e}")
                        print(f"  Segmentation data length: {len(seg)}")
                        continue
        
        # Draw bounding box
        if 'bbox' in ann and len(ann['bbox']) == 4:
            try:
                x, y, w, h = ann['bbox']
                if w > 0 and h > 0:  # Valid bounding box
                    rect = plt.Rectangle((x, y), w, h, fill=False, 
                                       edgecolor=color, linewidth=2, linestyle='--', alpha=0.7)
                    ax.add_patch(rect)
                    
                    # No text labels - clean visualization
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not draw bounding box for annotation {ann['id']}: {e}")
                continue
    
    # Create legend (optional - can be removed for even cleaner view)
    if category_colors:
        legend_elements = []
        for cat_id, color in category_colors.items():
            cat_name = categories_dict[cat_id]['name']
            legend_elements.append(mpatches.Patch(color=color, label=cat_name))
        ax.legend(handles=legend_elements, loc='upper right', frameon=False)
    
    ax.set_xlim(0, image_info['width'])
    ax.set_ylim(image_info['height'], 0)  # Flip Y axis for image coordinates
    
    return fig

def main():
    """Main function to analyze and visualize polymer annotations."""
    
    # Get polymer type from command line argument or default to PE
    if len(sys.argv) > 1:
        polymer_type = sys.argv[1].upper()
    else:
        print("Usage: python visualize_polymer_annotations.py <polymer_type>")
        print("Available polymer types: PP, PE, PS, PET")
        print("Example: python visualize_polymer_annotations.py PE")
        polymer_type = input("Enter polymer type (PP/PE/PS/PET): ").strip().upper()
    
    if not polymer_type:
        polymer_type = "PE"  # Default
    
    print(f"=== {polymer_type} Dataset Annotation Analyzer and Visualizer ===")
    
    try:
        # Get paths for the specified polymer
        config = get_polymer_paths(polymer_type)
        json_path = config['json_file']
        data_dir = config['data_dir']
        
        print(f"Using JSON file: {json_path}")
        print(f"Using image directory: {data_dir}")
        
        # Check if files exist
        if not os.path.exists(json_path):
            print(f"Error: JSON file not found: {json_path}")
            return
        if not os.path.exists(data_dir):
            print(f"Error: Image directory not found: {data_dir}")
            return
            
        # Load annotations
        data = load_annotations(json_path, polymer_type)
        
    except ValueError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Analyze relationships
    images_dict, annotations_by_image, categories_dict = analyze_image_annotation_relation(data, data_dir)
    
    print(f"\n=== Visualization ===")
    print("Displaying images with annotations...")
    
    # Visualize images with annotations
    images_with_annotations = [(img_id, img_info) for img_id, img_info in images_dict.items() 
                              if img_id in annotations_by_image and len(annotations_by_image[img_id]) > 0]
    
    if not images_with_annotations:
        print("No images found with annotations!")
        return
    
    print(f"Found {len(images_with_annotations)} images with annotations")
    
    # Show first few images
    for i, (image_id, image_info) in enumerate(images_with_annotations):  # Show all images
        image_path = os.path.join(data_dir, image_info['file_name'])
        annotations = annotations_by_image[image_id]
        
        print(f"\nShowing: {image_info['file_name']} ({len(annotations)} annotations)")
        
        fig = visualize_image_with_annotations(image_path, image_info, annotations, categories_dict)
        if fig:
            plt.show()
            plt.close(fig)  # Close after showing to avoid memory issues
    
    print(f"\nSummary:")
    print(f"- Total images in {polymer_type} dataset: {len(images_dict)}")
    print(f"- Images with annotations: {len(annotations_by_image)}")
    print(f"- Total annotations: {len(data['annotations'])}")
    print(f"- Categories: {', '.join([cat['name'] for cat in data['categories']])}")

if __name__ == "__main__":
    main()