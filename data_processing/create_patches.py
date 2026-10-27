#!/usr/bin/env python3
"""
Improved Patch Generation Script for Microplastic Datasets

LOGIC EXPLANATION:
==================

1. IMAGE PREPROCESSING:
   - Load full-size images (typically 1280x960 or similar)
   - Exclude bottom 128 rows to remove the SEM info bar (it starts at row 832 of 960)
   - This gives us effective image height = original_height - 128

2. PATCH EXTRACTION STRATEGY:
   - Use 256x256 pixel patches (standard size for deep learning)
   - Cover the whole effective area with a grid of patches
   - Calculate: patches_x = ceil(width / 256), patches_y = ceil(effective_height / 256),
     spread evenly so the last patch ends at the edge (1280x832 -> 5x4 patches,
     no overlap across, 64 px overlap between rows)

3. ANNOTATION PROCESSING:
   - For each patch, find annotations that intersect with patch boundaries
   - Use Shapely library for precise geometric operations:
     a) Convert COCO polygon to Shapely Polygon
     b) Create patch bounding box as Shapely box
     c) Use intersection() to clip polygon to patch boundaries
     d) Convert back to COCO format with patch-relative coordinates

4. QUALITY FILTERING:
   - Remove annotations with area < 4 pixels (noise)
   - Remove degenerate polygons (< 3 vertices)
   - Validate all coordinates are within [0, 256] range
   - Recalculate bounding boxes from clipped polygons

5. DATA ORGANIZATION:
   - Maintain COCO format for compatibility
   - Create sequential patch IDs and annotation IDs
   - Split into train/val/test with 70/15/15 ratio by source image,
     so all patches of one image stay in the same split (no leakage)
   - Preserve polymer type metadata in image info

6. ERROR HANDLING:
   - Gracefully handle malformed polygons
   - Skip corrupted annotations with warnings
   - Validate geometric operations
   - Ensure patch dimensions match expected 256x256

This approach ensures high-quality patches with accurate annotations
that are properly clipped to patch boundaries while maintaining
the original annotation semantics.
"""

import json
import os
import sys
import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import random
from pathlib import Path
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import Point, box
from shapely import __version__ as shapely_version
import shutil
import warnings

# Paths: the common data/ folder at the repository root (next to data_processing/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # data_processing/ sits at the repo root
DATA_ROOT = os.path.join(REPO_ROOT, "data")
OUTPUT_ROOT = os.path.join(DATA_ROOT, "patches", "coco")  # Where patches_256_<polymer>/ is written; set with --output_root

# Configuration
PATCH_SIZE = 256
SCALE_BAR_HEIGHT = 128  # Rows to exclude from bottom (SEM info bar, measured: starts at row 832 of 960)

EXCLUDED_IMAGES = {
    "PEL 500X.png",
    "PE-10000X.png",
    "PE 1000-1.png",
    "PE 2000X2.png",
    "PE-3000X.png",
    "PE-5000x.png",
    "PE 500-3.png",
}
MIN_ANNOTATION_AREA = 4.0  # Minimum area in pixels
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def setup_directories(output_base_dir: str) -> Dict[str, str]:
    """Create directory structure for patches."""
    dirs = {
        'base': output_base_dir,
        'train': os.path.join(output_base_dir, 'train'),
        'val': os.path.join(output_base_dir, 'val'), 
        'test': os.path.join(output_base_dir, 'test'),
        'train_images': os.path.join(output_base_dir, 'train', 'images'),
        'val_images': os.path.join(output_base_dir, 'val', 'images'),
        'test_images': os.path.join(output_base_dir, 'test', 'images')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def get_polymer_config() -> Dict[str, Dict[str, str]]:
    """Get configuration for all polymer types."""
    base_dir = os.path.join(DATA_ROOT, "full_images")
    
    return {
        "PP": {
            "json_file": f"{base_dir}/PP/pp.json",
            "images_dir": f"{base_dir}/PP"
        },
        "PE": {
            "json_file": f"{base_dir}/PE/pe.json", 
            "images_dir": f"{base_dir}/PE"
        },
        "PS": {
            "json_file": f"{base_dir}/PS/ps.json",
            "images_dir": f"{base_dir}/PS"
        },
        "PET": {
            "json_file": f"{base_dir}/PET/pet.json",
            "images_dir": f"{base_dir}/PET"
        }
    }

def validate_polygon_coordinates(coords: List[float]) -> bool:
    """Validate polygon coordinates."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return False
    
    # Check for valid numeric values
    try:
        for coord in coords:
            float(coord)
        return True
    except (ValueError, TypeError):
        return False

def clip_polygon_to_patch(polygon_coords: List[float],
                         patch_x: int, patch_y: int,
                         patch_size: int = PATCH_SIZE) -> List[List[float]]:
    """
    Clip polygon to patch boundaries using Shapely.

    A patch edge can cut one polygon into several pieces; all of them are kept.

    Returns:
        List of clipped polygons (each a flat coordinate list) in the
        patch-relative system, or empty list
    """
    try:
        # Validate input
        if not validate_polygon_coordinates(polygon_coords):
            return []
        
        # Convert to coordinate pairs
        points = []
        for i in range(0, len(polygon_coords), 2):
            x = float(polygon_coords[i])
            y = float(polygon_coords[i + 1])
            points.append((x, y))
        
        if len(points) < 3:
            return []
        
        # Create Shapely polygon
        try:
            original_polygon = ShapelyPolygon(points)
            if not original_polygon.is_valid:
                # Try to fix invalid polygon
                original_polygon = original_polygon.buffer(0)
                if not original_polygon.is_valid:
                    return []
        except Exception:
            return []
        
        # Create patch bounding box
        patch_box = box(patch_x, patch_y, 
                       patch_x + patch_size, patch_y + patch_size)
        
        # Check intersection
        if not original_polygon.intersects(patch_box):
            return []
        
        # Perform clipping
        try:
            clipped = original_polygon.intersection(patch_box)
        except Exception:
            return []
        
        # Handle different geometry types
        if clipped.is_empty:
            return []
        
        # Collect every polygon piece (ignore lines/points left on the patch edge)
        if hasattr(clipped, 'exterior'):
            pieces = [clipped]
        elif hasattr(clipped, 'geoms'):
            pieces = [geom for geom in clipped.geoms
                      if hasattr(geom, 'exterior') and geom.area > 0]
        else:
            pieces = []

        clipped_polygons = []
        for piece in pieces:
            extracted_coords = list(piece.exterior.coords[:-1])  # Remove duplicate last point
            if len(extracted_coords) < 3:
                continue

            # Convert to patch-relative coordinates
            relative_coords = []
            for x, y in extracted_coords:
                rel_x = x - patch_x
                rel_y = y - patch_y
                # Ensure coordinates are within patch bounds
                rel_x = max(0, min(patch_size, rel_x))
                rel_y = max(0, min(patch_size, rel_y))
                relative_coords.extend([rel_x, rel_y])
            clipped_polygons.append(relative_coords)

        return clipped_polygons
        
    except Exception as e:
        warnings.warn(f"Error clipping polygon: {e}")
        return []

def calculate_polygon_area(coords: List[float]) -> float:
    """Calculate polygon area using shoelace formula."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return 0.0
    
    try:
        points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
        n = len(points)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        return abs(area) / 2.0
    except Exception:
        return 0.0

def calculate_bbox_from_polygon(coords: List[float]) -> List[float]:
    """Calculate bounding box from polygon coordinates."""
    if len(coords) < 6 or len(coords) % 2 != 0:
        return [0, 0, 0, 0]
    
    try:
        x_coords = [coords[i] for i in range(0, len(coords), 2)]
        y_coords = [coords[i] for i in range(1, len(coords), 2)]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        return [min_x, min_y, max_x - min_x, max_y - min_y]
    except Exception:
        return [0, 0, 0, 0]

def patch_starts(length: int, patch_size: int = PATCH_SIZE) -> List[int]:
    """Start offsets of patches covering [0, length): as few as needed, spread evenly,
    the last one ending at the edge (adjacent patches overlap when length is not a multiple)."""
    if length <= patch_size:
        return [0]
    n = -(-length // patch_size)  # ceil
    return [round(i * (length - patch_size) / (n - 1)) for i in range(n)]

def create_patches_for_image(image_path: str, image_info: Dict, 
                           annotations: List[Dict], 
                           patch_id_start: int, 
                           annotation_id_start: int,
                           polymer_type: str) -> Tuple[List[Dict], List[Dict], int, int]:
    """
    Create patches from a single image with improved logic.
    
    Returns:
        (patch_images, patch_annotations, next_patch_id, next_annotation_id)
    """
    print(f"Processing {image_info['file_name']}...")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load {image_path}")
        return [], [], patch_id_start, annotation_id_start
    
    height, width = image.shape[:2]
    effective_height = height - SCALE_BAR_HEIGHT
    
    if effective_height <= 0:
        print(f"Warning: Image too small after scale bar exclusion: {image_path}")
        return [], [], patch_id_start, annotation_id_start
    
    print(f"  Image size: {width}x{height}, effective: {width}x{effective_height}")
    
    # Calculate patch grid
    if width < PATCH_SIZE or effective_height < PATCH_SIZE:
        print(f"Warning: Image smaller than one patch after scale bar exclusion: {image_path}")
        return [], [], patch_id_start, annotation_id_start
    
    xs = patch_starts(width, PATCH_SIZE)
    ys = patch_starts(effective_height, PATCH_SIZE)
    patches_x, patches_y = len(xs), len(ys)
    total_patches = patches_x * patches_y
    
    print(f"  Creating {patches_x}x{patches_y} = {total_patches} patches")
    
    patch_images = []
    patch_annotations = []
    current_patch_id = patch_id_start
    current_annotation_id = annotation_id_start
    
    valid_patches = 0
    total_annotations = 0
    
    for y in range(patches_y):
        for x in range(patches_x):
            # Calculate patch boundaries
            start_x = xs[x]
            start_y = ys[y]
            end_x = start_x + PATCH_SIZE
            end_y = start_y + PATCH_SIZE
            
            # Extract patch from image
            patch_image = image[start_y:end_y, start_x:end_x]
            
            # Create patch filename
            patch_filename = f"{polymer_type.lower()}_{os.path.splitext(image_info['file_name'])[0]}_patch_{x:03d}_{y:03d}.png"
            
            # Create patch metadata
            patch_info = {
                'id': current_patch_id,
                'width': PATCH_SIZE,
                'height': PATCH_SIZE,
                'file_name': patch_filename,
                'polymer_type': polymer_type,
                'source_image': image_info['file_name'],
                'patch_position': [x, y],
                'source_region': [start_x, start_y, PATCH_SIZE, PATCH_SIZE]
            }
            
            # Process annotations for this patch
            patch_anns = []
            
            for ann in annotations:
                if 'segmentation' not in ann or not ann['segmentation']:
                    continue
                
                # Check if annotation might overlap with patch
                bbox = ann.get('bbox', [0, 0, 0, 0])
                ann_x, ann_y, ann_w, ann_h = bbox
                
                # Quick overlap check
                if (ann_x >= end_x or ann_x + ann_w <= start_x or 
                    ann_y >= end_y or ann_y + ann_h <= start_y):
                    continue
                
                # Process all segmentation polygons
                clipped_segmentations = []
                total_area = 0
                
                for seg in ann['segmentation']:
                    if not isinstance(seg, list) or len(seg) < 6:
                        continue
                    
                    for clipped_seg in clip_polygon_to_patch(seg, start_x, start_y, PATCH_SIZE):
                        area = calculate_polygon_area(clipped_seg)
                        if area >= MIN_ANNOTATION_AREA:
                            clipped_segmentations.append(clipped_seg)
                            total_area += area

                # Create patch annotation if valid clipped segmentations exist
                if clipped_segmentations and total_area >= MIN_ANNOTATION_AREA:
                    # Bounding box covers all pieces of the particle in this patch
                    new_bbox = calculate_bbox_from_polygon(
                        [c for seg in clipped_segmentations for c in seg])
                    
                    patch_ann = {
                        'id': current_annotation_id,
                        'image_id': current_patch_id,
                        'category_id': ann['category_id'],
                        'segmentation': clipped_segmentations,
                        'area': total_area,
                        'bbox': new_bbox,
                        'iscrowd': ann.get('iscrowd', 0),
                        'source_annotation_id': ann.get('id', -1)
                    }
                    
                    patch_anns.append(patch_ann)
                    current_annotation_id += 1
            
            # Add patch even if it has no annotations (empty patches are useful)
            patch_images.append((patch_info, patch_image))
            patch_annotations.extend(patch_anns)
            
            valid_patches += 1
            total_annotations += len(patch_anns)
            current_patch_id += 1
    
    print(f"  Created {valid_patches} patches with {total_annotations} total annotations")
    return patch_images, patch_annotations, current_patch_id, current_annotation_id

def save_patches_and_annotations(patch_data: List[Tuple], 
                                patch_annotations: List[Dict],
                                output_dirs: Dict[str, str],
                                split_name: str,
                                categories: List[Dict]) -> None:
    """Save patches and create COCO annotation file."""
    
    # Save patch images
    images_dir = output_dirs[f'{split_name}_images']
    patch_metadata = []
    
    print(f"Saving {len(patch_data)} patches to {split_name}...")
    
    for patch_info, patch_image in patch_data:
        # Save patch image
        patch_path = os.path.join(images_dir, patch_info['file_name'])
        success = cv2.imwrite(patch_path, patch_image)
        
        if success:
            patch_metadata.append(patch_info)
        else:
            print(f"Warning: Failed to save {patch_path}")
    
    # Create COCO format annotation file
    coco_data = {
        'info': {
            'description': f'Microplastic patches - {split_name} set',
            'version': '1.0',
            'year': 2024,
            'contributor': 'MaskRCNN-MiNa Project',
            'date_created': '2024-01-01'
        },
        'categories': categories,
        'images': patch_metadata,
        'annotations': [ann for ann in patch_annotations 
                       if ann['image_id'] in [img['id'] for img in patch_metadata]]
    }
    
    # Save annotations
    ann_path = os.path.join(output_dirs[split_name], 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Saved {len(patch_metadata)} images and {len(coco_data['annotations'])} annotations for {split_name}")

def create_patches_for_polymer(polymer_type: str) -> None:
    """Create patches for a specific polymer type with improved logic."""
    
    print(f"\n{'='*60}")
    print(f"CREATING PATCHES FOR {polymer_type}")
    print(f"{'='*60}")
    print(f"Shapely version: {shapely_version}")
    print(f"Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
    print(f"Scale bar exclusion: {SCALE_BAR_HEIGHT} pixels")
    print(f"Min annotation area: {MIN_ANNOTATION_AREA} pixels")
    
    # Get polymer configuration
    config = get_polymer_config()
    if polymer_type not in config:
        raise ValueError(f"Unknown polymer type: {polymer_type}")
    
    polymer_config = config[polymer_type]
    
    # Load data
    print(f"Loading data from {polymer_config['json_file']}")
    with open(polymer_config['json_file'], 'r') as f:
        coco_data = json.load(f)
    
    print(f"Loaded {len(coco_data['images'])} images and {len(coco_data['annotations'])} annotations")
    
    # Setup output directories
    output_dir = os.path.join(OUTPUT_ROOT, f"patches_256_{polymer_type.lower()}")
    print(f"Output directory: {output_dir}")

    # Start from an empty folder so patches from an earlier split are not left behind
    if os.path.isdir(output_dir):
        print(f"Removing previous patches in {output_dir}")
        shutil.rmtree(output_dir)

    output_dirs = setup_directories(output_dir)
    
    # Process all images
    all_patch_data = []
    all_annotations = []
    patch_id = 0
    annotation_id = 0
    
    # Group annotations by image
    annotations_by_image = defaultdict(list)
    for ann in coco_data['annotations']:
        annotations_by_image[ann['image_id']].append(ann)
    
    # Process each image
    for img_info in coco_data['images']:
        if img_info['file_name'] in EXCLUDED_IMAGES:
            continue
        image_path = os.path.join(polymer_config['images_dir'], img_info['file_name'])
        image_annotations = annotations_by_image[img_info['id']]
        
        patches, patch_anns, patch_id, annotation_id = create_patches_for_image(
            image_path, img_info, image_annotations, 
            patch_id, annotation_id, polymer_type
        )
        
        all_patch_data.extend(patches)
        all_annotations.extend(patch_anns)
    
    if not all_patch_data:
        print("No patches created!")
        return
    
    print(f"\nTotal patches created: {len(all_patch_data)}")
    print(f"Total annotations: {len(all_annotations)}")
    
    # Split by source image so patches of one image never land in different splits
    source_images = sorted({info['source_image'] for info, _ in all_patch_data})
    random.seed(42)  # For reproducible splits
    random.shuffle(source_images)

    n_images = len(source_images)
    n_val = max(1, round(n_images * VAL_SPLIT)) if n_images >= 3 else 0
    n_test = max(1, round(n_images * TEST_SPLIT)) if n_images >= 3 else 0
    n_train = n_images - n_val - n_test

    split_of_image = {}
    for i, name in enumerate(source_images):
        split_of_image[name] = 'train' if i < n_train else ('val' if i < n_train + n_val else 'test')

    train_indices, val_indices, test_indices = [], [], []
    for idx, (info, _) in enumerate(all_patch_data):
        {'train': train_indices, 'val': val_indices, 'test': test_indices}[split_of_image[info['source_image']]].append(idx)

    print(f"Split by source image: train={n_train}, val={n_val}, test={n_test} images")
    print(f"Split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)} patches")

    # Record which source images went to which split
    split_file = os.path.join(output_dir, 'split_source_images.json')
    with open(split_file, 'w') as f:
        json.dump({s: sorted(n for n, v in split_of_image.items() if v == s)
                   for s in ['train', 'val', 'test']}, f, indent=2)
    print(f"Source-image split saved to {split_file}")
    
    # Save splits
    for split_name, indices in [('train', train_indices), ('val', val_indices), ('test', test_indices)]:
        if not indices:
            continue
            
        split_patches = [all_patch_data[i] for i in indices]
        split_patch_ids = {all_patch_data[i][0]['id'] for i in indices}
        split_annotations = [ann for ann in all_annotations 
                           if ann['image_id'] in split_patch_ids]
        
        save_patches_and_annotations(
            split_patches, split_annotations, 
            output_dirs, split_name, coco_data['categories']
        )
    
    print(f"\nPatches for {polymer_type} completed successfully!")

def main():
    """Main function to create patches for all polymers or specified one."""
    
    global OUTPUT_ROOT, PATCH_SIZE
    import argparse
    parser = argparse.ArgumentParser(description="Create square patches (default 256x256) from full-size images")
    parser.add_argument("polymer_type", help="PP, PE, PS, PET, or ALL")
    parser.add_argument("--output_root", default=OUTPUT_ROOT,
                        help="Folder to write patches_256_<polymer>/ into (default: <repo>/data/patches/coco)")
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE,
                        help="Patch side in pixels (default: 256). Folder names keep the patches_256_ prefix.")
    args = parser.parse_args()

    OUTPUT_ROOT = args.output_root
    PATCH_SIZE = args.patch_size
    polymer_type = args.polymer_type.upper()
    
    if polymer_type == "ALL":
        for ptype in ["PP", "PE", "PS", "PET"]:
            try:
                create_patches_for_polymer(ptype)
            except Exception as e:
                print(f"Error processing {ptype}: {e}")
                import traceback
                traceback.print_exc()
    else:
        create_patches_for_polymer(polymer_type)

if __name__ == "__main__":
    main()