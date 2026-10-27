#!/usr/bin/env python3
"""
Combine patches from all polymer types into unified dataset.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

def setup_combined_directories(base_dir):
    """Create directory structure for combined patches."""
    dirs = {
        'base': base_dir,
        'train': os.path.join(base_dir, 'train'),
        'val': os.path.join(base_dir, 'val'),
        'test': os.path.join(base_dir, 'test'),
        'train_images': os.path.join(base_dir, 'train', 'images'),
        'val_images': os.path.join(base_dir, 'val', 'images'),
        'test_images': os.path.join(base_dir, 'test', 'images')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs

def combine_datasets(base_data_dir, oversample=None):
    """Combine all polymer patch datasets in base_data_dir into unified dataset.
    
    oversample: {polymer: n} repeats each training patch of that polymer that has at
    least one annotation n times in total (as copies named <patch>_rep<k>.png).
    Val and test are never oversampled.
    """
    oversample = oversample or {}

    print("Combining patch datasets from all polymers...")

    # Paths
    polymer_dirs = {
        'PP': f"{base_data_dir}/patches_256_pp",
        'PE': f"{base_data_dir}/patches_256_pe", 
        'PS': f"{base_data_dir}/patches_256_ps",
        'PET': f"{base_data_dir}/patches_256_pet"
    }
    
    combined_dir = f"{base_data_dir}/patches_256_combined"
    print(f"Output directory: {combined_dir}")

    # Start from an empty folder so patches from an earlier split are not left behind
    if os.path.isdir(combined_dir):
        print(f"Removing previous patches in {combined_dir}")
        shutil.rmtree(combined_dir)

    # Setup combined directories
    combined_dirs = setup_combined_directories(combined_dir)
    
    # Category mapping (1=PP, 2=PE, 3=PS, 4=PET)
    category_mapping = {'PP': 1, 'PE': 2, 'PS': 3, 'PET': 4}
    
    # Combined categories
    combined_categories = [
        {'id': 1, 'name': 'PP', 'supercategory': 'microplastic'},
        {'id': 2, 'name': 'PE', 'supercategory': 'microplastic'},
        {'id': 3, 'name': 'PS', 'supercategory': 'microplastic'},
        {'id': 4, 'name': 'PET', 'supercategory': 'microplastic'}
    ]
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\nProcessing {split} split...")
        
        combined_images = []
        combined_annotations = []
        current_image_id = 0
        current_annotation_id = 0
        
        stats = defaultdict(lambda: {'images': 0, 'annotations': 0})
        
        # Process each polymer
        for polymer, polymer_dir in polymer_dirs.items():
            split_ann_file = os.path.join(polymer_dir, split, 'annotations.json')
            split_img_dir = os.path.join(polymer_dir, split, 'images')
            
            if not os.path.exists(split_ann_file):
                print(f"Warning: {split_ann_file} not found, skipping {polymer} {split}")
                continue
                
            # Load polymer data
            with open(split_ann_file, 'r') as f:
                polymer_data = json.load(f)
            
            print(f"  {polymer}: {len(polymer_data['images'])} images, {len(polymer_data['annotations'])} annotations")
            
            # Image ID mapping
            old_to_new_image_id = {}
            
            # Process images
            for img in polymer_data['images']:
                # Copy image file
                old_img_path = os.path.join(split_img_dir, img['file_name'])
                new_img_path = os.path.join(combined_dirs[f'{split}_images'], img['file_name'])
                
                if os.path.exists(old_img_path):
                    shutil.copy2(old_img_path, new_img_path)
                    
                    # Update image metadata
                    old_to_new_image_id[img['id']] = current_image_id
                    
                    new_img = {
                        'id': current_image_id,
                        'width': img['width'],
                        'height': img['height'],
                        'file_name': img['file_name'],
                        'polymer_type': polymer,
                        'source_image': img.get('source_image', ''),
                        'patch_position': img.get('patch_position', []),
                        'source_region': img.get('source_region', [])
                    }
                    
                    combined_images.append(new_img)
                    current_image_id += 1
                    stats[polymer]['images'] += 1
                else:
                    print(f"Warning: Image file not found: {old_img_path}")
            
            # Process annotations
            for ann in polymer_data['annotations']:
                if ann['image_id'] in old_to_new_image_id:
                    new_ann = {
                        'id': current_annotation_id,
                        'image_id': old_to_new_image_id[ann['image_id']],
                        'category_id': category_mapping[polymer],  # Map to unified category
                        'segmentation': ann['segmentation'],
                        'area': ann['area'],
                        'bbox': ann['bbox'],
                        'iscrowd': ann.get('iscrowd', 0),
                        'source_annotation_id': ann.get('source_annotation_id', ann.get('id', -1))
                    }
                    
                    combined_annotations.append(new_ann)
                    current_annotation_id += 1
                    stats[polymer]['annotations'] += 1
        
        # Oversample rare polymers in the training split only
        if split == 'train' and oversample:
            anns_by_image = defaultdict(list)
            for ann in combined_annotations:
                anns_by_image[ann['image_id']].append(ann)
            for img in list(combined_images):
                n = oversample.get(img['polymer_type'], 1)
                if n <= 1 or not anns_by_image[img['id']]:
                    continue
                stem, ext = os.path.splitext(img['file_name'])
                for k in range(1, n):
                    copy_name = f"{stem}_rep{k}{ext}"
                    shutil.copy2(os.path.join(combined_dirs['train_images'], img['file_name']),
                                 os.path.join(combined_dirs['train_images'], copy_name))
                    combined_images.append({**img, 'id': current_image_id, 'file_name': copy_name,
                                            'oversample_copy_of': img['file_name']})
                    for ann in anns_by_image[img['id']]:
                        combined_annotations.append({**ann, 'id': current_annotation_id, 'image_id': current_image_id})
                        current_annotation_id += 1
                    stats[img['polymer_type']]['images'] += 1
                    stats[img['polymer_type']]['annotations'] += len(anns_by_image[img['id']])
                    current_image_id += 1
            print(f"  Oversampled {oversample} -> {len(combined_images)} train images")
        
        # Create combined annotation file
        combined_data = {
            'info': {
                'description': f'Combined microplastic patches - {split} set',
                'version': '1.0',
                'year': 2024,
                'contributor': 'MaskRCNN-MiNa Project',
                'date_created': '2024-01-01'
            },
            'categories': combined_categories,
            'images': combined_images,
            'annotations': combined_annotations
        }
        
        # Save combined annotations
        combined_ann_file = os.path.join(combined_dirs[split], 'annotations.json')
        with open(combined_ann_file, 'w') as f:
            json.dump(combined_data, f, indent=2)
        
        print(f"  Combined {split}: {len(combined_images)} images, {len(combined_annotations)} annotations")
        
        # Print statistics
        for polymer, stat in stats.items():
            print(f"    {polymer}: {stat['images']} images, {stat['annotations']} annotations")
    
    print(f"\nCombined dataset created successfully at {combined_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Combine per-polymer patches into patches_256_combined/")
    parser.add_argument("--data_root",
                        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "patches", "coco"),
                        help="Folder holding patches_256_<polymer>/ (default: <repo>/data/patches/coco)")
    parser.add_argument("--oversample", nargs="*", default=[], metavar="POLYMER=N",
                        help="Repeat training patches (with particles) of a polymer N times in total, e.g. PE=3")
    args = parser.parse_args()
    oversample = {k.upper(): int(v) for k, v in (item.split("=") for item in args.oversample)}
    combine_datasets(args.data_root, oversample)