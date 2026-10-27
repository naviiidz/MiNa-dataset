#!/usr/bin/env python3
"""
Generate Segmentations from Trained Models
Uses trained models to generate segmentation results on image patches
Saves results in organized directory structure
"""

import os
import json
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import transforms as T
from detectron2.utils.env import seed_all_rng
import random
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycocotools import mask as mask_util
from detectron2.structures import BoxMode

# Paths relative to the project folder (one level above this script's folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches)
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo")

def filter_small_annotations(dataset_name, min_size_pixels=5):
    """Filter out annotations with bounding boxes smaller than min_size_pixels."""
    print(f"Filtering annotations smaller than {min_size_pixels} pixels...")
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    filtered_dicts = []
    
    total_annotations = 0
    filtered_annotations = 0
    
    for record in dataset_dicts:
        filtered_record = record.copy()
        filtered_annotations_list = []
        
        for annotation in record["annotations"]:
            total_annotations += 1
            bbox = annotation["bbox"]
            width, height = bbox[2], bbox[3]  # COCO format: [x, y, width, height]
            
            # Keep annotation if both width and height are >= min_size_pixels
            if width >= min_size_pixels and height >= min_size_pixels:
                filtered_annotations_list.append(annotation)
            else:
                filtered_annotations += 1
        
        # Only keep images that still have annotations after filtering
        if filtered_annotations_list:
            filtered_record["annotations"] = filtered_annotations_list
            filtered_dicts.append(filtered_record)
    
    print(f"Filtered {filtered_annotations} out of {total_annotations} annotations")
    print(f"Kept {len(filtered_dicts)} images with valid annotations")
    
    # Re-register the filtered dataset
    DatasetCatalog.register(f"{dataset_name}_filtered", lambda: filtered_dicts)
    MetadataCatalog.get(f"{dataset_name}_filtered").set(
        thing_classes=MetadataCatalog.get(dataset_name).thing_classes
    )
    
    return f"{dataset_name}_filtered"

def setup_model_config(model_path):
    """Setup configuration for inference matching the training setup."""
    cfg = get_cfg()
    
    # Use ResNet-101 configuration (matching training)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Enhanced FPN settings (matching training)
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.FPN.OUT_CHANNELS = 384
    cfg.MODEL.BACKBONE.FREEZE_AT = 0
    
    # Enhanced anchor settings (matching training)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 16],      # p2
        [24, 32],     # p3
        [48, 64],      # p4
        [128, 192]     # p5
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]
    
    # RPN settings (matching training)
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4"]
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    
    # ROI settings (matching training)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 384
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6  # 75% confidence threshold for high precision
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    # Input settings (matching training)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def register_datasets():
    """Register datasets for inference."""
    import sys
    sys.path.append(SCRIPT_DIR)
    from train import modify_annotations_for_single_class
    
    # Use the same single-class modification
    modify_annotations_for_single_class()
    
    dataset_root = DATA_ROOT
    
    # Register datasets
    for split in ["train", "val", "test"]:
        register_coco_instances(
            f"microplastic_{split}_single",
            {},
            os.path.join(dataset_root, split, "annotations_single_class.json"),
            os.path.join(dataset_root, split, "images")
        )
        MetadataCatalog.get(f"microplastic_{split}_single").thing_classes = ["Microplastic"]
    
    # Apply filtering to match training
    print("🔍 Applying dataset filtering...")
    filtered_datasets = {}
    for split in ["train", "val", "test"]:
        filtered_name = filter_small_annotations(f"microplastic_{split}_single", min_size_pixels=3)
        filtered_datasets[split] = filtered_name
    
    return filtered_datasets

def create_segmentation_overlay(image, predictions, alpha=0.6):
    """Create visualization overlay with predictions."""
    metadata = MetadataCatalog.get("microplastic_test_single_filtered")
    
    # Create visualizer
    visualizer = Visualizer(
        image[:, :, ::-1],  # BGR to RGB
        metadata=metadata,
        scale=1.0,
        instance_mode=ColorMode.IMAGE_BW
    )
    
    # Draw predictions
    vis_output = visualizer.draw_instance_predictions(predictions)
    
    # Convert back to BGR for OpenCV
    overlay_image = vis_output.get_image()[:, :, ::-1]
    
    return overlay_image

def save_individual_masks(predictions, output_dir, image_name):
    """Save individual mask files for each detection."""
    masks_dir = os.path.join(output_dir, "individual_masks")
    os.makedirs(masks_dir, exist_ok=True)
    
    if len(predictions) == 0:
        return []
    
    masks = predictions.pred_masks.cpu().numpy()
    scores = predictions.scores.cpu().numpy()
    
    mask_files = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Save as PNG mask
        mask_filename = f"{image_name}_mask_{i:03d}_score_{score:.3f}.png"
        mask_path = os.path.join(masks_dir, mask_filename)
        
        # Convert boolean mask to 0-255 image
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(mask_path, mask_image)
        mask_files.append(mask_path)
    
    return mask_files

def save_combined_mask(predictions, output_dir, image_name):
    """Save combined mask of all detections."""
    combined_dir = os.path.join(output_dir, "combined_masks")
    os.makedirs(combined_dir, exist_ok=True)
    
    if len(predictions) == 0:
        # Create empty mask
        empty_mask = np.zeros((256, 256), dtype=np.uint8)  # Assuming 256x256 patches
        combined_path = os.path.join(combined_dir, f"{image_name}_combined_mask.png")
        cv2.imwrite(combined_path, empty_mask)
        return combined_path
    
    masks = predictions.pred_masks.cpu().numpy()
    
    # Combine all masks
    combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
    
    combined_path = os.path.join(combined_dir, f"{image_name}_combined_mask.png")
    cv2.imwrite(combined_path, combined_mask)
    
    return combined_path

def save_prediction_data(predictions, output_dir, image_name):
    """Save prediction data as JSON."""
    data_dir = os.path.join(output_dir, "prediction_data")
    os.makedirs(data_dir, exist_ok=True)
    
    prediction_data = {
        "image_name": image_name,
        "num_detections": len(predictions),
        "detections": []
    }
    
    if len(predictions) > 0:
        boxes = predictions.pred_boxes.tensor.cpu().numpy()
        scores = predictions.scores.cpu().numpy()
        masks = predictions.pred_masks.cpu().numpy()
        
        for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            # Convert mask to RLE for compact storage
            mask_rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            mask_rle['counts'] = mask_rle['counts'].decode('utf-8')  # Convert bytes to string
            
            detection = {
                "detection_id": i,
                "bbox": box.tolist(),  # [x1, y1, x2, y2]
                "score": float(score),
                "mask_rle": mask_rle,
                "area": float(np.sum(mask))
            }
            prediction_data["detections"].append(detection)
    
    data_path = os.path.join(data_dir, f"{image_name}_predictions.json")
    with open(data_path, 'w') as f:
        json.dump(prediction_data, f, indent=2)
    
    return data_path

def generate_segmentations_for_model(model_path, seed, dataset_names, output_base_dir):
    """Generate segmentations for a single model."""
    print(f"\n{'='*60}")
    print(f"GENERATING SEGMENTATIONS - SEED {seed}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_base_dir, f"segmentations_seed_{seed}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Setup configuration and predictor
    cfg = setup_model_config(model_path)
    predictor = DefaultPredictor(cfg)
    
    results_summary = {
        "seed": seed,
        "model_path": model_path,
        "output_dir": model_output_dir,
        "dataset_results": {}
    }
    
    # Process only test dataset
    split = "test"
    dataset_name = dataset_names[split]
    print(f"\n📊 Processing {split} set ({dataset_name})...")
    
    # Create split-specific output directory
    split_output_dir = os.path.join(model_output_dir, split)
    os.makedirs(split_output_dir, exist_ok=True)
    
    # Create only overlays directory
    overlays_dir = os.path.join(split_output_dir, "overlays")
    os.makedirs(overlays_dir, exist_ok=True)
    
    # Get dataset
    dataset_dicts = DatasetCatalog.get(dataset_name)
    
    split_stats = {
        "total_images": len(dataset_dicts),
        "images_with_detections": 0,
        "total_detections": 0,
        "processed_images": []
    }
    
    # Process each image
    for idx, dataset_dict in enumerate(tqdm(dataset_dicts, desc=f"Processing {split}")):
        try:
            # Load image
            image_path = dataset_dict["file_name"]
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not load {image_path}")
                continue
            
            # Get image name without extension
            image_name = Path(image_path).stem
            
            # Run inference
            outputs = predictor(image)
            predictions = outputs["instances"].to("cpu")
            
            # Count detections
            num_detections = len(predictions)
            if num_detections > 0:
                split_stats["images_with_detections"] += 1
                split_stats["total_detections"] += num_detections
            
            # Create and save overlay only
            overlay_image = create_segmentation_overlay(image, predictions)
            overlay_path = os.path.join(overlays_dir, f"{image_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)
            
            # Record only basic image info
            image_info = {
                "image_name": image_name,
                "image_path": image_path,
                "num_detections": num_detections,
                "overlay_path": overlay_path
            }
            split_stats["processed_images"].append(image_info)
            
        except Exception as e:
            print(f"Error processing {dataset_dict.get('file_name', 'unknown')}: {e}")
            continue
    
    print(f"✅ {split} set complete:")
    print(f"   Images processed: {split_stats['total_images']}")
    print(f"   Images with detections: {split_stats['images_with_detections']}")
    print(f"   Total detections: {split_stats['total_detections']}")
    
    # Save split statistics
    split_stats_path = os.path.join(split_output_dir, f"{split}_statistics.json")
    with open(split_stats_path, 'w') as f:
        json.dump(split_stats, f, indent=2)
    
    results_summary["dataset_results"][split] = split_stats
    
    # Save overall model results
    model_results_path = os.path.join(model_output_dir, "model_segmentation_results.json")
    with open(model_results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Model {seed} segmentation generation complete!")
    print(f"📁 Results saved in: {model_output_dir}")
    
    return results_summary

def find_trained_models():
    """Find all trained models from multi-seed training."""
    base_dirs = [
        os.path.join(ZOO_ROOT, "multi_seed")
    ]
    models = []
    
    # Look for seed-specific model directories in multiple locations
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        for item in os.listdir(base_dir):
            if item.startswith("seed_"):
                seed = item.replace("seed_", "")
                model_dir = os.path.join(base_dir, item)
                model_path = os.path.join(model_dir, "model_final.pth")
                
                if os.path.exists(model_path):
                    models.append({
                        "seed": seed,
                        "model_dir": model_dir,
                        "model_path": model_path
                    })
                else:
                    print(f"Warning: model_final.pth not found in {model_dir}")
    
    # Also check for the main improved model (assign seed "main")
    main_model_dirs = [
        os.path.join(ZOO_ROOT, "R101-FPN")
    ]
    
    for model_dir in main_model_dirs:
        model_path = os.path.join(model_dir, "model_final.pth")
        if os.path.exists(model_path):
            models.append({
                "seed": "main",
                "model_dir": model_dir,
                "model_path": model_path
            })
            break  # Only add one main model
    
    return sorted(models, key=lambda x: (x["seed"] == "main", x["seed"]))

def main():
    print("="*80)
    print("SEGMENTATION GENERATION FROM TRAINED MODELS")
    print("="*80)
    print("This script will:")
    print("✓ Find all trained models from multi-seed training")
    print("✓ Generate overlay visualizations on test dataset only")
    print("✓ Save overlays in organized directory structure")
    print("="*80)
    
    # Register datasets
    print("\n🗃️  Registering datasets...")
    dataset_names = register_datasets()
    print(f"Registered datasets: {list(dataset_names.keys())}")
    
    # Find trained models
    print("\n🔍 Finding trained models...")
    trained_models = find_trained_models()
    
    if not trained_models:
        print("❌ No trained models found!")
        print(f"Expected directory pattern: {os.path.join(ZOO_ROOT, 'multi_seed', 'seed_*')}")
        return
    
    print(f"Found {len(trained_models)} trained models:")
    for model in trained_models:
        print(f"  - Seed {model['seed']}: {model['model_path']}")
    
    # Create base output directory
    output_base_dir = os.path.join(PROJECT_ROOT, "generated_segmentations")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate segmentations for each model
    all_results = []
    for model in trained_models:
        try:
            result = generate_segmentations_for_model(
                model["model_path"],
                model["seed"],
                dataset_names,
                output_base_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error generating segmentations for seed {model['seed']}: {e}")
            continue
    
    # Save consolidated results
    consolidated_results = {
        "summary": {
            "num_models": len(all_results),
            "seeds_processed": [r["seed"] for r in all_results],
            "output_base_dir": output_base_dir
        },
        "individual_results": all_results
    }
    
    consolidated_path = os.path.join(output_base_dir, "consolidated_segmentation_results.json")
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("SEGMENTATION GENERATION COMPLETED!")
    print(f"{'='*80}")
    print(f"📁 Output Directory Structure:")
    print(f"   {output_base_dir}/")
    for result in all_results:
        seed = result["seed"]
        print(f"   ├── segmentations_seed_{seed}/")
        print(f"   │   └── test/")
        print(f"   │       ├── overlays/")
        print(f"   │       └── test_statistics.json")
        print(f"   └── model_segmentation_results.json")
    print(f"   └── consolidated_segmentation_results.json")
    print(f"\n🎯 Successfully generated test overlays for {len(all_results)} models!")

if __name__ == "__main__":
    main()