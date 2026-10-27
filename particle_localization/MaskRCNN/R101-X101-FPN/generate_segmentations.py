#!/usr/bin/env python3
"""
Generate Model Prediction Overlays
Uses trained models from zoo folder to generate prediction overlays on test images
Shows model predictions overlaid on original images for visualization
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

# Get project root directory (assuming script is in a subfolder of the project)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from R101-FPN folder
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches)
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo")

def modify_annotations_for_single_class():
    """
    Modify the existing annotations to treat all polymer types as 'Microplastic' class.
    """
    
    for split in ['train', 'val', 'test']:
        ann_file = os.path.join(DATA_ROOT, split, "annotations.json")
        
        with open(ann_file, 'r') as f:
            data = json.load(f)
        
        # Modify categories to have only one class
        data['categories'] = [
            {'id': 1, 'name': 'Microplastic', 'supercategory': 'microplastic'}
        ]
        
        # Update all annotations to use category_id = 1
        for ann in data['annotations']:
            ann['category_id'] = 1
        
        # Save modified annotations
        modified_ann_file = os.path.join(DATA_ROOT, split, "annotations_single_class.json")
        with open(modified_ann_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Created single-class annotations for {split}: {len(data['annotations'])} annotations")


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

def setup_model_config(model_path, model_type="R101"):
    """Setup configuration for inference matching the training setup."""
    cfg = get_cfg()
    
    if model_type == "R101":
        # ResNet-101 configuration (matching training)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        
        # Enhanced FPN settings (EXACTLY matching training)
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.OUT_CHANNELS = 384
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        
        # Enhanced anchor settings (EXACTLY matching training)
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
            [8, 16],      # p2
            [24, 32],     # p3
            [48, 64],     # p4
            [128, 192]    # p5
        ]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]
        
        # RPN settings (EXACTLY matching training)
        cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match 4 anchor sizes
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.6
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 384
        
    else:  # X101
        # ResNeXt-101 configuration (matching enhanced training setup)
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"))
        
        # Enhanced FPN settings (EXACTLY matching training)
        cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]
        cfg.MODEL.FPN.OUT_CHANNELS = 384
        cfg.MODEL.BACKBONE.FREEZE_AT = 0
        
        # Enhanced anchor settings (EXACTLY matching training)
        cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
            [8, 16],      # p2
            [24, 32],     # p3
            [48, 64],     # p4
            [128, 192]    # p5
        ]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]
        
        # RPN settings (EXACTLY matching training)
        cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match 4 anchor sizes
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
        cfg.MODEL.RPN.NMS_THRESH = 0.6
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 384
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # ROI settings
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # Lower threshold for better detection
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    
    # Input settings (matching training)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    return cfg

def register_datasets():
    """Register datasets for inference."""
    import sys
    
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

def create_prediction_overlay(image, predictions, gt_annotations=None, alpha=0.7):
    """Create colorful filled overlay with predictions only."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Polygon
    import numpy as np
    
    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img_rgb)
    ax.axis('off')
    
    # Define a variety of vibrant colors for masks
    colors = [
        'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'cyan', 'magenta',
        'lime', 'pink', 'brown', 'gray', 'olive', 'navy', 'teal', 'maroon',
        'gold', 'coral', 'indigo', 'crimson'
    ]
    
    # Draw predictions with filled colored masks
    if len(predictions) > 0:
        boxes = predictions.pred_boxes.tensor.numpy()
        scores = predictions.scores.numpy()
        masks = predictions.pred_masks.numpy()
        
        for i, (box, score, mask) in enumerate(zip(boxes, scores, masks)):
            x1, y1, x2, y2 = box
            
            # Select color cyclically
            color = colors[i % len(colors)]
            
            # Alpha based on confidence score - made more transparent
            alpha_score = min(0.4, max(0.15, score * 0.5))
            
            # Create filled mask overlay using matplotlib's built-in color mapping
            # This preserves the original background and only colors the mask areas
            mask_rgba = np.zeros((*mask.shape, 4))
            
            # Convert color name to RGB values
            color_map = {
                'red': [1.0, 0.0, 0.0], 'blue': [0.0, 0.0, 1.0], 'green': [0.0, 1.0, 0.0], 
                'yellow': [1.0, 1.0, 0.0], 'orange': [1.0, 0.65, 0.0], 'purple': [0.5, 0.0, 0.5],
                'cyan': [0.0, 1.0, 1.0], 'magenta': [1.0, 0.0, 1.0], 'lime': [0.2, 0.8, 0.2],
                'pink': [1.0, 0.75, 0.8], 'brown': [0.65, 0.16, 0.16], 'gray': [0.5, 0.5, 0.5],
                'olive': [0.5, 0.5, 0.0], 'navy': [0.0, 0.0, 0.5], 'teal': [0.0, 0.5, 0.5],
                'maroon': [0.5, 0.0, 0.0], 'gold': [1.0, 0.84, 0.0], 'coral': [1.0, 0.5, 0.31],
                'indigo': [0.29, 0.0, 0.51], 'crimson': [0.86, 0.08, 0.24]
            }
            
            rgb_color = color_map.get(color, [1.0, 0.0, 0.0])  # Default to red if color not found
            
            # Set the color and alpha for mask pixels only
            mask_rgba[mask, :3] = rgb_color
            mask_rgba[mask, 3] = alpha_score  # Alpha channel
            
            # Apply the colored mask overlay
            ax.imshow(mask_rgba, interpolation='nearest')
    
    # Title removed for cleaner visualization
    
    # Convert plot to image
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    # Convert ARGB to RGB
    buf = buf[:, :, [1, 2, 3]]
    buf = np.roll(buf, 3, axis=2)  # Roll RGB to BGR for OpenCV
    
    plt.close(fig)
    return buf

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

def load_ground_truth_annotations():
    """Load ground truth annotations for comparison."""
    dataset_root = DATA_ROOT
    ann_file = os.path.join(dataset_root, "test", "annotations_single_class.json")
    
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    # Create mapping from image_id to annotations
    gt_annotations = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in gt_annotations:
            gt_annotations[image_id] = []
        gt_annotations[image_id].append(ann)
    
    # Create mapping from filename to image_id
    filename_to_id = {}
    for img in data['images']:
        filename_to_id[img['file_name']] = img['id']
    
    return gt_annotations, filename_to_id

def generate_overlays_for_model(model_path, model_name, model_type, dataset_names, output_base_dir):
    """Generate prediction overlays for a single model."""
    print(f"\n{'='*60}")
    print(f"GENERATING OVERLAYS - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Type: {model_type}")
    
    # Create output directory for this model
    model_output_dir = os.path.join(output_base_dir, f"overlays_{model_name}")
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Setup configuration and predictor
    cfg = setup_model_config(model_path, model_type)
    predictor = DefaultPredictor(cfg)
    
    # Load ground truth annotations
    gt_annotations, filename_to_id = load_ground_truth_annotations()
    
    results_summary = {
        "model_name": model_name,
        "model_type": model_type,
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
            
            # Get ground truth for this image
            image_filename = os.path.basename(image_path)
            image_id = filename_to_id.get(image_filename, None)
            gt_anns = gt_annotations.get(image_id, []) if image_id else []
            
            # Create and save comprehensive overlay
            overlay_image = create_prediction_overlay(image, predictions, gt_anns)
            overlay_path = os.path.join(overlays_dir, f"{image_name}_overlay.jpg")
            cv2.imwrite(overlay_path, overlay_image)
            
            # Record image info with ground truth comparison
            image_info = {
                "image_name": image_name,
                "image_path": image_path,
                "num_detections": num_detections,
                "num_ground_truth": len(gt_anns),
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
    model_results_path = os.path.join(model_output_dir, "model_overlay_results.json")
    with open(model_results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ Model {model_name} overlay generation complete!")
    print(f"📁 Results saved in: {model_output_dir}")
    
    return results_summary

def find_zoo_models():
    """Find trained models from zoo folder."""
    zoo_dir = os.path.join(PROJECT_ROOT, "zoo")
    models = []
    
    if not os.path.exists(zoo_dir):
        print(f"Warning: Zoo directory not found: {zoo_dir}")
        return models
    
    # Look for model directories in zoo folder
    for item in os.listdir(zoo_dir):
        model_dir = os.path.join(zoo_dir, item)
        if os.path.isdir(model_dir):
            model_path = os.path.join(model_dir, "model_final.pth")
            
            if os.path.exists(model_path):
                # Determine model type from directory name
                model_type = "R101" if "R101" in item else "X101"
                
                models.append({
                    "name": item,
                    "type": model_type,
                    "model_dir": model_dir,
                    "model_path": model_path
                })
            else:
                print(f"Warning: model_final.pth not found in {model_dir}")
    
    return sorted(models, key=lambda x: x["name"])

def main():
    print("="*80)
    print("MODEL PREDICTION OVERLAY GENERATION")
    print("="*80)
    print("This script will:")
    print("✓ Find all trained models from zoo folder")
    print("✓ Generate prediction overlays on test dataset")
    print("✓ Compare predictions with ground truth")
    print("✓ Save overlays in organized directory structure")
    print("="*80)
    
    # Register datasets
    print("\n🗃️  Registering datasets...")
    dataset_names = register_datasets()
    print(f"Registered datasets: {list(dataset_names.keys())}")
    
    # Find zoo models
    print("\n🔍 Finding zoo models...")
    zoo_models = find_zoo_models()
    
    if not zoo_models:
        print("❌ No models found in zoo folder!")
        print("Expected models in: ../zoo/R101-FPN/ and ../zoo/X101-FPN/")
        return
    
    print(f"Found {len(zoo_models)} zoo models:")
    for model in zoo_models:
        print(f"  - {model['name']} ({model['type']}): {model['model_path']}")
    
    # Create base output directory
    output_base_dir = os.path.join(PROJECT_ROOT, "generated_overlays")
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Generate overlays for each model
    all_results = []
    for model in zoo_models:
        try:
            result = generate_overlays_for_model(
                model["model_path"],
                model["name"],
                model["type"],
                dataset_names,
                output_base_dir
            )
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error generating overlays for {model['name']}: {e}")
            continue
    
    # Save consolidated results
    consolidated_results = {
        "summary": {
            "num_models": len(all_results),
            "models_processed": [r["model_name"] for r in all_results],
            "output_base_dir": output_base_dir
        },
        "individual_results": all_results
    }
    
    consolidated_path = os.path.join(output_base_dir, "consolidated_overlay_results.json")
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_results, f, indent=2)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("PREDICTION OVERLAY GENERATION COMPLETED!")
    print(f"{'='*80}")
    print(f"📁 Output Directory Structure:")
    print(f"   {output_base_dir}/")
    for result in all_results:
        model_name = result["model_name"]
        print(f"   ├── overlays_{model_name}/")
        print(f"   │   └── test/")
        print(f"   │       ├── overlays/  (prediction + ground truth overlays)")
        print(f"   │       └── test_statistics.json")
        print(f"   │   └── model_overlay_results.json")
    print(f"   └── consolidated_overlay_results.json")
    print(f"\n🎯 Successfully generated prediction overlays for {len(all_results)} models!")
    print(f"\n📊 Each overlay shows:")
    print(f"   🔴 Red boxes/masks: Model predictions with confidence scores")
    print(f"   🟢 Green dashed boxes: Ground truth annotations")
    print(f"   📈 Title shows prediction vs ground truth counts")

if __name__ == "__main__":
    main()