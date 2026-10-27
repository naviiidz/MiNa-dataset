#!/usr/bin/env python3
"""
Simplified Improved Mask R-CNN Training Script for Microplastic Detection
Key improvements without complex augmentation pipeline
"""

import os
import json
import torch
import detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
import cv2
import numpy as np
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import transforms as T
from sklearn.model_selection import KFold
import random
from collections import defaultdict
import statistics

# Paths relative to the project folder (one level above this script's folder)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches)
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo")

def create_kfold_splits(dataset_dicts, n_splits=5, random_seed=42):
    """Create k-fold splits from dataset."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # Shuffle the dataset
    shuffled_indices = list(range(len(dataset_dicts)))
    random.shuffle(shuffled_indices)
    
    # Create k-fold splits
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    
    folds = []
    for fold_idx, (train_indices, val_indices) in enumerate(kfold.split(shuffled_indices)):
        train_data = [dataset_dicts[shuffled_indices[i]] for i in train_indices]
        val_data = [dataset_dicts[shuffled_indices[i]] for i in val_indices]
        
        folds.append({
            'fold': fold_idx,
            'train': train_data,
            'val': val_data,
            'train_size': len(train_data),
            'val_size': len(val_data)
        })
        
        print(f"Fold {fold_idx + 1}: {len(train_data)} train, {len(val_data)} val")
    
    return folds

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

class ImprovedMicroplasticTrainer(DefaultTrainer):
    """Improved trainer for microplastic detection."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def get_improved_cfg(total_images, train_dataset_name, val_dataset_name, fold_idx):
    """Enhanced configuration for better microplastic detection."""
    cfg = get_cfg()
    
    # Use ResNet-101 for better feature extraction
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    # Dataset settings - use fold-specific datasets
    cfg.DATASETS.TRAIN = (train_dataset_name,)
    cfg.DATASETS.TEST = (val_dataset_name,)
    cfg.DATALOADER.NUM_WORKERS = 2  # Reduced to avoid issues
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Enhanced FPN settings for multi-scale detection
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]  # Include res2 for higher resolution
    cfg.MODEL.FPN.OUT_CHANNELS = 384  # Balanced channels - not too heavy, not too light
    cfg.MODEL.BACKBONE.FREEZE_AT = 0   # Don't freeze any layers for better fine-tuning
    
    # Balanced anchor settings for both small and large particles - consistent across all levels
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 12, 16],    # p2: targets ~8–16 px objects (3 anchors)
        [20, 28, 36],   # p3: targets ~20–36 px objects (3 anchors)
        [48, 64, 80],   # p4: medium objects (3 anchors)
        [96, 128, 160], # p5: larger objects (3 anchors)
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

    # Enhanced RPN for multi-scale detection  
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match 4 anchor sizes
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 4000  # Increased for multi-scale
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000   
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000 # Increased for diverse proposals
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
    
    # RPN settings optimized for small objects
    cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 3000  # Increased
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 2000   # Increased
    cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1500 # Increased
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000  # Increased
    cfg.MODEL.RPN.NMS_THRESH = 0.6           # Slightly lower for small objects
    
    # Balanced ROI settings for multi-scale object detection
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 384  # Balanced batch size
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5     # Standard ratio
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25    # Low threshold for sensitivity
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4       # Standard NMS for both scales
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # Multi-scale features
    
    # Loss balancing for multi-scale objects
    cfg.MODEL.RPN.LOSS_WEIGHT = 1.0  # Standard RPN loss weight
    cfg.MODEL.ROI_HEADS.LOSS_WEIGHT = 1.0  # Standard ROI loss weight
    
    # Balanced input resolution - not too extreme
    cfg.INPUT.MIN_SIZE_TRAIN = (400, 600, 800)      # Moderate resolution range
    cfg.INPUT.MAX_SIZE_TRAIN = 1333                  # Standard max size
    cfg.INPUT.MIN_SIZE_TEST = 800                    # Standard test resolution
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Training settings with improved schedule
    cfg.SOLVER.IMS_PER_BATCH = 2  # Reduced for high-resolution images and memory efficiency
    cfg.SOLVER.BASE_LR = 0.0005   # Conservative learning rate
    
    # Training schedule - moderate length for testing
    iterations_for_one_epoch = total_images / cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER = int(iterations_for_one_epoch * 25)  # 25 epochs
    cfg.SOLVER.STEPS = [
        int(iterations_for_one_epoch * 15),   # Reduce LR at 15 epochs
        int(iterations_for_one_epoch * 20)    # Reduce LR at 20 epochs
    ]
    cfg.SOLVER.GAMMA = 0.5  # LR reduction factor
    cfg.SOLVER.WARMUP_ITERS = int(iterations_for_one_epoch * 1)  # 1 epoch warmup
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    # Keep the higher resolution settings from above - do not override
    # cfg.INPUT settings are already configured for high-resolution processing
    
    # Evaluation settings - disable periodic evaluation to avoid cache issues
    cfg.TEST.EVAL_PERIOD = 0  # Disable periodic evaluation during training
    cfg.SOLVER.CHECKPOINT_PERIOD = int(iterations_for_one_epoch * 10)  # Every 10 epochs (reduced frequency)
    
    # Limit number of checkpoints to save disk space
    cfg.SOLVER.MAX_TO_KEEP = 2  # Only keep 2 most recent checkpoints + final model
    
    # Output directory - fold-specific
    models_dir = os.path.join(ZOO_ROOT, "5-fold", f"fold_{fold_idx}")
    os.makedirs(models_dir, exist_ok=True)
    cfg.OUTPUT_DIR = models_dir
    
    return cfg

def register_fold_datasets(fold_data, fold_idx):
    """Register datasets for a specific fold."""
    train_name = f"microplastic_train_fold_{fold_idx}"
    val_name = f"microplastic_val_fold_{fold_idx}"
    
    # Register fold-specific datasets
    DatasetCatalog.register(train_name, lambda: fold_data['train'])
    DatasetCatalog.register(val_name, lambda: fold_data['val'])
    
    MetadataCatalog.get(train_name).thing_classes = ["Microplastic"]
    MetadataCatalog.get(val_name).thing_classes = ["Microplastic"]
    
    return train_name, val_name

def prepare_base_dataset():
    """Prepare and filter the base dataset for k-fold splitting."""
    import sys
    sys.path.append(SCRIPT_DIR)
    from train import modify_annotations_for_single_class
    
    # Use the same single-class modification
    modify_annotations_for_single_class()
    
    dataset_root = DATA_ROOT
    
    # Register original datasets first
    register_coco_instances(
        "microplastic_train_single",
        {},
        os.path.join(dataset_root, "train", "annotations_single_class.json"),
        os.path.join(dataset_root, "train", "images")
    )
    
    register_coco_instances(
        "microplastic_val_single",
        {},
        os.path.join(dataset_root, "val", "annotations_single_class.json"),
        os.path.join(dataset_root, "val", "images")
    )
    
    MetadataCatalog.get("microplastic_train_single").thing_classes = ["Microplastic"]
    MetadataCatalog.get("microplastic_val_single").thing_classes = ["Microplastic"]
    
    # Apply moderate small particle filtering (remove particles < 3 pixels)
    print("🔍 Applying moderate small particle filtering...")
    train_filtered = filter_small_annotations("microplastic_train_single", min_size_pixels=3)
    val_filtered = filter_small_annotations("microplastic_val_single", min_size_pixels=3)
    
    # Combine train and val for k-fold splitting
    train_data = DatasetCatalog.get(train_filtered)
    val_data = DatasetCatalog.get(val_filtered)
    
    # Combine all data for k-fold splitting
    all_data = train_data + val_data
    print(f"✅ Total dataset: {len(all_data)} images after filtering")
    print(f"   - Original train: {len(train_data)} images")
    print(f"   - Original val: {len(val_data)} images")
    
    return all_data

def evaluate_improved_model(cfg, fold_idx, val_dataset_name):
    """Evaluate the improved model for a specific fold."""
    print("\n" + "="*60)
    print(f"IMPROVED MODEL EVALUATION - FOLD {fold_idx + 1}")
    print("="*60)
    
    # Load the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    from detectron2.engine import DefaultPredictor
    predictor = DefaultPredictor(cfg)
    
    # Run evaluation on validation set
    evaluator = COCOEvaluator(val_dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "evaluation"))
    val_loader = build_detection_test_loader(cfg, val_dataset_name)
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    print("\nImproved Model Results:")
    print("="*40)
    
    # Extract key metrics
    if 'segm' in results:
        segm_results = results['segm']
        print(f"Segmentation AP (IoU=0.50:0.95): {segm_results['AP']:.3f}")
        print(f"Segmentation AP50 (IoU=0.50): {segm_results['AP50']:.3f}")
        print(f"Segmentation AP75 (IoU=0.75): {segm_results['AP75']:.3f}")
    
    if 'bbox' in results:
        bbox_results = results['bbox']
        print(f"Bounding Box AP (IoU=0.50:0.95): {bbox_results['AP']:.3f}")
        print(f"Bounding Box AP50 (IoU=0.50): {bbox_results['AP50']:.3f}")
    
    return results

def aggregate_fold_results(fold_results):
    """Aggregate results across all folds."""
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION RESULTS SUMMARY")
    print("="*70)
    
    # Collect metrics across folds
    segm_aps = []
    segm_ap50s = []
    segm_ap75s = []
    bbox_aps = []
    bbox_ap50s = []
    
    print("\nPer-fold results:")
    print("-" * 50)
    
    for fold_idx, results in enumerate(fold_results):
        print(f"\nFold {fold_idx + 1}:")
        
        if 'segm' in results:
            segm_ap = results['segm']['AP']
            segm_ap50 = results['segm']['AP50']
            segm_ap75 = results['segm']['AP75']
            segm_aps.append(segm_ap)
            segm_ap50s.append(segm_ap50)
            segm_ap75s.append(segm_ap75)
            print(f"  Segm AP: {segm_ap:.3f}, AP50: {segm_ap50:.3f}, AP75: {segm_ap75:.3f}")
        
        if 'bbox' in results:
            bbox_ap = results['bbox']['AP']
            bbox_ap50 = results['bbox']['AP50']
            bbox_aps.append(bbox_ap)
            bbox_ap50s.append(bbox_ap50)
            print(f"  BBox AP: {bbox_ap:.3f}, AP50: {bbox_ap50:.3f}")
    
    # Calculate aggregate statistics
    print("\n" + "="*50)
    print("AGGREGATE RESULTS ACROSS 5 FOLDS")
    print("="*50)
    
    if segm_aps:
        print(f"\nSegmentation Metrics:")
        print(f"  AP (mean ± std): {statistics.mean(segm_aps):.3f} ± {statistics.stdev(segm_aps):.3f}")
        print(f"  AP50 (mean ± std): {statistics.mean(segm_ap50s):.3f} ± {statistics.stdev(segm_ap50s):.3f}")
        print(f"  AP75 (mean ± std): {statistics.mean(segm_ap75s):.3f} ± {statistics.stdev(segm_ap75s):.3f}")
    
    if bbox_aps:
        print(f"\nBounding Box Metrics:")
        print(f"  AP (mean ± std): {statistics.mean(bbox_aps):.3f} ± {statistics.stdev(bbox_aps):.3f}")
        print(f"  AP50 (mean ± std): {statistics.mean(bbox_ap50s):.3f} ± {statistics.stdev(bbox_ap50s):.3f}")
    
    # Save aggregated results
    aggregate_results = {
        'segmentation': {
            'AP_mean': statistics.mean(segm_aps) if segm_aps else 0,
            'AP_std': statistics.stdev(segm_aps) if len(segm_aps) > 1 else 0,
            'AP50_mean': statistics.mean(segm_ap50s) if segm_ap50s else 0,
            'AP50_std': statistics.stdev(segm_ap50s) if len(segm_ap50s) > 1 else 0,
            'AP75_mean': statistics.mean(segm_ap75s) if segm_ap75s else 0,
            'AP75_std': statistics.stdev(segm_ap75s) if len(segm_ap75s) > 1 else 0,
        },
        'bbox': {
            'AP_mean': statistics.mean(bbox_aps) if bbox_aps else 0,
            'AP_std': statistics.stdev(bbox_aps) if len(bbox_aps) > 1 else 0,
            'AP50_mean': statistics.mean(bbox_ap50s) if bbox_ap50s else 0,
            'AP50_std': statistics.stdev(bbox_ap50s) if len(bbox_ap50s) > 1 else 0,
        },
        'fold_results': fold_results
    }
    
    # Save to file
    results_file = os.path.join(ZOO_ROOT, "5-fold", "5fold_cv_results.json")
    with open(results_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return aggregate_results

def main():
    setup_logger()
    
    print("="*70)
    print("5-FOLD CROSS-VALIDATION MASK R-CNN MICROPLASTIC DETECTION")
    print("="*70)
    print("Key Improvements:")
    print("✓ ResNet-101 backbone (stronger features)")
    print("✓ Optimized anchors for small particles")
    print("✓ Better RPN and ROI head settings")
    print("✓ Improved training schedule")
    print("✓ Lower detection thresholds")
    print("✓ 5-fold cross-validation for robust evaluation")
    print("="*70)
    
    # Prepare base dataset
    all_data = prepare_base_dataset()
    
    # Create 5-fold splits
    print(f"\n📊 Creating 5-fold splits...")
    folds = create_kfold_splits(all_data, n_splits=5, random_seed=42)
    
    # Store results from each fold
    fold_results = []
    
    # Train and evaluate each fold
    for fold_idx, fold_data in enumerate(folds):
        print(f"\n" + "="*70)
        print(f"TRAINING FOLD {fold_idx + 1}/5")
        print(f"Training images: {fold_data['train_size']}")
        print(f"Validation images: {fold_data['val_size']}")
        print("="*70)
        
        # Register fold-specific datasets
        train_name, val_name = register_fold_datasets(fold_data, fold_idx)
        
        # Get fold-specific configuration
        cfg = get_improved_cfg(fold_data['train_size'], train_name, val_name, fold_idx)
        
        print(f"\nFold {fold_idx + 1} Configuration:")
        print(f"Train dataset: {train_name}")
        print(f"Val dataset: {val_name}")
        print(f"Output directory: {cfg.OUTPUT_DIR}")
        print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
        print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
        print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
        
        # Create trainer and train
        trainer = ImprovedMicroplasticTrainer(cfg)
        trainer.resume_or_load(resume=False)
        
        print(f"\n🚀 Starting training for fold {fold_idx + 1}...")
        trainer.train()
        print(f"✅ Training completed for fold {fold_idx + 1}!")
        
        # Evaluate the model for this fold
        eval_results = evaluate_improved_model(cfg, fold_idx, val_name)
        fold_results.append(eval_results)
        
        # Save fold-specific results
        fold_results_file = os.path.join(cfg.OUTPUT_DIR, "fold_results.json")
        with open(fold_results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        print(f"💾 Fold {fold_idx + 1} results saved to: {fold_results_file}")
    
    # Aggregate results across all folds
    print(f"\n📈 Aggregating results across all 5 folds...")
    aggregate_results = aggregate_fold_results(fold_results)
    
    print("\n" + "="*70)
    print("5-FOLD CROSS-VALIDATION COMPLETED!")
    print("="*70)
    print(f"Individual fold models saved in: {os.path.join(ZOO_ROOT, '5-fold', 'fold_*')}")
    print(f"Aggregate results saved in: {os.path.join(ZOO_ROOT, '5-fold', '5fold_cv_results.json')}")
    print("\nNext steps for further improvement:")
    print("- Analyze fold-to-fold variance")
    print("- Create ensemble from best folds")
    print("- Implement hyperparameter optimization")
    print("- Try different augmentation strategies")
    
    return aggregate_results

if __name__ == "__main__":
    main()