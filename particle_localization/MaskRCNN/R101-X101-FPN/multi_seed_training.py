#!/usr/bin/env python3
"""
Multi-Seed Training Script for Improved Mask R-CNN
Trains models with 3 different random seeds and evaluates on test set
Reports average performance with standard deviation
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
from detectron2.utils.env import seed_all_rng
import random
from detectron2.engine import DefaultPredictor
import statistics

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

class ImprovedMicroplasticTrainer(DefaultTrainer):
    """Improved trainer for microplastic detection."""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, False, output_folder)

def set_seed(seed):
    """Set seed for reproducible training."""
    print(f"🌱 Setting random seed to {seed} for reproducible results")
    
    # Set Python random seed
    random.seed(seed)
    
    # Set NumPy seed
    np.random.seed(seed)
    
    # Set PyTorch seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set Detectron2 seed
    seed_all_rng(seed)
    
    # Make CUDA operations deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for CUDA
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_improved_cfg(total_images, seed):
    """Enhanced configuration for better microplastic detection."""
    cfg = get_cfg()
    
    # Use ResNet-101 for better feature extraction
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    
    # Dataset settings - use filtered datasets
    cfg.DATASETS.TRAIN = ("microplastic_train_single_filtered",)
    cfg.DATASETS.TEST = ("microplastic_test_single_filtered",)  # Use test set for evaluation
    cfg.DATALOADER.NUM_WORKERS = 2  # Reduced to avoid issues
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    
    # Enhanced FPN settings for multi-scale detection
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]  # Include res2 for higher resolution
    cfg.MODEL.FPN.OUT_CHANNELS = 384  # Balanced channels - not too heavy, not too light
    cfg.MODEL.BACKBONE.FREEZE_AT = 0   # Don't freeze any layers for better fine-tuning
    
    # Balanced anchor settings for both small and large particles
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 16],      # p2
        [24, 32],     # p3
        [48, 64],      # p4
        [128, 192]     # p5
    ]
    
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]
    
    # Enhanced RPN for multi-scale detection  
    cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5"]  # 4 features to match anchor sizes
    
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
    
    # Evaluation settings - disable periodic evaluation to avoid cache issues
    cfg.TEST.EVAL_PERIOD = 0  # Disable periodic evaluation during training
    cfg.SOLVER.CHECKPOINT_PERIOD = int(iterations_for_one_epoch * 20)  # Every 5 epochs
    
    # Output directory with seed
    models_dir = os.path.join(ZOO_ROOT, "multi_seed", f"seed_{seed}")
    os.makedirs(models_dir, exist_ok=True)
    cfg.OUTPUT_DIR = models_dir
    
    return cfg

def register_datasets_with_improvements():
    """Register datasets with single-class modification and small particle filtering."""
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
    
    register_coco_instances(
        "microplastic_test_single",
        {},
        os.path.join(dataset_root, "test", "annotations_single_class.json"),
        os.path.join(dataset_root, "test", "images")
    )
    
    MetadataCatalog.get("microplastic_train_single").thing_classes = ["Microplastic"]
    MetadataCatalog.get("microplastic_val_single").thing_classes = ["Microplastic"]
    MetadataCatalog.get("microplastic_test_single").thing_classes = ["Microplastic"]
    
    # Apply moderate small particle filtering (remove particles < 3 pixels)
    print("🔍 Applying moderate small particle filtering...")
    train_filtered = filter_small_annotations("microplastic_train_single", min_size_pixels=3)
    val_filtered = filter_small_annotations("microplastic_val_single", min_size_pixels=3)
    test_filtered = filter_small_annotations("microplastic_test_single", min_size_pixels=3)
    
    # Load and return filtered dataset size
    filtered_train_data = DatasetCatalog.get(train_filtered)
    filtered_test_data = DatasetCatalog.get(test_filtered)
    print(f"✅ Training with {len(filtered_train_data)} images after filtering")
    print(f"✅ Testing with {len(filtered_test_data)} images after filtering")
    return len(filtered_train_data)

def evaluate_model_on_test(cfg, test_dataset_name, seed):
    """Evaluate the trained model on test set."""
    # Load the trained model
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.25  # Use same threshold as training
    
    predictor = DefaultPredictor(cfg)
    
    # Create organized test evaluation directory
    test_eval_dir = os.path.join(cfg.OUTPUT_DIR, "test_evaluation")
    os.makedirs(test_eval_dir, exist_ok=True)
    
    # Run evaluation on test set with organized output
    evaluator = COCOEvaluator(test_dataset_name, output_dir=test_eval_dir)
    test_loader = build_detection_test_loader(cfg, test_dataset_name)
    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    
    # Save individual seed results
    seed_results_file = os.path.join(test_eval_dir, f"test_results_seed_{seed}.json")
    with open(seed_results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Test results for seed {seed} saved to: {seed_results_file}")
    
    return results

def train_single_seed(seed, total_images):
    """Train model with a single seed and return test results."""
    print(f"\n{'='*80}")
    print(f"TRAINING WITH SEED {seed}")
    print(f"{'='*80}")
    
    # Set seed
    set_seed(seed)
    
    # Get configuration
    cfg = get_improved_cfg(total_images, seed)
    
    print(f"\nTraining Configuration (Seed {seed}):")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    
    # Create trainer and train
    trainer = ImprovedMicroplasticTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print(f"\n🚀 Starting training with seed {seed}...")
    trainer.train()
    print(f"✅ Training completed for seed {seed}!")
    
    # Evaluate on test set
    print(f"\n📊 Evaluating model (seed {seed}) on test set...")
    test_results = evaluate_model_on_test(cfg, "microplastic_test_single_filtered", seed)
    
    # Extract key metrics
    metrics = {
        'seed': seed,
        'output_dir': cfg.OUTPUT_DIR
    }
    
    if 'segm' in test_results:
        segm = test_results['segm']
        metrics.update({
            'segm_AP': segm['AP'],
            'segm_AP50': segm['AP50'],
            'segm_AP75': segm['AP75'],
            'segm_APs': segm.get('APs', 0),
            'segm_APm': segm.get('APm', 0),
            'segm_APl': segm.get('APl', 0)
        })
    
    if 'bbox' in test_results:
        bbox = test_results['bbox']
        metrics.update({
            'bbox_AP': bbox['AP'],
            'bbox_AP50': bbox['AP50'],
            'bbox_AP75': bbox['AP75'],
            'bbox_APs': bbox.get('APs', 0),
            'bbox_APm': bbox.get('APm', 0),
            'bbox_APl': bbox.get('APl', 0)
        })
    
    print(f"\n📋 Results for seed {seed}:")
    print(f"  Segmentation AP: {metrics.get('segm_AP', 0):.3f}")
    print(f"  Segmentation AP50: {metrics.get('segm_AP50', 0):.3f}")
    print(f"  Bounding Box AP: {metrics.get('bbox_AP', 0):.3f}")
    
    return metrics

def compute_statistics(all_results):
    """Compute mean and standard deviation for all metrics."""
    print(f"\n{'='*80}")
    print("COMPUTING STATISTICS ACROSS ALL SEEDS")
    print(f"{'='*80}")
    
    # Define metrics to compute statistics for
    metric_names = [
        'segm_AP', 'segm_AP50', 'segm_AP75', 'segm_APs', 'segm_APm', 'segm_APl',
        'bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl'
    ]
    
    statistics_results = {}
    
    for metric in metric_names:
        values = [result.get(metric, 0) for result in all_results]
        if any(v > 0 for v in values):  # Only compute stats if we have non-zero values
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            statistics_results[metric] = {
                'mean': mean_val,
                'std': std_val,
                'values': values
            }
    
    return statistics_results

def save_results(all_results, statistics_results):
    """Save detailed results to JSON file in organized structure."""
    # Create main results directory
    main_results_dir = os.path.join(ZOO_ROOT, "multi_seed")
    os.makedirs(main_results_dir, exist_ok=True)
    
    # Save consolidated results
    output_file = os.path.join(main_results_dir, "consolidated_test_results.json")
    
    results_data = {
        'individual_results': all_results,
        'statistics': statistics_results,
        'summary': {
            'num_seeds': len(all_results),
            'seeds_used': [r['seed'] for r in all_results],
            'individual_model_dirs': [r['output_dir'] for r in all_results]
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    # Also save a summary CSV for easy analysis
    csv_file = os.path.join(main_results_dir, "test_results_summary.csv")
    with open(csv_file, 'w') as f:
        f.write("Seed,Segm_AP,Segm_AP50,Segm_AP75,BBox_AP,BBox_AP50,BBox_AP75,Model_Dir\n")
        for result in all_results:
            f.write(f"{result.get('seed', 'N/A')},"
                   f"{result.get('segm_AP', 0):.3f},"
                   f"{result.get('segm_AP50', 0):.3f},"
                   f"{result.get('segm_AP75', 0):.3f},"
                   f"{result.get('bbox_AP', 0):.3f},"
                   f"{result.get('bbox_AP50', 0):.3f},"
                   f"{result.get('bbox_AP75', 0):.3f},"
                   f"{result['output_dir']}\n")
    
    print(f"\n💾 Consolidated results saved to:")
    print(f"   JSON: {output_file}")
    print(f"   CSV:  {csv_file}")
    print(f"   Individual test results in each model's test_evaluation/ subdirectory")
    
    return output_file

def main():
    setup_logger()
    
    print("="*80)
    print("MULTI-SEED IMPROVED MASK R-CNN TRAINING & EVALUATION")
    print("="*80)
    print("This script will:")
    print("✓ Train models with 3 different random seeds")
    print("✓ Evaluate each model on the test set")
    print("✓ Report average performance with standard deviation")
    print("="*80)
    
    # Define seeds to use
    seeds = [42, 123, 456]  # 3 different seeds
    
    # Register datasets (only need to do this once)
    print("\n🗃️  Registering datasets...")
    total_images = register_datasets_with_improvements()
    
    # Train models with different seeds
    all_results = []
    
    for seed in seeds:
        try:
            result = train_single_seed(seed, total_images)
            all_results.append(result)
        except Exception as e:
            print(f"❌ Error training with seed {seed}: {e}")
            continue
    
    if len(all_results) == 0:
        print("❌ No successful training runs!")
        return
    
    # Compute statistics
    statistics_results = compute_statistics(all_results)
    
    # Print final results
    print(f"\n{'='*80}")
    print("FINAL MULTI-SEED RESULTS")
    print(f"{'='*80}")
    print(f"Successfully trained {len(all_results)} models with seeds: {[r['seed'] for r in all_results]}")
    
    print(f"\n📊 SEGMENTATION METRICS:")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Individual Values'}")
    print("-" * 70)
    
    seg_metrics = ['segm_AP', 'segm_AP50', 'segm_AP75', 'segm_APs', 'segm_APm', 'segm_APl']
    metric_labels = ['AP (0.5:0.95)', 'AP50', 'AP75', 'APs (small)', 'APm (medium)', 'APl (large)']
    
    for metric, label in zip(seg_metrics, metric_labels):
        if metric in statistics_results:
            stats = statistics_results[metric]
            values_str = ', '.join([f"{v:.3f}" for v in stats['values']])
            print(f"{label:<20} {stats['mean']:<10.3f} {stats['std']:<10.3f} [{values_str}]")
    
    print(f"\n📦 BOUNDING BOX METRICS:")
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Individual Values'}")
    print("-" * 70)
    
    bbox_metrics = ['bbox_AP', 'bbox_AP50', 'bbox_AP75', 'bbox_APs', 'bbox_APm', 'bbox_APl']
    
    for metric, label in zip(bbox_metrics, metric_labels):
        if metric in statistics_results:
            stats = statistics_results[metric]
            values_str = ', '.join([f"{v:.3f}" for v in stats['values']])
            print(f"{label:<20} {stats['mean']:<10.3f} {stats['std']:<10.3f} [{values_str}]")
    
    # Highlight key results
    if 'segm_AP' in statistics_results:
        segm_ap_mean = statistics_results['segm_AP']['mean']
        segm_ap_std = statistics_results['segm_AP']['std']
        print(f"\n🎯 KEY RESULT:")
        print(f"   Segmentation mAP: {segm_ap_mean:.3f} ± {segm_ap_std:.3f}")
    
    if 'segm_AP50' in statistics_results:
        segm_ap50_mean = statistics_results['segm_AP50']['mean']
        segm_ap50_std = statistics_results['segm_AP50']['std']
        print(f"   Segmentation AP50: {segm_ap50_mean:.3f} ± {segm_ap50_std:.3f}")
    
    # Save results
    save_results(all_results, statistics_results)
    
    print(f"\n{'='*80}")
    print("MULTI-SEED EVALUATION COMPLETED!")
    print(f"{'='*80}")
    print(f"📁 File Organization:")
    print(f"   Individual models:")
    for result in all_results:
        print(f"     - Seed {result['seed']}: {result['output_dir']}")
        print(f"       └── test_evaluation/: Test results, predictions, COCO format files")
    print(f"   Consolidated results:")
    print(f"     - {os.path.join(ZOO_ROOT, 'multi_seed')}/")
    print(f"       ├── consolidated_test_results.json")
    print(f"       └── test_results_summary.csv")

if __name__ == "__main__":
    main()