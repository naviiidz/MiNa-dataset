#!/usr/bin/env python3
"""
Faster R-CNN Training Script for Microplastic Detection
Optimized for ResNet-50 backbone with FPN
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

# Get project root directory (assuming script is in a subfolder of the project)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # Go up one level from R101-FPN folder
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(PROJECT_ROOT)), "data", "patches", "coco", "patches_256_combined")  # common data folder (data/patches): full image area, PE x3 in train
ZOO_ROOT = os.path.join(PROJECT_ROOT, "zoo", "current")  # models trained on data/patches (older models stay in zoo/)
EVAL_SCORE_THRESH = 0.05  # COCO AP needs low-score detections (detectron2 default)
ANNOTATIONS = "annotations.json"  # "annotations_single_class.json" with --single-class
CLASSES = ["PP", "PE", "PS", "PET"]  # ["Microplastic"] with --single-class


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

def get_improved_cfg(total_images, model_variant="FasterRCNN-R50", run_suffix=""):
    """Enhanced configuration for Faster R-CNN microplastic detection."""
    cfg = get_cfg()
    
    # Configure model based on variant
    if model_variant == "FasterRCNN-R50":
        config_file = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    elif model_variant == "FasterRCNN-R101":
        config_file = "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"
    elif model_variant == "FasterRCNN-X101":
        config_file = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
    else:
        raise ValueError(f"Unknown model variant: {model_variant}. Choose 'FasterRCNN-R50', 'FasterRCNN-R101', or 'FasterRCNN-X101'")
    
    # Load configuration and weights based on variant
    cfg.merge_from_file(model_zoo.get_config_file(config_file))
    
    # Dataset settings - use filtered multi-class datasets
    cfg.DATASETS.TRAIN = ("microplastic_train_multi_filtered",)
    cfg.DATASETS.TEST = ("microplastic_val_multi",)
    cfg.DATALOADER.NUM_WORKERS = 2  # Reduced to avoid issues
    
    # Model settings
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(CLASSES)  # 4 polymers, or 1 with --single-class
    
    # Enhanced FPN settings for multi-scale detection
    cfg.MODEL.FPN.IN_FEATURES = ["res2", "res3", "res4", "res5"]  # Include res2 for higher resolution
    cfg.MODEL.FPN.OUT_CHANNELS = 384  # Balanced channels - not too heavy, not too light
    cfg.MODEL.BACKBONE.FREEZE_AT = 0   # Don't freeze any layers for better fine-tuning
    
    # Balanced anchor settings for both small and large particles
    # Enhanced anchor settings (matching training)
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [
        [8, 16],      # p2
        [24, 32],     # p3
        [48, 64],      # p4
        [128, 192]     # p5
    ]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.25, 0.5, 1.0, 2.0, 4]]

                                                 

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
    cfg.SOLVER.CHECKPOINT_PERIOD = int(iterations_for_one_epoch * 5)  # Every 5 epochs
    
    # Output directory - save in model zoo
    if model_variant == "FasterRCNN-R50":
        models_dir = os.path.join(ZOO_ROOT, "FasterRCNN-R50")
    elif model_variant == "FasterRCNN-R101":
        models_dir = os.path.join(ZOO_ROOT, "FasterRCNN-R101")
    elif model_variant == "FasterRCNN-X101":
        models_dir = os.path.join(ZOO_ROOT, "FasterRCNN-X101")
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    models_dir += run_suffix
    os.makedirs(models_dir, exist_ok=True)
    cfg.OUTPUT_DIR = models_dir
    
    return cfg

def register_datasets_with_improvements():
    """Register datasets with small particle filtering while keeping original classes."""
    
    # Register original datasets with multiple polymer classes
    register_coco_instances(
        "microplastic_train_multi",
        {},
        os.path.join(DATA_ROOT, "train", ANNOTATIONS),
        os.path.join(DATA_ROOT, "train", "images")
    )
    
    register_coco_instances(
        "microplastic_val_multi",
        {},
        os.path.join(DATA_ROOT, "val", ANNOTATIONS),
        os.path.join(DATA_ROOT, "val", "images")
    )
    
    register_coco_instances(
        "microplastic_test_multi",
        {},
        os.path.join(DATA_ROOT, "test", ANNOTATIONS),
        os.path.join(DATA_ROOT, "test", "images")
    )
    
    # Set thing classes for multiple polymer types (matching annotation file order)
    MetadataCatalog.get("microplastic_train_multi").thing_classes = CLASSES
    MetadataCatalog.get("microplastic_val_multi").thing_classes = CLASSES
    MetadataCatalog.get("microplastic_test_multi").thing_classes = CLASSES
    
    # Apply moderate small particle filtering (remove particles < 3 pixels)
    print("🔍 Applying moderate small particle filtering...")
    # Only the training set is filtered; val/test are evaluated in full, including empty patches
    train_filtered = filter_small_annotations("microplastic_train_multi", min_size_pixels=3)
    
    # Load and return filtered dataset size
    filtered_train_data = DatasetCatalog.get(train_filtered)
    print(f"Training with {len(filtered_train_data)} images after filtering")
    return len(filtered_train_data)

def evaluate_improved_model(cfg):
    """Evaluate the trained model with COCO metrics on the full val and test splits."""
    print("\n" + "="*60)
    print("MODEL EVALUATION (val + test)")
    print("="*60)
    
    # Load the trained model; keep low-score detections so COCO AP sees the whole PR curve
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = EVAL_SCORE_THRESH
    
    from detectron2.engine import DefaultPredictor
    predictor = DefaultPredictor(cfg)
    
    all_results = {"weights": cfg.MODEL.WEIGHTS, "score_thresh": EVAL_SCORE_THRESH}
    for split in ["val", "test"]:
        dataset_name = f"microplastic_{split}_multi"
        evaluator = COCOEvaluator(dataset_name, output_dir=os.path.join(cfg.OUTPUT_DIR, "evaluation", split))
        loader = build_detection_test_loader(cfg, dataset_name)
        results = inference_on_dataset(predictor.model, loader, evaluator)
        all_results[split] = {task: dict(metrics) for task, metrics in results.items()}
        all_results[split]["num_images"] = len(DatasetCatalog.get(dataset_name))
        
        print(f"\n{split.upper()} results ({all_results[split]['num_images']} images):")
        for task in ("bbox", "segm"):
            if task in results:
                r = results[task]
                print(f"  {task}: AP={r['AP']:.2f}  AP50={r['AP50']:.2f}  AP75={r['AP75']:.2f}  "
                      f"APs={r['APs']:.2f}  APm={r['APm']:.2f}  APl={r['APl']:.2f}")
    
    results_path = os.path.join(cfg.OUTPUT_DIR, "results", "test_metrics.json")
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMetrics saved to {results_path}")
    
    return all_results

def set_seed(seed=42):
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

def main(max_iter=None, model_variant="FasterRCNN-R50", seed=42, single_class=False):
    global ANNOTATIONS, CLASSES
    if single_class:
        ANNOTATIONS, CLASSES = "annotations_single_class.json", ["Microplastic"]
    run_suffix = ("_agnostic" if single_class else "") + (f"_seed{seed}" if seed != 42 else "")
    # Set seed for reproducible results
    set_seed(seed)
    
    setup_logger()
    
    print("="*70)
    print("FASTER R-CNN MULTI-CLASS MICROPLASTIC DETECTION")
    print(f"Classes: {', '.join(CLASSES)}")
    print("="*70)

    
    # Register datasets
    total_images = register_datasets_with_improvements()
    
    # Get improved configuration
    cfg = get_improved_cfg(total_images, model_variant=model_variant, run_suffix=run_suffix)
    if max_iter:  # short dry run; outputs go to a separate folder
        cfg.SOLVER.MAX_ITER = max_iter
        cfg.OUTPUT_DIR = cfg.OUTPUT_DIR + f"_dryrun{max_iter}"
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    if model_variant == "FasterRCNN-R50":
        backbone_name = "ResNet-50 (Faster R-CNN)"
    elif model_variant == "FasterRCNN-R101":
        backbone_name = "ResNet-101 (Faster R-CNN)"
    elif model_variant == "FasterRCNN-X101":
        backbone_name = "ResNeXt-101 (Faster R-CNN)"
    
    print(f"\nFaster R-CNN Training Configuration:")
    print(f"Backbone: {backbone_name} + FPN")
    print(f"Model Variant: {model_variant}")
    print(f"Output directory: {cfg.OUTPUT_DIR}")
    print(f"Max iterations: {cfg.SOLVER.MAX_ITER}")
    print(f"Learning rate: {cfg.SOLVER.BASE_LR}")
    print(f"Batch size: {cfg.SOLVER.IMS_PER_BATCH}")
    print(f"Anchor sizes: {cfg.MODEL.ANCHOR_GENERATOR.SIZES}")
    print(f"Anchor ratios: {cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS}")
    print(f"Total training images: {total_images}")
    
    # Create trainer and train
    trainer = ImprovedMicroplasticTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    print(f"\nStarting improved training...")
    trainer.train()
    print(f"Training completed!")
    
    # Evaluate the improved model
    eval_results = evaluate_improved_model(cfg)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Faster R-CNN for Microplastic Detection")
    parser.add_argument("--model-variant", 
                        choices=["FasterRCNN-R50", "FasterRCNN-R101", "FasterRCNN-X101"], 
                        default="FasterRCNN-R50",
                        help="Model variant to train: FasterRCNN-R50 (Faster R-CNN with ResNet-50), FasterRCNN-R101 (Faster R-CNN with ResNet-101), or FasterRCNN-X101 (Faster R-CNN with ResNeXt-101)")
    
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Override the number of iterations (dry run; writes to <output>_dryrun<N>)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default 42; other seeds write to <output>_seed<N>)")
    parser.add_argument("--single-class", action="store_true",
                        help="Polymer-agnostic detector: one 'Microplastic' class (writes to <output>_agnostic)")
    args = parser.parse_args()
    main(max_iter=args.max_iter, model_variant=args.model_variant, seed=args.seed, single_class=args.single_class)