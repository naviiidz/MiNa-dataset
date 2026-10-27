#!/usr/bin/env python3
"""
YOLOv26 Model Evaluation Script
Evaluates the trained model on test set with detailed metrics and visualizations.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pandas as pd

def find_best_weights(model_name):
    """Latest best.pt written by quick_train.py --model <model_name> (runs/<model>/mnp_<model>*)."""
    from pathlib import Path
    runs = sorted(Path("runs", model_name).glob(f"mnp_{model_name}*/weights/best.pt"),
                  key=lambda p: p.stat().st_mtime)
    return str(runs[-1]) if runs else None


def load_model(path):
    """RT-DETR checkpoints need the RTDETR class; everything else loads with YOLO."""
    from ultralytics import YOLO, RTDETR
    return (RTDETR if "rtdetr" in path else YOLO)(path)


def evaluate_model(argparse_model='yolo26s'):
    """Comprehensive evaluation of the trained YOLOv26 model on test set."""
    from ultralytics import YOLO
    
    print("🔬 YOLOv26 Model Evaluation on Test Set")
    print("=" * 45)
    
    # Find the best trained model
    project_root = os.getcwd()
    
    # Look for the latest training run of this model
    found = find_best_weights(argparse_model)
    if found:
        best_model_path = Path(found)
        print(f"📁 Found trained model: {best_model_path}")
    else:
        print(f"❌ No trained model found in runs/{argparse_model}. Please run training first.")
        return
    
    if not best_model_path.exists():
        print(f"❌ Best model not found at {best_model_path}")
        return
    
    # Load the trained model
    model = load_model(str(best_model_path))
    print("✅ Model loaded successfully")
    
    # Print model info
    model.info(detailed=False, verbose=False)
    
    print("\n🚀 Running evaluation on test set...")
    
    # Run validation on test set
    test_results = model.val(
        data='data.yaml',
        split='test',  # Use test split instead of val
        imgsz=640,
        batch=8,
        device='0',
        project=f'{project_root}/runs/test',
        name='test_evaluation',
        save_json=True,  # Save results in JSON format
        save_txt=True,   # Save results in txt format
        plots=True,      # Generate evaluation plots
        verbose=True
    )
    
    print(f"\n📊 Test Results Summary:")
    print(f"{'='*50}")
    
    # Overall metrics
    if hasattr(test_results, 'results_dict'):
        results_dict = test_results.results_dict
        print(f"📈 Overall Performance:")
        print(f"   mAP@0.5    : {results_dict.get('metrics/mAP50(B)', 0):.3f}")
        print(f"   mAP@0.5:0.95: {results_dict.get('metrics/mAP50-95(B)', 0):.3f}")
        print(f"   Precision  : {results_dict.get('metrics/precision(B)', 0):.3f}")
        print(f"   Recall     : {results_dict.get('metrics/recall(B)', 0):.3f}")
    
    # Class-wise performance
    print(f"\n🎯 Class-wise Performance:")
    class_names = ['PP', 'PE', 'PS', 'PET']
    
    if hasattr(test_results, 'ap_class_index') and test_results.ap_class_index is not None:
        for i, class_name in enumerate(class_names):
            if i < len(test_results.ap):
                print(f"   {class_name:3s}: mAP@0.5 = {test_results.ap[i]:.3f}, mAP@0.5:0.95 = {test_results.ap50_95[i]:.3f}")
    
    # Speed metrics
    if hasattr(test_results, 'speed'):
        speed = test_results.speed
        print(f"\n⚡ Speed Performance:")
        print(f"   Preprocessing: {speed.get('preprocess', 0):.1f} ms/image")
        print(f"   Inference    : {speed.get('inference', 0):.1f} ms/image")
        print(f"   Postprocess  : {speed.get('postprocess', 0):.1f} ms/image")
        print(f"   Total        : {sum(speed.values()):.1f} ms/image")
    
    # Generate additional analysis
    print(f"\n📈 Generating detailed analysis...")
    
    # Create confusion matrix analysis if available
    results_dir = Path(project_root) / "runs" / "test" / "test_evaluation"
    if results_dir.exists():
        print(f"📁 Results saved to: {results_dir}")
        
        # Look for generated plots
        plots_found = []
        plot_files = ['confusion_matrix.png', 'confusion_matrix_normalized.png', 
                     'F1_curve.png', 'P_curve.png', 'R_curve.png', 'PR_curve.png']
        
        for plot_file in plot_files:
            if (results_dir / plot_file).exists():
                plots_found.append(plot_file)
        
        if plots_found:
            print(f"📊 Generated plots: {', '.join(plots_found)}")
    
    # Performance recommendations
    print(f"\n💡 Performance Analysis:")
    
    if hasattr(test_results, 'results_dict'):
        map50 = results_dict.get('metrics/mAP50(B)', 0)
        map50_95 = results_dict.get('metrics/mAP50-95(B)', 0)
        
        if map50 > 0.7:
            print("   ✅ Excellent detection performance!")
        elif map50 > 0.5:
            print("   ✅ Good detection performance")
            print("   💭 Consider fine-tuning for better results")
        else:
            print("   ⚠️  Detection performance could be improved")
            print("   💭 Consider: more training data, data augmentation, or hyperparameter tuning")
        
        if map50_95 > 0.4:
            print("   ✅ Good localization accuracy")
        else:
            print("   💭 Localization could be improved - check bounding box quality")
    
    print(f"\n🎯 Use this model for inference:")
    print(f"   model = YOLO('{best_model_path}')")
    print(f"   results = model.predict('path/to/image.jpg')")
    
    return test_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s", help="model name used with quick_train.py --model")
    evaluate_model(argparse_model=parser.parse_args().model)
