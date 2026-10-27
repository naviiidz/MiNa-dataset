#!/usr/bin/env python3
"""
Test Set Evaluation for Trained YOLOv8s Model
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

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


def evaluate_on_test(argparse_model='yolo26s'):
    """Evaluate the trained YOLOv8s model on test set."""
    from ultralytics import YOLO
    import os
    import glob
    
    print("🔬 YOLOv8s Test Set Evaluation")
    print("=" * 35)
    
    # Find the latest trained model
    model_name = argparse_model
    runs = f"runs/{model_name}"
    model_path = find_best_weights(model_name)
    
    if not model_path:
        print(f"❌ No trained model found in {runs}")
        print("Available runs:")
        if os.path.exists(runs):
            for run in os.listdir(runs):
                weights_path = os.path.join(runs, run, "weights")
                if os.path.exists(weights_path):
                    print(f"   - {run} (weights available)")
                else:
                    print(f"   - {run} (no weights)")
        return
    
    print(f"📁 Loading model: {model_path}")
    model = load_model(model_path)
    
    print("🚀 Evaluating on test set...")
    
    # Run evaluation on test set
    results = model.val(
        data='data.yaml',
        split='test',  # Use test split
        imgsz=640,
        batch=4,       # Smaller batch to avoid memory issues
        device='0',
        save=True,
        plots=True,
        project=runs,
        name='test_eval',
        exist_ok=True
    )
    
    print("\n📊 Test Results:")
    print(f"   mAP@0.5    : {results.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {results.box.map:.3f}")
    print(f"   Precision  : {results.box.mp:.3f}")
    print(f"   Recall     : {results.box.mr:.3f}")
    
    # Class-wise results
    print(f"\n🎯 Per-class Results:")
    class_names = ['PP', 'PE', 'PS', 'PET']
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        for i, name in enumerate(class_names):
            if i < len(results.box.ap50):
                print(f"   {name}: mAP@0.5 = {results.box.ap50[i]:.3f}")
    
    # Model info
    print(f"\n🤖 Model Information:")
    print(f"   Model used: {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    print(f"\n✅ Evaluation completed!")
    print(f"📁 Results saved in: {runs}/test_eval/")
    
    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo26s", help="model name used with quick_train.py --model")
    evaluate_on_test(argparse_model=parser.parse_args().model)
