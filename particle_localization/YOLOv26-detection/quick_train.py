#!/usr/bin/env python3
"""
Detection training script for the MNP dataset (4 classes: PP, PE, PS, PET).
Trains an ultralytics YOLO detector, then scores best.pt on the test split.
RT-DETR has its own folder: ../RT-DETR

Run from this folder:
    python quick_train.py --model yolo26s
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def count_test_images(data):
    """Number of images in the test split listed in the dataset yaml."""
    import yaml
    with open(data) as f:
        cfg = yaml.safe_load(f)
    return len(os.listdir(os.path.join(cfg['path'], cfg['test'])))


def save_test_metrics(model, save_dir, imgsz, data='data.yaml', single_cls=False):
    """Score the model on the test split (default conf=0.001 so mAP sees the whole PR curve)."""
    metrics = model.val(data=data, split='test', imgsz=imgsz, batch=8, device='0', single_cls=single_cls,
                        project=str(save_dir), name='test_eval', exist_ok=True, plots=True)
    names = ['Microplastic'] if single_cls else ['PP', 'PE', 'PS', 'PET']
    out = {
        "split": "test",
        "weights": str(save_dir / "weights" / "best.pt"),
        "data": data,
        "num_images": count_test_images(data),
        "single_cls": single_cls,
        "box": {"map": float(metrics.box.map), "map50": float(metrics.box.map50),
                "map75": float(metrics.box.map75), "mp": float(metrics.box.mp), "mr": float(metrics.box.mr)},
        "per_class_map": {names[int(c)]: float(m) for c, m in zip(metrics.box.ap_class_index, metrics.box.maps[metrics.box.ap_class_index])},
    }
    path = save_dir / "test_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Test metrics saved to {path}")
    print(f"   mAP50-95 {out['box']['map']:.3f} | mAP50 {out['box']['map50']:.3f} | mAP75 {out['box']['map75']:.3f}")
    return out


def quick_train(model_name, epochs, data='data.yaml', patience=10, name=None, seed=0, single_cls=False):
    """Train a detector on the patch split and evaluate it on the test split."""
    from ultralytics import YOLO, RTDETR

    print(f"🎯 Training {model_name} for microplastic detection")
    print("=" * 55)
    model = (RTDETR if model_name.startswith("rtdetr") else YOLO)(f'{model_name}.pt')
    print(f"✓ {model_name} model loaded")

    project_root = os.getcwd()
    print(f"Project root: {project_root}")

    imgsz = 640
    results = model.train(
        data=data,                     # Dataset config
        epochs=epochs,                 # Number of epochs
        imgsz=imgsz,                   # Image size (256 patches are upsampled)
        batch=8,
        device='0',
        project=f'{project_root}/runs/{model_name}',  # v4 runs are kept apart from older runs
        name=name or f'mnp_{model_name}',  # Experiment name
        patience=patience,             # Early stopping patience
        seed=seed,
        single_cls=single_cls,         # False: 4 polymer classes; True: polymer-agnostic
        plots=True,
        save=True,
        verbose=True
    )

    save_dir = results.save_dir if hasattr(results, "save_dir") else model.trainer.save_dir
    print(f"\n🎉 Training completed! Results saved to: {save_dir}")

    best = (RTDETR if model_name.startswith("rtdetr") else YOLO)(str(save_dir / "weights" / "best.pt"))
    return save_test_metrics(best, save_dir, imgsz, data, single_cls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 4-class microplastic detector")
    parser.add_argument("--model", default="yolo26s", help="Ultralytics model name, e.g. yolo26s (RT-DETR: use ../RT-DETR/train.py)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (use 1 for a dry run)")
    parser.add_argument("--data", default="data.yaml", help="Dataset yaml (default: data.yaml)")
    parser.add_argument("--patience", type=int, default=10, help="Early-stopping patience in epochs")
    parser.add_argument("--name", default=None, help="Run name under runs/<model>/ (default: mnp_<model>)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    parser.add_argument("--single_cls", action="store_true", help="Polymer-agnostic detector (one class)")
    args = parser.parse_args()
    quick_train(args.model, args.epochs, args.data, args.patience, args.name, args.seed, args.single_cls)
