#!/usr/bin/env python3
"""
YOLOv8n-seg training script for the MNP dataset.
Trains on the patch split (single class MNP), then scores best.pt on the test split.

Run from this folder:
    python train.py [--epochs N]
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def count_test_images():
    """Number of images in the test split listed in data.yaml."""
    import yaml
    with open('data.yaml') as f:
        cfg = yaml.safe_load(f)
    return len(os.listdir(os.path.join(cfg['path'], cfg['test'])))


def save_test_metrics(model, save_dir, imgsz):
    """Score the model on the test split (default conf=0.001 so mAP sees the whole PR curve)."""
    metrics = model.val(data='data.yaml', split='test', imgsz=imgsz, batch=8, device='0',
                        project=str(save_dir), name='test_eval', exist_ok=True, plots=True)
    summary = lambda m: {"map": float(m.map), "map50": float(m.map50), "map75": float(m.map75),
                         "mp": float(m.mp), "mr": float(m.mr)}
    out = {
        "split": "test",
        "weights": str(save_dir / "weights" / "best.pt"),
        "num_images": count_test_images(),
        "box": summary(metrics.box),
        "mask": summary(metrics.seg),
    }
    path = save_dir / "test_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Test metrics saved to {path}")
    for k in ("box", "mask"):
        print(f"   {k:4s} mAP50-95 {out[k]['map']:.3f} | mAP50 {out[k]['map50']:.3f} | mAP75 {out[k]['map75']:.3f}")
    return out


def main(epochs, seed=0):
    from ultralytics import YOLO

    project_root = os.getcwd()
    print(f"Project root: {project_root}")

    model = YOLO('yolov8n-seg.pt')
    print("✓ yolov8n-seg.pt loaded")

    imgsz = 640
    results = model.train(
        data='data.yaml',              # Your dataset config
        epochs=epochs,                 # Number of epochs
        imgsz=imgsz,                   # Image size (256 patches are upsampled)
        batch=8,
        device='0',
        project=f'{project_root}/runs/segment',
        name='yolov8n-seg_patches' + (f'_seed{seed}' if seed else ''),
        seed=seed,
        patience=10,                   # Early stopping patience
        single_cls=True,               # Single class (MNP)
        plots=True,
        save=True,
        verbose=True
    )

    save_dir = results.save_dir if hasattr(results, "save_dir") else model.trainer.save_dir
    print(f"\n🎉 Training completed! Results saved to: {save_dir}")
    return save_test_metrics(YOLO(str(save_dir / "weights" / "best.pt")), save_dir, imgsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a single-class microplastic segmentation model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (use 1 for a dry run)")
    args = parser.parse_args()
    main(args.epochs, args.seed)
