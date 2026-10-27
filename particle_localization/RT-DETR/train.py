#!/usr/bin/env python3
"""
RT-DETR training script for the MNP dataset (4 classes: PP, PE, PS, PET).
Trains RT-DETR, then scores best.pt on the test split (4-class and class-agnostic, see evaluate.py).

Run from this folder:
    python train.py --name rtdetr-l_pe-x3_p30 --patience 30
    python train.py --data data_gt_masked.yaml --name rtdetr-l_all-masked_pe-x3_p30 --patience 30
"""

import argparse
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from evaluate import save_test_metrics


def train(model_name, epochs, data='data.yaml', patience=30, name=None, seed=0, single_cls=False):
    """Train RT-DETR on the patch split and evaluate it on the test split."""
    from ultralytics import RTDETR

    print(f"🎯 Training {model_name} for microplastic detection")
    print("=" * 55)
    model = RTDETR(f'{model_name}.pt')

    imgsz = 640
    results = model.train(
        data=data,                     # Dataset config
        epochs=epochs,
        imgsz=imgsz,                   # Image size (256 patches are upsampled)
        batch=8,
        device='0',
        project=f'{os.getcwd()}/runs/{model_name}',
        name=name or f'mnp_{model_name}',
        patience=patience,             # Early stopping patience
        seed=seed,
        single_cls=single_cls,         # False: 4 polymer classes; True: polymer-agnostic
        plots=True,
        save=True,
        verbose=True
    )

    save_dir = results.save_dir if hasattr(results, "save_dir") else model.trainer.save_dir
    print(f"\n🎉 Training completed! Results saved to: {save_dir}")
    return save_test_metrics(RTDETR(str(save_dir / "weights" / "best.pt")), save_dir, imgsz, data, single_cls)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a 4-class RT-DETR microplastic detector")
    parser.add_argument("--model", default="rtdetr-l", help="rtdetr-l or rtdetr-x")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (use 1 for a dry run)")
    parser.add_argument("--data", default="data.yaml", help="Dataset yaml (default: data.yaml)")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience in epochs")
    parser.add_argument("--name", default=None, help="Run name under runs/<model>/ (default: mnp_<model>)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default 0)")
    parser.add_argument("--single_cls", action="store_true", help="Polymer-agnostic detector (one class)")
    args = parser.parse_args()
    train(args.model, args.epochs, args.data, args.patience, args.name, args.seed, args.single_cls)
