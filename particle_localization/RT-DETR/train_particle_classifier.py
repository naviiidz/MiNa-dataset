#!/usr/bin/env python3
"""
Train a polymer classifier on single-particle masked crops (stage 2 of the two-stage pipeline).

Crops come from ../../data_processing/make_particle_crops.py (one particle per crop, fixed grey
background). Trained on train, best epoch picked on val, then scored on test:
top-1 accuracy, per-class precision/recall and the confusion matrix.

Run from this folder:
    python train_particle_classifier.py [--model yolo11s-cls] [--epochs 60]
"""

import argparse
import json
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np

CLASSES = ["PE", "PET", "PP", "PS"]  # ultralytics sorts ImageFolder class names


def evaluate(model, data_dir, split, imgsz):
    """Per-class precision/recall and confusion matrix on one split (rows = true class)."""
    names = model.names
    conf = np.zeros((len(names), len(names)), dtype=int)
    idx = {v: k for k, v in names.items()}
    for cls in sorted(os.listdir(os.path.join(data_dir, split))):
        files = sorted(os.listdir(os.path.join(data_dir, split, cls)))
        for i in range(0, len(files), 256):
            batch = [os.path.join(data_dir, split, cls, f) for f in files[i:i + 256]]
            for r in model.predict(batch, imgsz=imgsz, device='0', verbose=False):
                conf[idx[cls], int(r.probs.top1)] += 1
    order = [names[i] for i in range(len(names))]
    per_class = {}
    for i, c in enumerate(order):
        tp = conf[i, i]
        per_class[c] = {"n": int(conf[i].sum()),
                        "precision": float(tp / conf[:, i].sum()) if conf[:, i].sum() else 0.0,
                        "recall": float(tp / conf[i].sum()) if conf[i].sum() else 0.0}
    return {"split": split, "accuracy": float(np.trace(conf) / conf.sum()),
            "balanced_accuracy": float(np.mean([v["recall"] for v in per_class.values()])),
            "classes": order, "confusion_matrix": conf.tolist(), "per_class": per_class}


def main(model_name, epochs, data_dir, imgsz):
    from ultralytics import YOLO
    model = YOLO(f"{model_name}.pt")
    results = model.train(
        data=data_dir, epochs=epochs, imgsz=imgsz, batch=128, device='0',
        project=f"{os.getcwd()}/runs/particle_classifier", name=model_name,
        patience=15, flipud=0.5, fliplr=0.5, hsv_h=0.0, hsv_s=0.0,  # SEM: no orientation, grey images
        plots=True, verbose=True,
    )
    save_dir = results.save_dir if hasattr(results, "save_dir") else model.trainer.save_dir
    best = YOLO(str(save_dir / "weights" / "best.pt"))
    out = {"weights": str(save_dir / "weights" / "best.pt"), "data": data_dir, "imgsz": imgsz,
           "val": evaluate(best, data_dir, "val", imgsz), "test": evaluate(best, data_dir, "test", imgsz)}
    with open(save_dir / "classifier_metrics.json", "w") as f:
        json.dump(out, f, indent=2)
    for split in ("val", "test"):
        m = out[split]
        print(f"{split}: accuracy {100 * m['accuracy']:.1f} | balanced accuracy {100 * m['balanced_accuracy']:.1f} | "
              + " ".join(f"{c} P{100 * v['precision']:.0f}/R{100 * v['recall']:.0f}" for c, v in m["per_class"].items()))
    print(f"Metrics saved to {save_dir / 'classifier_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the single-particle polymer classifier")
    parser.add_argument("--model", default="yolo11s-cls")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--data", default="../../data/particle_crops")
    parser.add_argument("--imgsz", type=int, default=64)
    args = parser.parse_args()
    main(args.model, args.epochs, args.data, args.imgsz)
