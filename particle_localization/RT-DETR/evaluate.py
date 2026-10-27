#!/usr/bin/env python3
"""
Score a trained RT-DETR run on the test split, 4-class and class-agnostic.

4-class:        the normal benchmark (PP, PE, PS, PET).
class-agnostic: the same predictions scored with single_cls=True, i.e. "was the particle
                found", whatever polymer was predicted. The gap between the two is the
                cost of polymer confusion.

Writes <run>/test_metrics.json (conf=0.001, so mAP sees the whole PR curve).

Run from this folder:
    python evaluate.py --run runs/rtdetr-l/rtdetr-l_pe-x3_p30 [--data data.yaml]
"""

import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

NAMES = ['PP', 'PE', 'PS', 'PET']


def count_test_images(data):
    """Number of images in the test split listed in the dataset yaml."""
    import yaml
    with open(data) as f:
        cfg = yaml.safe_load(f)
    return len(os.listdir(os.path.join(cfg['path'], cfg['test'])))


def box_summary(box):
    return {"map": float(box.map), "map50": float(box.map50), "map75": float(box.map75),
            "mp": float(box.mp), "mr": float(box.mr)}


def save_test_metrics(model, save_dir, imgsz=640, data='data.yaml', single_cls=False):
    """Score best.pt on the test split, 4-class and class-agnostic; save test_metrics.json.
    A polymer-agnostic model (single_cls=True) is scored class-agnostic only."""
    save_dir = Path(save_dir).resolve()  # absolute, or ultralytics nests it under runs/detect
    if single_cls:
        m1 = model.val(data=data, split='test', imgsz=imgsz, batch=8, device='0', single_cls=True,
                       project=str(save_dir), name='test_eval_agnostic', exist_ok=True, plots=True)
        out = {"split": "test", "weights": str(save_dir / "weights" / "best.pt"), "data": data,
               "num_images": count_test_images(data), "single_cls": True, "class_agnostic": box_summary(m1.box)}
        with open(save_dir / "test_metrics.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"Test metrics saved to {save_dir / 'test_metrics.json'}")
        print(f"   agnostic mAP50 {out['class_agnostic']['map50']:.3f} | mAP50-95 {out['class_agnostic']['map']:.3f}")
        return out
    m4 = model.val(data=data, split='test', imgsz=imgsz, batch=8, device='0',
                   project=str(save_dir), name='test_eval', exist_ok=True, plots=True)
    m1 = model.val(data=data, split='test', imgsz=imgsz, batch=8, device='0', single_cls=True,
                   project=str(save_dir), name='test_eval_agnostic', exist_ok=True, plots=True)
    idx = m4.box.ap_class_index
    out = {
        "split": "test",
        "weights": str(save_dir / "weights" / "best.pt"),
        "data": data,
        "num_images": count_test_images(data),
        "box": box_summary(m4.box),
        "per_class_map": {NAMES[int(c)]: float(v) for c, v in zip(idx, m4.box.maps[idx])},
        "per_class_ap50": {NAMES[int(c)]: float(v) for c, v in zip(idx, m4.box.ap50)},
        "class_agnostic": box_summary(m1.box),
    }
    path = save_dir / "test_metrics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Test metrics saved to {path}")
    print(f"   4-class  mAP50 {out['box']['map50']:.3f} | mAP50-95 {out['box']['map']:.3f} | "
          + " ".join(f"{k} {v:.3f}" for k, v in out["per_class_ap50"].items()))
    print(f"   agnostic mAP50 {out['class_agnostic']['map50']:.3f} | mAP50-95 {out['class_agnostic']['map']:.3f}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score an RT-DETR run on the test split")
    parser.add_argument("--run", required=True, help="Run folder, e.g. runs/rtdetr-l/rtdetr-l_pe-x3_p30")
    parser.add_argument("--data", default="data.yaml", help="Dataset yaml the run was trained on")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--single_cls", action="store_true", help="Run was trained polymer-agnostic")
    args = parser.parse_args()
    from ultralytics import RTDETR
    save_test_metrics(RTDETR(os.path.join(args.run, "weights", "best.pt")), args.run, args.imgsz, args.data, args.single_cls)
