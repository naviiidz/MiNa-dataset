import argparse
import os

from config import BATCH, EPOCHS, IMGSZ, PATCH_SIZES, PATIENCE, PRETRAINED, SEEDS, size_paths


def train(size, seed, epochs=EPOCHS):
    from ultralytics import YOLO
    paths = size_paths(size, seed)
    model = YOLO(PRETRAINED)
    model.train(data=os.path.join(paths["yolo"], "data.yaml"), epochs=epochs, patience=PATIENCE, imgsz=IMGSZ,
                batch=BATCH, device=0, seed=seed, deterministic=True, project=paths["runs"], name=paths["run_name"], exist_ok=True,
                plots=True, verbose=False)
    print(f"patch size {size} seed {seed}: weights -> {paths['weights']}")


def main():
    p = argparse.ArgumentParser(description="Train YOLO26s on the patch dataset of each size (identical settings).")
    p.add_argument("--sizes", type=int, nargs="+", default=PATCH_SIZES)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    args = p.parse_args()
    for seed in args.seeds:
        for size in args.sizes:
            train(size, seed, args.epochs)


if __name__ == "__main__":
    main()
