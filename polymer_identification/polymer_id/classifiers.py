import argparse
import json
import math
import os
import time
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .config import BATCH, CLASSIFIERS, EPOCHS, NORM_CENTER, POLYMERS, SEED, add_fold_args, fold_paths

warnings.filterwarnings("ignore")


class Particles(Dataset):
    def __init__(self, df, root, train):
        self.images = np.stack([cv2.imread(os.path.join(root, f), cv2.IMREAD_GRAYSCALE) for f in df["image"]])
        self.labels = df["label"].to_numpy()
        self.aug = T.Compose([
            T.RandomRotation(180, interpolation=T.InterpolationMode.BILINEAR, fill=NORM_CENTER),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.85, 1.15), fill=NORM_CENTER),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.3),
        ]) if train else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = torch.from_numpy(self.images[i])[None]
        if self.aug is not None:
            x = self.aug(x)
        return x, int(self.labels[i])


def build(name):
    source, ident, lr = CLASSIFIERS[name]
    if source == "ultralytics":
        from ultralytics import YOLO
        model = YOLO(ident).model.float()
        head = model.model[-1]
        head.linear = nn.Linear(head.linear.in_features, len(POLYMERS))
        for p in model.parameters():
            p.requires_grad_(True)
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        import timm
        kwargs = {"img_size": 224} if "dinov2" in ident else {}
        model = timm.create_model(ident, pretrained=True, num_classes=len(POLYMERS), **kwargs)
        cfg = timm.data.resolve_model_data_config(model)
        mean, std = cfg["mean"], cfg["std"]
    return model, torch.tensor(mean).view(1, 3, 1, 1), torch.tensor(std).view(1, 3, 1, 1), lr


def load_trained(name, fold, device="cuda"):
    model, mean, std, _ = build(name)
    weights = os.path.join(fold_paths(fold)["runs"], name, "best.pt")
    model.load_state_dict(torch.load(weights, map_location=device))
    return model.to(device), mean.to(device), std.to(device)


def logits_of(out):
    return out[1] if isinstance(out, tuple) else out


def prepare(x, mean, std):
    return ((x.float() / 255.0).repeat(1, 3, 1, 1) - mean) / std


def tta_views(x):
    for k in range(4):
        r = torch.rot90(x, k, dims=(2, 3))
        yield r
        yield torch.flip(r, dims=(3,))


@torch.no_grad()
def predict(model, loader, mean, std, device="cuda", tta=True):
    model.eval()
    probs = []
    for x, _ in loader:
        x = prepare(x.to(device), mean, std)
        views = tta_views(x) if tta else [x]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.stack([logits_of(model(v)).float().softmax(1) for v in views]).mean(0)
        probs.append(p.cpu().numpy())
    return np.concatenate(probs)


def train_fold(name, fold, epochs=EPOCHS, device="cuda"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    paths = fold_paths(fold)
    index = pd.read_csv(os.path.join(paths["images"], "index.csv"))
    parts = {s: g.reset_index(drop=True) for s, g in index.groupby("split")}
    data = {s: Particles(g, paths["images"], train=(s == "train")) for s, g in parts.items()}
    counts = np.bincount(data["train"].labels, minlength=len(POLYMERS))
    sampler = WeightedRandomSampler(torch.tensor(1.0 / counts[data["train"].labels], dtype=torch.double),
                                    num_samples=len(data["train"]), replacement=True,
                                    generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(data["train"], batch_size=BATCH, sampler=sampler, num_workers=8, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(data["val"], batch_size=256, num_workers=4)
    test_loader = DataLoader(data["test"], batch_size=128, num_workers=4)

    model, mean, std, lr = build(name)
    model, mean, std = model.to(device), mean.to(device), std.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps, warmup = epochs * len(train_loader), len(train_loader)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    out_dir = os.path.join(paths["runs"], name)
    os.makedirs(out_dir, exist_ok=True)
    best_f1, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in train_loader:
            x, y = prepare(x.to(device, non_blocking=True), mean, std), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(logits_of(model(x)).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        f1 = f1_score(data["val"].labels, predict(model, val_loader, mean, std, device, tta=False).argmax(1), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(train_loader), "val_macro_f1": f1})
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        print(f"{name} fold {fold} epoch {epoch}/{epochs} loss {total / len(train_loader):.3f} "
              f"val macro F1 {100 * f1:.1f} (best {100 * best_f1:.1f}) {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out_dir, "history.json"), "w") as fh:
        json.dump(history, fh, indent=2)

    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    probs = predict(model, test_loader, mean, std, device, tta=True)
    df = pd.DataFrame({"particle_id": parts["test"]["particle_id"], "true": data["test"].labels, "pred": probs.argmax(1)})
    for i, p in enumerate(POLYMERS):
        df[f"p_{p}"] = probs[:, i]
    os.makedirs(paths["predictions"], exist_ok=True)
    df.to_csv(os.path.join(paths["predictions"], f"{name}.csv"), index=False)
    print(f"{name} fold {fold} done: best val macro F1 {100 * best_f1:.1f}, "
          f"test macro F1 {100 * f1_score(df['true'], df['pred'], average='macro'):.1f}", flush=True)


def main():
    p = add_fold_args(argparse.ArgumentParser(description="Train a deep image classifier on the single-particle images "
                                                          "and predict the test particles of each fold."))
    p.add_argument("--model", nargs="+", required=True, choices=list(CLASSIFIERS), help="one or more classifiers")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs (default: %(default)s)")
    args = p.parse_args()
    for name in args.model:
        for fold in args.folds:
            train_fold(name, fold, args.epochs)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
