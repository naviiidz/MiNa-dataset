import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import math
import textwrap
import warnings

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from . import classifiers
from .config import (APPEARANCE_Z, FEATURES_CSV, FEATURES_DIR, FOLDS, IMAGE_SIZE, NORM_CENTER, NORM_SCALE, POLYMERS,
                     RESULTS_DIR, SHAPE, SIZE, fold_paths, particle_id)
from .images import particle_mask_on_canvas

warnings.filterwarnings("ignore")

LABELS = list(range(len(POLYMERS)))
DESCRIPTORS = SIZE + SHAPE + APPEARANCE_Z
CONDITIONS = ["original", "silhouette", "texture_shuffle", "blur_1", "blur_2", "blur_4", "inscribed_disc"]
CONDITION_NAMES = {"original": "Original", "silhouette": "Silhouette (texture removed)",
                   "texture_shuffle": "Texture shuffled inside outline", "blur_1": "Blur σ = 1 px",
                   "blur_2": "Blur σ = 2 px", "blur_4": "Blur σ = 4 px",
                   "inscribed_disc": "Inscribed disc (outline removed, texture kept)"}
MODEL_NAMES = {"dinov2_vits14": "DINOv2 ViT-S/14", "convnext_tiny": "ConvNeXt-T"}


def ablate(img, m, condition, rng):
    out = img.copy()
    inside = m > 0
    if condition == "silhouette":
        out[inside] = NORM_CENTER + NORM_SCALE
    elif condition == "texture_shuffle":
        out[inside] = rng.permutation(img[inside])
    elif condition.startswith("blur_"):
        sigma = float(condition.split("_")[1])
        blurred = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma)
        out[inside] = np.clip(blurred[inside], 0, 255).astype(np.uint8)
    elif condition == "inscribed_disc":
        dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
        cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
        r = max(1.0, float(dist.max()))
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        out[:] = NORM_CENTER
        out[disc] = img[disc]
    return out


def load_fold(fold, feats):
    dirs = fold_paths(fold)
    index = pd.read_csv(os.path.join(dirs["images"], "index.csv"))
    images = {s: np.stack([cv2.imread(os.path.join(dirs["images"], f), cv2.IMREAD_GRAYSCALE) for f in g["image"]])
              for s, g in index.groupby("split")}
    masks = {s: np.stack([particle_mask_on_canvas(cv2.imread(os.path.join(FEATURES_DIR, feats.loc[p, "mask"]), cv2.IMREAD_GRAYSCALE))
                          for p in g["particle_id"]]) for s, g in index.groupby("split")}
    parts = {s: g.reset_index(drop=True) for s, g in index.groupby("split")}
    return dirs, parts, images, masks


def probs(model, mean, std, arrays):
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays[:, None]), torch.zeros(len(arrays), dtype=torch.long)),
                        batch_size=128)
    return classifiers.predict(model, loader, mean, std, torch.device("cuda"), tta=True)


def gradcam(model, model_name, mean, std, arrays, targets):
    acts, grads = {}, {}
    layer = model.stages[-2] if model_name == "convnext_tiny" else model.blocks[-2]
    h1 = layer.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda mod, gi, go: grads.__setitem__("g", go[0]))
    cams = []
    model.eval()
    for i in range(0, len(arrays), 32):
        x = classifiers.prepare(torch.from_numpy(arrays[i:i + 32, None]).cuda(), mean, std).requires_grad_(True)
        out = classifiers.logits_of(model(x))
        model.zero_grad()
        out.gather(1, torch.as_tensor(targets[i:i + 32], device="cuda")[:, None]).sum().backward()
        a, g = acts["a"].detach(), grads["g"].detach()
        if a.dim() == 3:
            n = int(math.sqrt(a.shape[1] - 1))
            a = a[:, -n * n:].transpose(1, 2).reshape(a.shape[0], -1, n, n)
            g = g[:, -n * n:].transpose(1, 2).reshape(g.shape[0], -1, n, n)
        cam = torch.relu((g.mean(dim=(2, 3), keepdim=True) * a).sum(1))
        cam = torch.nn.functional.interpolate(cam[:, None], size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)[:, 0]
        cams.append(cam.cpu().numpy())
    h1.remove()
    h2.remove()
    return np.concatenate(cams)


def cam_shares(cams, masks):
    rows = []
    kernel = np.ones((9, 9), np.uint8)
    for cam, m in zip(cams, masks):
        inner = cv2.erode(m, kernel) > 0
        outer = cv2.dilate(m, kernel) > 0
        boundary = outer & ~inner
        mean = cam.mean() + 1e-9
        rows.append({"interior": cam[inner].mean() / mean if inner.any() else np.nan,
                     "boundary": cam[boundary].mean() / mean, "background": cam[~outer].mean() / mean})
    return pd.DataFrame(rows)


def overlay(img, cam, m):
    cam = cam / (cam.max() + 1e-9)
    heat = cv2.applyColorMap((255 * cam).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 1)
    return out


def gradcam_figure(model_name, imgs, msks, cams, yt, yp, out_dir, n_ok=4, n_bad=2):
    rng = np.random.default_rng(0)
    cols = n_ok + n_bad
    fig, axes = plt.subplots(len(POLYMERS), 2 * cols, figsize=(2 * cols * 1.25, len(POLYMERS) * 1.55 + 0.6))
    for r, c in enumerate(LABELS):
        ok = np.flatnonzero((yt == c) & (yp == c))
        bad = np.flatnonzero((yt == c) & (yp != c))
        pick = list(rng.choice(ok, min(n_ok, len(ok)), replace=False)) + list(rng.choice(bad, min(n_bad, len(bad)), replace=False))
        for j in range(cols):
            a_img, a_cam = axes[r, 2 * j], axes[r, 2 * j + 1]
            a_img.axis("off")
            a_cam.axis("off")
            if j < len(pick):
                i = pick[j]
                a_img.imshow(imgs[i], cmap="gray", vmin=0, vmax=255)
                a_cam.imshow(cv2.cvtColor(overlay(imgs[i], cams[i], msks[i]), cv2.COLOR_BGR2RGB))
                a_img.set_title(f"{POLYMERS[yt[i]]} → {POLYMERS[yp[i]]}", fontsize=8, loc="left",
                                color="green" if yt[i] == yp[i] else "red")
    for j in range(cols):
        axes[0, 2 * j + 1].text(0.5, 1.28, "Grad-CAM", fontsize=7, ha="center", transform=axes[0, 2 * j + 1].transAxes)
        axes[0, 2 * j].text(0.5, 1.28, "Original", fontsize=7, ha="center", transform=axes[0, 2 * j].transAxes)
    fig.suptitle(f"{MODEL_NAMES[model_name]}: test particles (fold 0) and Grad-CAM; "
                 f"{n_ok} correct, {n_bad} wrong per polymer (true → predicted)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(out_dir, f"gradcam_{model_name}.png"), dpi=200)
    plt.close(fig)


def figures_only(models, feats, out_dir):
    dirs, parts, images, masks = load_fold(0, feats)
    y = parts["test"]["label"].to_numpy()
    for model_name in models:
        model, mean, std = classifiers.load_trained(model_name, 0)
        test_pred = probs(model, mean, std, images["test"]).argmax(1)
        cams = gradcam(model, model_name, mean, std, images["test"], test_pred)
        gradcam_figure(model_name, images["test"], masks["test"], cams, y, test_pred, out_dir)
        print(os.path.join(out_dir, f"gradcam_{model_name}.png"))


def macro_f1(y, p):
    return 100 * f1_score(y, p, labels=LABELS, average="macro", zero_division=0)


def main():
    parser = argparse.ArgumentParser(description="Explain trained classifiers: input ablations, Grad-CAM and a "
                                                 "descriptor surrogate model.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAMES), choices=list(MODEL_NAMES))
    parser.add_argument("--figures_only", action="store_true", help="Only redraw the Grad-CAM figures from fold 0")
    args = parser.parse_args()
    rng = np.random.default_rng(0)

    feats = pd.read_csv(FEATURES_CSV)
    feats["particle_id"] = [particle_id(p, s, a) for p, s, a in zip(feats["polymer"], feats["source_image"], feats["annotation_id"])]
    feats = feats.set_index("particle_id")
    feats[APPEARANCE_Z] = feats[APPEARANCE_Z].fillna(feats[APPEARANCE_Z].median())
    if args.figures_only:
        figures_only(args.models, feats, os.path.join(RESULTS_DIR, "interpretability"))
        return

    ablation_rows, share_rows, surrogate_rows, importance_rows, examples = [], [], [], [], {}
    for fold in FOLDS:
        dirs, parts, images, masks = load_fold(fold, feats)
        y = {s: parts[s]["label"].to_numpy() for s in parts}
        for model_name in args.models:
            model, mean, std = classifiers.load_trained(model_name, fold)

            for condition in CONDITIONS:
                arr = np.stack([ablate(i, m, condition, rng) for i, m in zip(images["test"], masks["test"])])
                pred = probs(model, mean, std, arr).argmax(1)
                ablation_rows += [{"model": model_name, "condition": condition, "fold": fold, "true": t, "pred": p}
                                  for t, p in zip(y["test"], pred)]

            test_pred = probs(model, mean, std, images["test"]).argmax(1)
            cams = gradcam(model, model_name, mean, std, images["test"], test_pred)
            shares = cam_shares(cams, masks["test"])
            shares["model"], shares["polymer"] = model_name, parts["test"]["polymer"].to_numpy()
            shares["correct"] = test_pred == y["test"]
            share_rows.append(shares)
            if fold == 0:
                examples[model_name] = (images["test"], masks["test"], cams, y["test"], test_pred)

            train_pred = probs(model, mean, std, images["train"]).argmax(1)
            Xtr = feats.loc[parts["train"]["particle_id"], DESCRIPTORS].to_numpy(dtype=np.float64)
            Xte = feats.loc[parts["test"]["particle_id"], DESCRIPTORS].to_numpy(dtype=np.float64)
            surrogate = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=8, random_state=0).fit(Xtr, train_pred)
            fidelity = f1_score(test_pred, surrogate.predict(Xte), labels=LABELS, average="macro", zero_division=0)
            surrogate_rows.append({"model": model_name, "fold": fold, "fidelity_macro_f1": 100 * fidelity,
                                   "agreement": 100 * np.mean(test_pred == surrogate.predict(Xte))})
            imp = permutation_importance(surrogate, Xte, test_pred, scoring="f1_macro", n_repeats=10, random_state=0, n_jobs=8)
            importance_rows += [{"model": model_name, "fold": fold, "feature": f, "importance": 100 * v}
                                for f, v in zip(DESCRIPTORS, imp.importances_mean)]
            print(f"fold {fold} {model_name}: original {macro_f1(y['test'], test_pred):.1f}, surrogate agreement "
                  f"{surrogate_rows[-1]['agreement']:.1f}%", flush=True)
            del model
            torch.cuda.empty_cache()

    abl = pd.DataFrame(ablation_rows)
    shares = pd.concat(share_rows)
    sur = pd.DataFrame(surrogate_rows)
    imp = pd.DataFrame(importance_rows)
    out_dir = os.path.join(RESULTS_DIR, "interpretability")
    os.makedirs(out_dir, exist_ok=True)
    abl.to_csv(os.path.join(out_dir, "ablation_predictions.csv"), index=False)
    shares.to_csv(os.path.join(out_dir, "gradcam_attention.csv"), index=False)
    imp.to_csv(os.path.join(out_dir, "surrogate_importance.csv"), index=False)

    lines = ["## Interpretability", "",
             "### Input ablations: macro F1 (%) on altered test images", "",
             "| Test input | " + " | ".join(MODEL_NAMES[m] for m in args.models) + " |", "|---|" + "---|" * len(args.models)]
    for c in CONDITIONS:
        cells = []
        for m in args.models:
            d = abl[(abl["model"] == m) & (abl["condition"] == c)]
            cells.append(f"{macro_f1(d['true'], d['pred']):.1f}")
        lines.append(f"| {CONDITION_NAMES[c]} | " + " | ".join(cells) + " |")

    lines += ["", "### Grad-CAM: attention per unit area, relative to the image average (1.0 = average)", "",
              "| Model | Polymer | Particle interior | Particle boundary (±4 px) | Background |", "|---|---|---|---|---|"]
    for m in args.models:
        for p in POLYMERS + ["all"]:
            d = shares[shares["model"] == m] if p == "all" else shares[(shares["model"] == m) & (shares["polymer"] == p)]
            lines.append(f"| {MODEL_NAMES[m]} | {p} | {d['interior'].mean():.2f} | {d['boundary'].mean():.2f} | "
                         f"{d['background'].mean():.2f} |")

    lines += ["", "### Surrogate model: random forest predicting the network's decisions from measured descriptors", "",
              "| Model | Agreement with network (%) | Macro F1 vs network predictions (%) |", "|---|---|---|"]
    for m in args.models:
        d = sur[sur["model"] == m]
        lines.append(f"| {MODEL_NAMES[m]} | {d['agreement'].mean():.1f} ± {d['agreement'].std(ddof=1):.1f} | "
                     f"{d['fidelity_macro_f1'].mean():.1f} ± {d['fidelity_macro_f1'].std(ddof=1):.1f} |")
    lines += ["", "Permutation importance of each descriptor for reproducing the network's decisions (drop in macro F1, "
              "points, mean over folds; top 10):", "",
              "| Rank | " + " | ".join(MODEL_NAMES[m] for m in args.models) + " |", "|---|" + "---|" * len(args.models)]
    tops = {m: imp[imp["model"] == m].groupby("feature")["importance"].mean().sort_values(ascending=False).head(10) for m in args.models}
    for k in range(10):
        lines.append(f"| {k + 1} | " + " | ".join(f"{tops[m].index[k]} ({tops[m].iloc[k]:.1f})" for m in args.models) + " |")
    with open(os.path.join(out_dir, "summary.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    for m, example in examples.items():
        gradcam_figure(m, *example, out_dir)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.38
    for k, m in enumerate(args.models):
        vals = [macro_f1(*abl[(abl["model"] == m) & (abl["condition"] == c)][["true", "pred"]].to_numpy().T) for c in CONDITIONS]
        ax.bar(np.arange(len(CONDITIONS)) + (k - 0.5) * w, vals, w, label=MODEL_NAMES[m])
    ax.axhline(25, color="k", ls=":", lw=0.8)
    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels([textwrap.fill(CONDITION_NAMES[c], 16) for c in CONDITIONS], fontsize=7)
    ax.set_ylabel("Macro F1 (%)")
    ax.set_title("Input ablations on test particles (chance = 25%)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablations.png"), dpi=200)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
