import argparse
import os

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .config import FEATURES_CSV, FOLDS, POLYMERS, RESULTS_DIR, fold_paths, particle_id

METHODS = [
    ("features_logistic_regression", "Features + Logistic regression"),
    ("features_knn", "Features + k-NN"),
    ("features_svm", "Features + SVM"),
    ("features_random_forest", "Features + Random forest"),
    ("features_gradient_boosting", "Features + Gradient boosting"),
    ("yolov8s-cls", "YOLOv8s-cls"),
    ("yolo26s-cls", "YOLO26s-cls"),
    ("hgnetv2_b4", "HGNetV2-B4"),
    ("resnet101", "ResNet-101"),
    ("convnext_tiny", "ConvNeXt-T"),
    ("dinov2_vits14", "DINOv2 ViT-S/14"),
]
RESOLUTION_TIERS_PX = [0, 5, 10, 15, 20, 30, 40, 60, 80]
LABELS = list(range(len(POLYMERS)))


def feret_px():
    f = pd.read_csv(FEATURES_CSV)
    f["particle_id"] = [particle_id(p, s, a) for p, s, a in zip(f["polymer"], f["source_image"], f["annotation_id"])]
    return (f["feret_max_um"] / f["um_per_px"]).set_axis(f["particle_id"])


def collect_predictions(folds=FOLDS):
    rows = []
    for fold in folds:
        paths = fold_paths(fold)
        split_path = os.path.join(paths["splits"], "balanced.csv")
        if not os.path.exists(split_path):
            continue
        test = pd.read_csv(split_path)
        test = test[test["split"] == "test"]
        images = pd.read_csv(os.path.join(paths["splits"], "image_split.csv")).set_index("source_image")
        for key, name in METHODS:
            path = os.path.join(paths["predictions"], f"{key}.csv")
            if os.path.exists(path):
                p = pd.read_csv(path).set_index("particle_id").loc[test["particle_id"]].reset_index()
                p["source_image"] = test["source_image"].to_numpy()
                p["magnification"] = images.loc[p["source_image"], "magnification"].to_numpy()
                p["fold"], p["method"] = fold, name
                rows.append(p)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["feret_px"] = out["particle_id"].map(feret_px()).to_numpy()
    return out


def scores(true, pred):
    p, r, f, n = precision_recall_fscore_support(true, pred, labels=LABELS, zero_division=0)
    return p, r, f, n


def metric_tables(preds):
    names = [n for _, n in METHODS if n in set(preds["method"])]
    metrics, folds, tiers = [], [], []
    for name in names:
        d = preds[preds["method"] == name]
        p, r, f, n = scores(d["true"], d["pred"])
        for i, c in enumerate(POLYMERS):
            metrics.append({"method": name, "class": c, "precision": 100 * p[i], "recall": 100 * r[i], "f1": 100 * f[i],
                            "test_particles": int(n[i])})
        metrics.append({"method": name, "class": "macro", "precision": 100 * p.mean(), "recall": 100 * r.mean(),
                        "f1": 100 * f.mean(), "test_particles": int(n.sum())})
        for fold, g in d.groupby("fold"):
            folds.append({"method": name, "fold": fold, "macro_f1": 100 * scores(g["true"], g["pred"])[2].mean()})
        for t in RESOLUTION_TIERS_PX:
            s = d[d["feret_px"] >= t]
            tiers.append({"method": name, "min_feret_px": t, "macro_f1": 100 * scores(s["true"], s["pred"])[2].mean(),
                          **{f"n_{c}": int((s["true"] == i).sum()) for i, c in enumerate(POLYMERS)}})
    per_image = (preds.assign(correct=preds["true"] == preds["pred"])
                 .pivot_table(index=["source_image", "magnification", "fold"], columns="method", values="correct", aggfunc="mean")
                 .mul(100).reindex(columns=names).reset_index())
    return pd.DataFrame(metrics), pd.DataFrame(folds), pd.DataFrame(tiers), per_image


def summary(metrics, folds):
    macro = metrics[metrics["class"] == "macro"].set_index("method")
    stats = folds.groupby("method")["macro_f1"].agg(["mean", "std"])
    lines = ["| Method | Precision | Recall | F1 | F1 per fold |", "|---|---|---|---|---|"]
    for name, r in macro.iterrows():
        lines.append(f"| {name} | {r['precision']:.1f} | {r['recall']:.1f} | {r['f1']:.1f} | "
                     f"{stats.loc[name, 'mean']:.1f} ± {stats.loc[name, 'std']:.1f} |")
    return "\n".join(lines)


def evaluate():
    preds = collect_predictions()
    if preds.empty:
        print("no predictions found")
        return None
    metrics, folds, tiers, per_image = metric_tables(preds)
    out = RESULTS_DIR
    os.makedirs(out, exist_ok=True)
    preds.to_csv(os.path.join(out, "predictions.csv"), index=False)
    metrics.to_csv(os.path.join(out, "metrics.csv"), index=False)
    folds.to_csv(os.path.join(out, "fold_f1.csv"), index=False)
    tiers.to_csv(os.path.join(out, "resolution.csv"), index=False)
    per_image.to_csv(os.path.join(out, "per_image_accuracy.csv"), index=False)
    table = summary(metrics, folds)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write(f"Macro-averaged scores (%), pooled over folds\n\n{table}\n")
    print(f"results written to {out}\n\n{table}\n")
    return metrics


def main():
    argparse.ArgumentParser(description="Score every trained method on the balanced test particles of all folds.").parse_args()
    evaluate()


if __name__ == "__main__":
    main()
