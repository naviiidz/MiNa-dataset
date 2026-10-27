import os

import numpy as np
import pandas as pd

from .config import (DRAW_SEED, EXCLUDE_NAME_MISMATCH, EXCLUDED_IMAGE_PREFIXES, FEATURES_CSV, MIN_AREA_PX, N_FOLDS,
                     POLYMERS, SPLITS, fold_paths, parser, particle_id)


def load_particles():
    df = pd.read_csv(FEATURES_CSV)
    df = df[(df["area_px"] >= MIN_AREA_PX) & df["polymer"].isin(POLYMERS)].copy()
    if EXCLUDE_NAME_MISMATCH:
        df = df[df["name_polymer_mismatch"] == 0]
    for prefix in EXCLUDED_IMAGE_PREFIXES:
        df = df[~df["source_image"].str.startswith(prefix)]
    df["particle_id"] = [particle_id(p, s, a) for p, s, a in zip(df["polymer"], df["source_image"], df["annotation_id"])]
    return df


def image_folds(df, fold):
    rows = []
    for polymer, g in df.groupby("polymer"):
        images = g.groupby("source_image").agg(magnification=("magnification", "first"), particles=("area_px", "size"))
        images = images.sort_values(["magnification", "particles"]).reset_index()
        for i, r in images.iterrows():
            f = i % N_FOLDS
            split = "test" if f == fold else "val" if f == (fold + 1) % N_FOLDS else "train"
            rows.append({"polymer": polymer, "source_image": r["source_image"], "magnification": r["magnification"],
                         "particles": r["particles"], "fold": f, "split": split})
    return pd.DataFrame(rows)


def make_split(fold):
    out = fold_paths(fold)["splits"]
    os.makedirs(out, exist_ok=True)
    df = load_particles()
    images = image_folds(df, fold)
    images.to_csv(os.path.join(out, "image_split.csv"), index=False)
    df = df.drop(columns="split").merge(images[["polymer", "source_image", "split"]], on=["polymer", "source_image"])
    df["label"] = df["polymer"].map({p: i for i, p in enumerate(POLYMERS)})
    n_per_class = df.groupby(["split", "polymer"]).size().unstack()[POLYMERS].loc[SPLITS].min(axis=1)

    rng = np.random.default_rng(DRAW_SEED)
    picked = []
    for split in SPLITS:
        for polymer in POLYMERS:
            g = df[(df["split"] == split) & (df["polymer"] == polymer)]
            picked.append(g.loc[rng.choice(g.index.to_numpy(), size=int(n_per_class[split]), replace=False)])
    balanced = pd.concat(picked)
    balanced[["particle_id", "polymer", "label", "split", "source_image", "crop", "mask"]].to_csv(
        os.path.join(out, "balanced.csv"), index=False)
    return {s: int(n_per_class[s]) for s in SPLITS}


def main():
    args = parser("Split source images into cross-validation folds and draw class-balanced particle sets.").parse_args()
    for fold in args.folds:
        sizes = make_split(fold)
        print(f"fold {fold}: particles per polymer {sizes}")


if __name__ == "__main__":
    main()
