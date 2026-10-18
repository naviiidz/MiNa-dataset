import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import POLYMERS, RESULTS_DIR

GROUPS = [("Features + ", "Measured descriptors + ML", "#9e9e9e", "--"),
          ("", "Deep classifiers", "#2f6db5", "-")]


def group_of(name):
    return next(g for key, *g in GROUPS if key in name)


def resolution_plot(min_particles=20):
    tiers = pd.read_csv(os.path.join(RESULTS_DIR, "resolution.csv"))
    thresholds = sorted(tiers["min_feret_px"].unique())
    counts = tiers.drop_duplicates("min_feret_px").set_index("min_feret_px")[[f"n_{c}" for c in POLYMERS]]
    x = list(range(len(thresholds)))

    fig, ax = plt.subplots(figsize=(12, 6.5))
    small = [i for i, t in enumerate(thresholds) if counts.loc[t].min() < min_particles]
    if small:
        ax.axvspan(min(small) - 0.5, max(small) + 0.5, color="#f2f2f2", zorder=0)
        ax.text(min(small) - 0.4, 73.5, f"fewer than {min_particles} test\nparticles in a polymer", fontsize=7, va="top", color="#666666")
    seen, ends = set(), []
    for name, g in tiers.groupby("method", sort=False):
        label, color, style = group_of(name)
        y = g.set_index("min_feret_px").loc[thresholds, "macro_f1"].to_numpy()
        ax.plot(x, y, color=color, ls=style, lw=1.4, alpha=0.85, marker="o", ms=3, label=None if label in seen else label)
        seen.add(label)
        ends.append([y[-1], name, color])
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 1.6)
    for y, name, color in ends:
        ax.text(x[-1] + 0.15, y, name, fontsize=7, color=color, va="center")
    ax.axhline(100 / len(POLYMERS), color="k", lw=0.8, ls=":")
    ax.text(0.05, 100 / len(POLYMERS) + 1, f"chance ({100 / len(POLYMERS):.0f}%)", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([("all" if t == 0 else f"≥ {t} px") + f"\nn = {counts.loc[t].sum()}\nmin {counts.loc[t].min()}"
                        for t in thresholds], fontsize=8)
    ax.set_xlabel("Particle resolution: max Feret diameter in pixels (cumulative); n = test particles, min = smallest polymer")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_ylim(15, 75)
    ax.set_xlim(-0.4, len(thresholds) + 1.2)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    ax.set_title("Polymer identification vs particle resolution (5-fold CV grouped by image, balanced test)", fontsize=10)
    fig.tight_layout()
    out = os.path.join(RESULTS_DIR, "resolution.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    argparse.ArgumentParser(description="Plot macro F1 against particle resolution (run polymer_id.evaluate first).").parse_args()
    print(resolution_plot())


if __name__ == "__main__":
    main()
