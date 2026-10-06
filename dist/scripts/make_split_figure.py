"""
Illustrate the training / validation split used by scripts/run_experiments.py.

  * primary  : driver-disjoint GroupKFold(n_splits=5) on dongle_id
  * secondary: leave-one-brand-out (LeaveOneGroupOut) for cross-platform OOD

The fold assignment is recomputed from data/derived/forecast_table.parquet with the
exact same call used at evaluation time, so the picture matches the runs.

Writes results/split_explainer.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adas_to import config as C

TRAIN = "#4C78A8"
TEST = "#E45756"
INK = "#222222"
GRID = "#B8B8B8"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
})


def groupkold_assignment(df: pd.DataFrame, target: str, n_splits: int = 5):
    d = df[df[target].notna()].copy()
    y = d[target]
    groups = d["dongle_id"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    fold_of_driver: dict = {}
    fold_stats = []
    for k, (tr, te) in enumerate(gkf.split(d, y, groups)):
        test_drivers = set(d.iloc[te]["dongle_id"])
        for g in test_drivers:
            fold_of_driver[g] = k
        fold_stats.append({
            "fold": k,
            "n_train_rows": int(len(tr)),
            "n_test_rows": int(len(te)),
            "n_test_drivers": len(test_drivers),
        })
    return d, fold_of_driver, pd.DataFrame(fold_stats)


def panel_a(ax, d, fold_of_driver, fold_stats, n_splits):
    size = d.groupby("dongle_id").size()
    order = sorted(size.index, key=lambda g: (fold_of_driver[g], -size[g]))
    M = np.zeros((n_splits, len(order)))
    for j, g in enumerate(order):
        M[fold_of_driver[g], j] = 1.0

    ax.imshow(M, aspect="auto", interpolation="nearest",
              cmap=matplotlib.colors.ListedColormap([TRAIN, TEST]),
              vmin=0, vmax=1)

    ax.set_yticks(range(n_splits))
    ax.set_yticklabels([f"Round {k+1}" for k in range(n_splits)], fontsize=9)
    ax.set_ylabel("CV round\n(one fold held out)", fontsize=9)
    ax.set_xticks([])
    ax.set_xlabel("206 drivers (dongle_id) - every driver's clips and all 14 sliding windows stay in one fold",
                  fontsize=9)
    ax.set_title("A.  Primary split: driver-disjoint GroupKFold(n_splits=5)",
                 fontsize=12, fontweight="bold", loc="left", pad=30)

    bounds, start = [], 0
    for k in range(n_splits):
        cnt = sum(1 for g in order if fold_of_driver[g] == k)
        bounds.append((start, start + cnt))
        start += cnt

    y0, y1 = -0.60, -0.60
    for k, (a, b) in enumerate(bounds):
        st = fold_stats.iloc[k]
        ax.plot([a - 0.5, b - 0.5], [y0, y0], color=GRID, lw=1.0, clip_on=False)
        ax.plot([a - 0.5, a - 0.5], [y0, y0 + 0.28], color=GRID, lw=1.0, clip_on=False)
        ax.plot([b - 0.5, b - 0.5], [y0, y0 + 0.28], color=GRID, lw=1.0, clip_on=False)
        ax.text((a + b) / 2 - 0.5, -0.74,
                f"held out #{k+1}\n{st.n_test_drivers} drivers, {st.n_test_rows:,} rows",
                ha="center", va="bottom", fontsize=7.6, color=TEST, clip_on=False)
    ax.text(len(order) + 2, n_splits - 1,
            "Rows per driver\nmin 14  median 28  max 1,960",
            ha="left", va="center", fontsize=8, color=INK, clip_on=False)
    ax.legend(handles=[Patch(facecolor=TRAIN, label="train"),
                       Patch(facecolor=TEST, label="validation / test (held out)")],
              loc="upper left", bbox_to_anchor=(1.005, 1.0),
              frameon=False, fontsize=9, title="Fold role", title_fontsize=9)
    for s in ax.spines.values():
        s.set_visible(False)


def panel_b(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    ax.set_title("B.  Why it is grouped by driver: one driver's whole history lands in a single split",
                 fontsize=12, fontweight="bold", loc="left")

    def driver(y, label, note, n_clips, color):
        ax.text(1, y + 12.6, label, fontsize=10, fontweight="bold", color=INK)
        ax.text(1, y + 9.6, note, fontsize=8.4, color=color)
        for ci in range(n_clips):
            yy = y + 5.4 - ci * 3.4
            ax.text(13.6, yy + 1.2, f"clip {ci+1}", ha="right", va="center",
                    fontsize=8, color=INK)
            for w in range(14):
                ax.add_patch(Rectangle((15 + w * 2.35, yy), 2.05, 2.6,
                                       facecolor=color, edgecolor="white", lw=0.6))

    driver(26, "Driver A  ->  train fold", "all clips, all 14 windows: train",
           2, TRAIN)
    driver(6, "Driver B  ->  validation / test fold", "all clips, all 14 windows: held out",
           2, TEST)

    ax.annotate("each clip = 14 sliding windows\n(lead 0 -> 6.5 s, stride 0.5 s)",
                xy=(15 + 13 * 2.35 + 1.0, 30.6), xytext=(58, 31.0),
                fontsize=8.6, color=INK, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=GRID, lw=1.0))
    ax.text(1, -1.0,
            "Grouping key dongle_id subsumes clip, and every clip carries its 14 windows, so no window of a "
            "test driver can appear in training.",
            fontsize=8.8, color=INK)


def panel_c(ax, d):
    vc = d["brand"].value_counts().sort_values(ascending=False)
    colors = [TEST if b == "Ford" else TRAIN for b in vc.index]
    x = np.arange(len(vc))
    ax.bar(x, vc.values, color=colors, width=0.72, edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(vc.index, rotation=55, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(8, max(vc.values) * 3.0)
    ax.set_ylabel("rows (windows)", fontsize=9)
    ax.set_title("C.  Secondary split: leave-one-brand-out OOD (LeaveOneGroupOut, 21 rounds)",
                 fontsize=12, fontweight="bold", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax.annotate("one brand held out entirely as the\nOOD test set, repeated for all 21 brands",
                xy=(0.02, vc.values[0] * 1.04), xytext=(0.45, vc.values[0] * 1.62),
                fontsize=8.8, color=TEST, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=TEST, lw=1.1))
    ax.text(0.985, 0.93, f"largest brand = {vc.values[0]:,} rows  |  smallest = {vc.values[-1]}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=INK)


def main() -> None:
    target = "post_maneuver_type"
    df = pd.read_parquet(C.FORECAST_TABLE)
    d, fold_of_driver, fold_stats = groupkold_assignment(df, target)

    fig = plt.figure(figsize=(15.5, 11.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.05, 1.20, 1.95], hspace=0.85)

    panel_a(fig.add_subplot(gs[0]), d, fold_of_driver, fold_stats, 5)
    panel_b(fig.add_subplot(gs[1]))
    panel_c(fig.add_subplot(gs[2]), d)

    fig.suptitle("Train / validation split of the ADAS-TO sliding forecast dataset",
                 fontsize=15, fontweight="bold", x=0.09, ha="left", y=0.985)
    fig.text(0.09, 0.955,
             f"{len(d):,} rows  .  {d['dongle_id'].nunique()} drivers  .  "
             f"{d.drop_duplicates(['car_model','driver','route','clip_id_pub']).shape[0]:,} clips  .  "
             f"{d['brand'].nunique()} brands  .  no fixed test file: every row is scored once as held-out",
             fontsize=9.5, color=INK, ha="left")

    out = C.RESULTS / "split_explainer.png"
    fig.savefig(out)
    print(f"wrote {out}")
    print(fold_stats.to_string(index=False))


if __name__ == "__main__":
    main()
