"""
Per-class evaluation for the headline task `post_maneuver_type`.

Eval-only reuse of the M1 protocol in run_experiments.py: same table, same
collapse_rare(10), same driver-disjoint GroupKFold / leave-one-brand-out splits,
same `M.prepare` / `M.encode` / `M.make_model` path, same seed. The only addition
is that out-of-fold predictions are accumulated so that per-class
precision/recall/F1 and confusion matrices can be reported.

Writes:
  results/per_class_results.json   full per-class + per-fold detail
  results/per_class_results.csv    flat per-class table (split x model x class)
  results/per_class_f1.png         grouped bar figure (driver split)
  results/per_class_confusion.png  confusion matrices (driver split)

Usage:
  python scripts/eval_per_class.py --models logistic lightgbm tabpfn
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from adas_to import config as C
from adas_to import model as M

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score, accuracy_score)

TARGET = "post_maneuver_type"
# Plot order: majority first, then rarer classes.
CLASS_ORDER = ["stabilize", "lane_change", "braking", "acceleration", "other"]


def load():
    df = pd.read_parquet(C.MODEL_TABLE)
    schema = json.loads(C.SCHEMA_JSON.read_text(encoding="utf-8"))
    return df, schema


def make_folds(df, P, y, split, n_splits=5):
    if split == "driver":
        return list(GroupKFold(n_splits=n_splits).split(P.X, y, P.groups))
    if split == "brand":
        codes = pd.factorize(df.loc[y.index, "brand"])[0]
        return list(LeaveOneGroupOut().split(P.X, y, codes))
    raise ValueError(split)


def run(df, schema, model_kind, split, seed=0, verbose=True):
    P = M.prepare(df, schema, TARGET)
    y = M.collapse_rare(P.y, 10)
    folds = make_folds(df, P, y, split, n_splits=5)

    truth, preds, used = [], [], 0
    for k, (tr, te) in enumerate(folds):
        if len(te) == 0 or len(tr) < 20:
            continue
        Xtr, Xte = M.encode(P.X.iloc[tr], P.X.iloc[te], P.categorical)
        ytr, yte = y.iloc[tr], y.iloc[te]
        if ytr.nunique() < 2:
            continue
        est = M.make_model(model_kind, task="classification", seed=seed)
        try:
            est.fit(Xtr, ytr)
            pred = np.asarray(est.predict(Xte))
        except Exception as e:
            if verbose:
                print(f"    fold {k}: FAILED {type(e).__name__}: {str(e)[:140]}")
            continue
        truth.append(pd.Series(np.asarray(yte), index=yte.index))
        preds.append(pd.Series(pred, index=yte.index))
        used += 1

    if not truth:
        return None

    y_true = pd.concat(truth)
    y_pred = pd.concat(preds)

    labels = [c for c in CLASS_ORDER if c in set(y_true.unique())]
    labels += [c for c in sorted(y_true.unique()) if c not in labels]

    rep = classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {}
    for c in labels:
        r = rep[c]
        per_class[c] = {"precision": round(r["precision"], 4),
                        "recall": round(r["recall"], 4),
                        "f1": round(r["f1-score"], 4),
                        "support": int(r["support"])}

    pooled = {
        "balanced_acc": round(balanced_accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "acc": round(accuracy_score(y_true, y_pred), 4),
        "n_test": int(len(y_true)),
        "n_folds_used": used,
    }
    return {
        "task": "classification",
        "target": TARGET,
        "model": model_kind,
        "split": split,
        "labels": labels,
        "per_class": per_class,
        "pooled": pooled,
        "confusion_matrix": cm.tolist(),
    }


def flatten(rows):
    out = []
    for r in rows:
        for c, m in r["per_class"].items():
            out.append({
                "split": r["split"], "model": r["model"], "class": c,
                "precision": m["precision"], "recall": m["recall"],
                "f1": m["f1"], "support": m["support"],
            })
    return pd.DataFrame(out)


def plot(rows, out_png, out_cm_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    driver = [r for r in rows if r["split"] == "driver"]
    if not driver:
        return
    models = [r["model"] for r in driver]
    labels = driver[0]["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharey=True)
    metrics = [("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]
    x = np.arange(len(labels))
    width = 0.8 / max(len(models), 1)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for ax, (key, title) in zip(axes, metrics):
        for i, r in enumerate(driver):
            vals = [r["per_class"][c][key] for c in labels]
            bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                          label=r["model"], color=colors[i % len(colors)])
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=7.5)
        ax.set_title(f"Per-class {title} — driver-disjoint OOF", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.axhline(0, color="#999", lw=0.6)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("score")
    axes[0].legend(frameon=False)
    fig.suptitle("post_maneuver_type — class-by-class scores "
                 "(pooled out-of-fold, 5-fold GroupKFold by driver)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    # confusion matrices, row-normalised
    n = len(driver)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4))
    if n == 1:
        axes = [axes]
    cmap = LinearSegmentedColormap.from_list("blues2", ["#f7fbff", "#08306b"])
    for ax, r in zip(axes, driver):
        cm = np.asarray(r["confusion_matrix"], dtype=float)
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(range(len(r["labels"])))
        ax.set_xticklabels(r["labels"], rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(r["labels"])))
        ax.set_yticklabels(r["labels"], fontsize=8)
        ax.set_title(f"{r['model']} (bal-acc {r['pooled']['balanced_acc']:.3f})",
                     fontsize=10)
        ax.set_xlabel("predicted")
        for i in range(len(r["labels"])):
            for j in range(len(r["labels"])):
                ax.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center",
                        fontsize=7.5,
                        color="white" if norm[i, j] > 0.55 else "#333")
    axes[0].set_ylabel("actual")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="row-normalised")
    fig.suptitle("Confusion matrices — post_maneuver_type, driver-disjoint OOF",
                 fontsize=12)
    fig.savefig(out_cm_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["logistic", "lightgbm", "tabpfn"])
    ap.add_argument("--splits", nargs="+", default=["driver", "brand"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, schema = load()
    print(f"model table: {len(df):,} rows")
    ok, why = M.tabpfn_available()
    print(f"TabPFN available: {ok}  ({why})\n")

    rows = []
    for split in args.splits:
        for mk in args.models:
            print(f"--- {TARGET}  model={mk:9s} split={split}")
            r = run(df, schema, mk, split, seed=args.seed)
            if r:
                p = r["pooled"]
                print(f"    pooled bal_acc={p['balanced_acc']:.3f} "
                      f"macroF1={p['macro_f1']:.3f} acc={p['acc']:.3f} "
                      f"(n={p['n_test']})")
                for c, m in r["per_class"].items():
                    print(f"      {c:13s} P={m['precision']:.3f} "
                          f"R={m['recall']:.3f} F1={m['f1']:.3f} "
                          f"(n={m['support']})")
                rows.append(r)

    (C.RESULTS / "per_class_results.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    tab = flatten(rows)
    tab.to_csv(C.RESULTS / "per_class_results.csv", index=False)
    plot(rows, C.RESULTS / "per_class_f1.png",
         C.RESULTS / "per_class_confusion.png")
    print(f"\nwrote {C.RESULTS / 'per_class_results.json'}")
    print(f"wrote {C.RESULTS / 'per_class_results.csv'}")
    print(f"wrote {C.RESULTS / 'per_class_f1.png'}")
    print(f"wrote {C.RESULTS / 'per_class_confusion.png'}")


if __name__ == "__main__":
    main()
