"""
M1 - honest evaluation.

Protocol (the dataset's own recommendation):
  * driver-disjoint GroupKFold  -> "prevent within-driver information leakage"
  * leave-one-brand-out OOD     -> cross-platform generalisation
  * report sensitivity by log_kind (qlog 10 Hz vs rlog 100 Hz), which the paper mandates

Tasks:
  T1  pre_* -> post_maneuver_type            (5->4 class after rare-collapse)
  T2  pre_* -> post_max_abs_steer_torque     (regression, reaction intensity)
  T3  pre_* -> post_max_abs_jerk_mps3        (regression, ride quality)

Usage:
  python scripts/run_experiments.py                 # auto (tabpfn if available else lightgbm)
  python scripts/run_experiments.py --models logistic lightgbm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

from adas_to import config as C
from adas_to import model as M

from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import (balanced_accuracy_score, f1_score, mean_absolute_error,
                             r2_score, accuracy_score)


def load():
    df = pd.read_parquet(C.MODEL_TABLE)
    schema = json.loads(C.SCHEMA_JSON.read_text(encoding="utf-8"))
    return df, schema


def cv_eval(df, schema, target, task, model_kind, *, split="driver",
            n_splits=5, seed=0, verbose=True):
    P = M.prepare(df, schema, target)
    y = P.y
    if task == "classification":
        y = M.collapse_rare(y, 10)
    else:
        y = pd.to_numeric(y, errors="coerce")

    if split == "driver":
        # GroupKFold has no shuffle; assign each group a fold deterministically
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(P.X, y, P.groups))
    elif split == "brand":
        codes = pd.factorize(df.loc[y.index, "brand"])[0]
        logo = LeaveOneGroupOut()
        folds = list(logo.split(P.X, y, codes))
    else:
        raise ValueError(split)

    metrics, per_fold = [], []
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        if len(te) == 0 or len(tr) < 20:
            continue
        Xtr, Xte = M.encode(P.X.iloc[tr], P.X.iloc[te], P.categorical)
        ytr, yte = y.iloc[tr], y.iloc[te]
        if task == "classification" and ytr.nunique() < 2:
            continue
        est = M.make_model(model_kind, task=task, seed=seed)
        try:
            est.fit(Xtr, ytr)
            pred = est.predict(Xte)
        except Exception as e:
            if verbose:
                print(f"      fold {k}: FAILED {type(e).__name__}: {str(e)[:160]}")
            continue
        if task == "classification":
            m = {
                "balanced_acc": balanced_accuracy_score(yte, pred),
                "macro_f1": f1_score(yte, pred, average="macro"),
                "acc": accuracy_score(yte, pred),
                "n_test": len(te),
            }
        else:
            m = {
                "mae": mean_absolute_error(yte, pred),
                "r2": r2_score(yte, pred) if len(te) > 2 else np.nan,
                "n_test": len(te),
            }
        per_fold.append(m)
        metrics.append(m)
    dt = time.time() - t0
    if not metrics:
        return None
    mean = pd.DataFrame(metrics).mean(numeric_only=True).to_dict()
    return {"task": task, "target": target, "model": model_kind, "split": split,
            "mean": mean, "n_folds": len(metrics), "seconds": round(dt, 1),
            "per_fold": per_fold}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["auto"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df, schema = load()
    print(f"model table: {len(df):,} rows x {len(schema['features_numeric'])} numeric features")

    ok, why = M.tabpfn_available()
    print(f"TabPFN available: {ok}  ({why})\n")

    models = args.models
    if "auto" in models:
        models = ["tabpfn"] if ok else ["lightgbm"]

    tasks = [
        ("post_maneuver_type", "classification"),
        ("post_max_abs_steer_torque", "regression"),
        ("post_max_abs_jerk_mps3", "regression"),
    ]

    rows = []
    for target, task in tasks:
        for mk in models:
            label = f"{task[:3]}:{target}"
            print(f"--- {label:52s} model={mk:9s} split=driver")
            r = cv_eval(df, schema, target, task, mk, split="driver", seed=args.seed)
            if r:
                if task == "classification":
                    print(f"      bal_acc={r['mean']['balanced_acc']:.3f}  "
                          f"macroF1={r['mean']['macro_f1']:.3f}  "
                          f"acc={r['mean']['acc']:.3f}  "
                          f"folds={r['n_folds']}  {r['seconds']}s")
                else:
                    print(f"      MAE={r['mean']['mae']:.4f}  R2={r['mean']['r2']:.3f}  "
                          f"folds={r['n_folds']}  {r['seconds']}s")
                rows.append(r)

    # --- log-rate sensitivity on the headline task ---
    print("\n--- log_kind sensitivity (paper mandates reporting this) ---")
    for lk in ["qlog", "rlog"]:
        sub = df[df["log_kind"] == lk]
        for mk in models:
            r = cv_eval(sub, schema, "post_maneuver_type", "classification", mk,
                        split="driver", seed=args.seed)
            if r:
                print(f"      {lk} (n={len(sub):4d}) {mk:9s} "
                      f"bal_acc={r['mean']['balanced_acc']:.3f}  "
                      f"macroF1={r['mean']['macro_f1']:.3f}")
                r["stratum"] = lk
                rows.append(r)

    # --- brand-disjoint OOD ---
    print("\n--- leave-one-brand-out (cross-platform OOD) ---")
    for mk in models:
        r = cv_eval(df, schema, "post_maneuver_type", "classification", mk,
                    split="brand", seed=args.seed)
        if r:
            print(f"      {mk:9s} bal_acc={r['mean']['balanced_acc']:.3f}  "
                  f"macroF1={r['mean']['macro_f1']:.3f}  brands={r['n_folds']}")
            rows.append(r)

    out = C.RESULTS / "m1_results.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}")

    tab = pd.DataFrame([
        {"task": r["task"], "target": r["target"], "model": r["model"],
         "split": r.get("split"), "stratum": r.get("stratum"),
         **{k: round(v, 4) for k, v in r["mean"].items()}}
        for r in rows])
    tab.to_csv(C.RESULTS / "m1_results.csv", index=False)
    print("\n===== SUMMARY =====")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
