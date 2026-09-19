"""
Vision ablation - does adding YOLOv8n geometry to the CAN features improve the
sliding takeover forecast?

Three feature sets, one identical CV protocol, so the comparison is apples-to-apples:

    can   = the published `forecast_schema.json` features (must reproduce the
            baselines in results/forecast_results.json fold-for-fold)
    vis   = only the `vis_*` features (pure vision)
    both  = CAN + vision

The CV core (`cv_eval`), preprocessing (`prepare`/`encode`), model factory and
rare-class collapse are IMPORTED from the existing harness - this script does not
fork them. Driver `GroupKFold(5)` is deterministic (no shuffle) and the folds are
a function of `dongle_id` only, so all three feature sets see identical splits.

Writes:
    results/vision_results.json
    results/vision_results.csv
    results/vision_ablation.csv     (task, model, feature_set, metric, value)
    results/vision_by_lead.csv      (mirrors forecast_by_lead.csv schema)

Usage:
    python scripts/run_vision_experiments.py                      # auto backend
    python scripts/run_vision_experiments.py --models logistic lightgbm
    python scripts/run_vision_experiments.py --feature-sets can both --no-per-lead
    python scripts/run_vision_experiments.py --brand --models logistic lightgbm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import warnings
warnings.filterwarnings("ignore")

from adas_to import config as C
from adas_to import model as M
import run_experiments as RE

TASKS = [
    ("post_maneuver_type", "classification"),
    ("post_max_abs_steer_torque", "regression"),
    ("post_max_abs_jerk_mps3", "regression"),
]
LEAD_TASKS = [
    ("post_maneuver_type", "classification"),
    ("post_max_abs_steer_torque", "regression"),
]


def load_vision():
    df = pd.read_parquet(C.FORECAST_TABLE_VISION)
    schema = json.loads(C.FORECAST_SCHEMA_VISION.read_text(encoding="utf-8"))
    return df, schema


def feature_set_schema(schema: dict, name: str, drop: set) -> dict | None:
    vis = [c for c in schema["features_numeric"] if c.startswith("vis_") and c not in drop]
    can = [c for c in schema["features_numeric"] if not c.startswith("vis_") and c not in drop]
    if name == "can":
        return {**schema, "features_numeric": can,
                "features_categorical": schema["features_categorical"]}
    if name == "both":
        return {**schema, "features_numeric": can + vis,
                "features_categorical": schema["features_categorical"]}
    if name == "vis":
        if not vis:
            return None
        return {**schema, "features_numeric": vis, "features_categorical": []}
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["auto"])
    ap.add_argument("--feature-sets", nargs="+", default=["can", "vis", "both"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-lead", action="store_true")
    ap.add_argument("--brand", action="store_true",
                    help="also run leave-one-brand-out for classification")
    args = ap.parse_args()

    df, schema = load_vision()
    n_expected = int(schema["n_rows"])
    assert len(df) == n_expected, f"vision table has {len(df)} rows, expected {n_expected}"
    dead = {c for c in schema["features_numeric"] if df[c].isna().all()}
    if dead:
        print(f"[ablation] dropping all-NaN feature columns: {sorted(dead)}")

    ok, why = M.tabpfn_available()
    print(f"[ablation] TabPFN available: {ok}  ({why})")
    models = args.models
    if "auto" in models:
        models = ["tabpfn"] if ok else ["lightgbm"]
    print(f"[ablation] models = {models}\n")

    schemas = {n: feature_set_schema(schema, n, dead) for n in args.feature_sets}
    runs: list[dict] = []

    print("=== full-table (all leads pooled) ===")
    for fs, sch in schemas.items():
        if sch is None:
            print(f"--- {fs}: no usable features, skipped")
            continue
        for target, task in TASKS:
            for mk in models:
                t0 = time.time()
                r = RE.cv_eval(df, sch, target, task, mk, split="driver", seed=args.seed)
                if not r:
                    print(f"      {fs:4s} {task[:3]}:{target:28s} {mk:9s} SKIPPED")
                    continue
                r["feature_set"] = fs
                r["n_features"] = len(sch["features_numeric"])
                runs.append(r)
                if task == "classification":
                    print(f"      {fs:4s} {task[:3]}:{target:28s} {mk:9s} "
                          f"bal_acc={r['mean']['balanced_acc']:.3f} "
                          f"macroF1={r['mean']['macro_f1']:.3f} "
                          f"({len(sch['features_numeric'])} feats, {time.time()-t0:.0f}s)")
                else:
                    print(f"      {fs:4s} {task[:3]}:{target:28s} {mk:9s} "
                          f"MAE={r['mean']['mae']:.4f} R2={r['mean']['r2']:.3f} "
                          f"({len(sch['features_numeric'])} feats, {time.time()-t0:.0f}s)")

    if args.brand:
        print("\n=== leave-one-brand-out (classification) ===")
        for fs, sch in schemas.items():
            if sch is None:
                continue
            for mk in models:
                r = RE.cv_eval(df, sch, "post_maneuver_type", "classification", mk,
                               split="brand", seed=args.seed)
                if r:
                    r["feature_set"] = fs
                    runs.append(r)
                    print(f"      {fs:4s} brand {mk:9s} "
                          f"bal_acc={r['mean']['balanced_acc']:.3f} "
                          f"macroF1={r['mean']['macro_f1']:.3f}")

    per_lead_rows = []
    if not args.no_per_lead:
        print("\n=== per-lead breakdown (headline) ===")
        leads = sorted(df["lead_s"].dropna().unique())
        for lead in leads:
            sub = df[df["lead_s"] == lead]
            row = {"lead_s": float(lead), "n": int(len(sub))}
            for fs, sch in schemas.items():
                if sch is None:
                    continue
                for target, task in LEAD_TASKS:
                    for mk in models:
                        r = RE.cv_eval(sub, sch, target, task, mk, split="driver",
                                       seed=args.seed)
                        if not r:
                            continue
                        r["feature_set"] = fs
                        r["lead_s"] = float(lead)
                        runs.append(r)
                        if task == "classification":
                            row[f"{fs}|{mk}|balanced_acc"] = r["mean"]["balanced_acc"]
                            row[f"{fs}|{mk}|macro_f1"] = r["mean"]["macro_f1"]
                        else:
                            row[f"{fs}|{mk}|steer_r2"] = r["mean"]["r2"]
                            row[f"{fs}|{mk}|steer_mae"] = r["mean"]["mae"]
            per_lead_rows.append(row)
            clf = " ".join(f"{k.split('|')[0]}:{k.split('|')[1]}={v:.3f}"
                           for k, v in row.items() if k.endswith("balanced_acc"))
            r2 = " ".join(f"{k.split('|')[0]}:{k.split('|')[1]}={v:.3f}"
                          for k, v in row.items() if k.endswith("steer_r2"))
            print(f"      lead={lead:4.1f}s  bal_acc[{clf}]  steerR2[{r2}]")

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    results_json = C.RESULTS / "vision_results.json"
    existing: list[dict] = []
    if results_json.exists():
        try:
            existing = json.loads(results_json.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    def run_key(r: dict):
        lead = r.get("lead_s")
        return (r.get("feature_set"), r["task"], r["target"], r["model"], r["split"],
                None if lead is None else round(float(lead), 6))

    merged = {run_key(r): r for r in existing}
    for r in runs:
        merged[run_key(r)] = r
    all_runs = list(merged.values())
    results_json.write_text(json.dumps(all_runs, indent=2, default=str), encoding="utf-8")

    flat = pd.DataFrame([
        {"scope": "per_lead" if "lead_s" in r else "full",
         "feature_set": r.get("feature_set"), "task": r["task"], "target": r["target"],
         "model": r["model"], "split": r["split"], "lead_s": r.get("lead_s"),
         "n_features": r.get("n_features"), "n_folds": r.get("n_folds"),
         **{k: round(v, 4) for k, v in r["mean"].items()}}
        for r in all_runs])
    flat.to_csv(C.RESULTS / "vision_results.csv", index=False)

    ablation = pd.DataFrame([
        {"task": r["task"], "model": r["model"], "feature_set": r.get("feature_set"),
         "metric": metric, "value": round(val, 4)}
        for r in all_runs if "lead_s" not in r
        for metric, val in r["mean"].items()])
    ablation.to_csv(C.RESULTS / "vision_ablation.csv", index=False)

    lead_runs = [r for r in all_runs if "lead_s" in r]
    if lead_runs:
        rows = []
        for lead in sorted({float(r["lead_s"]) for r in lead_runs}):
            sub = [r for r in lead_runs if float(r["lead_s"]) == lead]
            row = {"lead_s": lead,
                   "n": int(round(sub[0]["mean"].get("n_test", np.nan)
                                  * sub[0].get("n_folds", 1)))}
            for r in sorted(sub, key=lambda x: (x.get("feature_set", ""), x["model"], x["task"])):
                fs, mk = r.get("feature_set"), r["model"]
                if r["task"] == "classification":
                    row[f"{fs}|{mk}|balanced_acc"] = r["mean"]["balanced_acc"]
                    row[f"{fs}|{mk}|macro_f1"] = r["mean"]["macro_f1"]
                else:
                    row[f"{fs}|{mk}|steer_r2"] = r["mean"]["r2"]
                    row[f"{fs}|{mk}|steer_mae"] = r["mean"]["mae"]
            rows.append(row)
        pd.DataFrame(rows).to_csv(C.RESULTS / "vision_by_lead.csv", index=False)

    print(f"\nwrote {results_json} ({len(all_runs)} runs)")
    print(f"wrote {C.RESULTS / 'vision_ablation.csv'}")
    if lead_runs:
        print(f"wrote {C.RESULTS / 'vision_by_lead.csv'}")


if __name__ == "__main__":
    main()
