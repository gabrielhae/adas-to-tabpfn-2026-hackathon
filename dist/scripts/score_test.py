"""
Score every comparison model once across the whole held-out test split.

This is the batch counterpart to the explorer's per-clip /api/predict: for each
model it fits a single estimator on the train+val drivers and predicts every test
clip in one call (no clip is ever scored by a model that saw it).

Writes results/test_scores.json, which the app serves at GET /api/score and shows
top-left. Re-run after changing a model or the split:

  python scripts/score_test.py
  python scripts/score_test.py --models tabpfn lightgbm
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C
from adas_to import model as M
from adas_to import splits as S

from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, r2_score)

CLASS_TARGET = "post_maneuver_type"
REG_TARGETS = ["post_max_abs_steer_torque", "post_max_abs_jerk_mps3"]
RARE_MIN = 10


def _collapse(y: pd.Series, rare: set) -> pd.Series:
    y = y.astype("string")
    return y.where(~y.isin(rare), "other")


def score_classification(tr, te, schema, target, model_kind):
    P_tr = M.prepare(tr, schema, target)
    P_te = M.prepare(te, schema, target)
    vc = tr[target].astype("string").value_counts()
    rare = set(vc[vc < RARE_MIN].index)
    ytr, yte = _collapse(P_tr.y, rare), _collapse(P_te.y, rare)

    Xtr, Xte = M.encode(P_tr.X, P_te.X, P_tr.categorical)
    est = M.make_model(model_kind, task="classification", kv_cache=True)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)

    classes = np.unique(ytr)
    return {
        "target": target,
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(yte, pred)), 4),
        "macro_f1": round(float(f1_score(yte, pred, average="macro")), 4),
        "chance": round(1.0 / len(classes), 4),
        "classes": [str(c) for c in classes],
    }


def score_regression(tr, te, schema, target, model_kind):
    P_tr, P_te = M.prepare(tr, schema, target), M.prepare(te, schema, target)
    ytr = pd.to_numeric(P_tr.y, errors="coerce")
    yte = pd.to_numeric(P_te.y, errors="coerce")
    ok_tr, ok_te = ytr.notna(), yte.notna()
    if ok_tr.sum() < 2 or ok_te.sum() < 2:
        return None
    Xtr, Xte = M.encode(P_tr.X[ok_tr], P_te.X[ok_te], P_tr.categorical)
    est = M.make_model(model_kind, task="regression", kv_cache=True)
    est.fit(Xtr, ytr[ok_tr])
    pred = est.predict(Xte)
    y = yte[ok_te]
    return {
        "target": target,
        "n_train": int(ok_tr.sum()),
        "n_test": int(ok_te.sum()),
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }


def score_model(model_kind, tr, te, schema) -> dict:
    out: dict = {"available": True, "classification": None, "regression": {}}
    try:
        out["classification"] = score_classification(
            tr, te, schema, CLASS_TARGET, model_kind)
        for t in REG_TARGETS:
            if t in schema.get("targets_regression", []):
                r = score_regression(tr, te, schema, t, model_kind)
                if r:
                    out["regression"][t] = r
    except Exception as e:  # model unavailable / failed -> keep the others
        out["available"] = False
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=C.COMPARE_MODELS,
                    help=f"default: {' '.join(C.COMPARE_MODELS)}")
    args = ap.parse_args()

    assign = S.load_assignments()
    if assign is None:
        raise SystemExit("no split: run scripts/build_splits.py first")
    df = pd.read_parquet(C.MODEL_TABLE)
    schema = json.loads(C.SCHEMA_JSON.read_text(encoding="utf-8"))
    d = S.add_split(df, assign)
    tr = d[d.split.isin(["train", "val"])]
    te = d[d.split == "test"]

    print(f"scoring test split | train+val={len(tr)} rows "
          f"({tr.dongle_id.nunique()} drivers) | test={len(te)} rows "
          f"({te.dongle_id.nunique()} drivers)")

    models = {}
    for kind in args.models:
        r = score_model(kind, tr, te, schema)
        models[kind] = r
        if not r["available"]:
            print(f"  {kind:<9} UNAVAILABLE ({r['error']})")
            continue
        c = r["classification"]
        reg = r["regression"].get("post_max_abs_steer_torque", {})
        print(f"  {kind:<9} bal-acc={c['balanced_accuracy']:.3f} "
              f"macroF1={c['macro_f1']:.3f} acc={c['accuracy']:.3f}"
              + (f"  steer R2={reg['r2']:.3f}" if reg else ""))

    out = {
        "available": True,
        "order": list(args.models),
        "protocol": "driver-disjoint (dongle_id); fit train+val -> predict test",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_train_drivers": int(tr.dongle_id.nunique()),
        "n_test_drivers": int(te.dongle_id.nunique()),
        "models": models,
    }
    C.TEST_SCORES.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {C.TEST_SCORES}")


if __name__ == "__main__":
    main()
