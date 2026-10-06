"""
M1 - build the modeling table.

Sources
-------
1. `_recon/id_map.csv`         public sample clips -> repo (dongle_id, route_id, clip_id)
2. `analysis_master.csv`       210 columns incl. labels + pre_*/post_* aggregates

Design decisions (all justified in PLAN.md REV 3)
-------------------------------------------------
* FEATURES come only from the **pre** window (t in [-5, 0]); TARGETS only from the
  **post** window (t in [0, 5]). The windows meet at t=0, so this is a genuine
  forecast, not leakage. Enforced by `assert_no_leakage()`.
* `primary_trigger` / `trig_*` are EXCLUDED from features: the paper defines them from
  signals in [-3.0, +0.5] s, which overlaps the pre window. Using them would be circular.
* Splits are driver-disjoint (the paper's stated protocol).

Outputs: data/derived/model_table.parquet, clip_index.parquet, schema.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C

# ---------------------------------------------------------------- clip index


def build_clip_index() -> pd.DataFrame:
    """One row per clip present on disk, with video path + meta."""
    rows = []
    root = C.SAMPLE_DIR
    for dp, _dn, fn in os.walk(root):
        if "meta.json" not in fn:
            continue
        rel = Path(dp).relative_to(root)
        parts = rel.parts
        if len(parts) != 4:
            continue
        cm, dr, ro, ci = parts
        try:
            m = json.loads((Path(dp) / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            m = {}
        rec = {
            "car_model": cm, "driver": dr, "route": ro, "clip_id_pub": int(ci),
            "clip_dir": str(Path(dp)),
            "video": str(Path(dp) / "takeover.mp4"),
        }
        for k in ("log_kind", "log_hz", "vid_kind", "camera_fps",
                  "clip_dur_s", "clip_start_s", "video_time_s", "event_mono"):
            rec[k] = m.get(k)
        rows.append(rec)
    idx = pd.DataFrame(rows).sort_values(["car_model", "driver", "route", "clip_id_pub"])
    return idx.reset_index(drop=True)


# ---------------------------------------------------------------- model table

TEXT_COLS = {
    # non-numeric pre_*/post_* columns - must NOT be coerced or used as features
    "post_maneuver_type", "post_fcw_source", "post_alert_text",
    "pre_fcw_source", "pre_alert_text",
}

FEATURE_DENY = {
    # label itself and its raw rule inputs (derived from the trigger window [-3,+0.5])
    "primary_trigger", "trig_steer", "trig_brake", "trig_gas", "n_triggers",
    # rule-derived labels (would be circular as features)
    "scenario", "ego_reason", "nonego_reason", "label",
    # every post-window aggregate
    # (filtered by prefix anyway, listed for documentation)
    # bookkeeping / non-feature columns
    "is_noise", "n_segs", "source", "source_group",
}


def _is_post(c: str) -> bool:
    return c.startswith("post")


def build_model_table() -> tuple[pd.DataFrame, dict]:
    idmap = pd.read_csv(C.ID_MAP, low_memory=False)
    ok = idmap[(idmap["n_match"] == 1) & idmap["dongle_id"].notna()].copy()
    ok["route_id_repo"] = ok["route_id_repo"].astype(str)
    ok["clip_id_repo"] = ok["clip_id_repo"].astype(int)

    link = ok[["car_model", "dongle_id", "route_id_repo", "clip_id_repo",
               "driver", "route", "clip_id_pub"]].rename(
        columns={"route_id_repo": "route_id", "clip_id_repo": "clip_id"})

    am = pd.read_csv(C.ANALYSIS_MASTER, low_memory=False)
    am["route_id"] = am["route_id"].astype(str)
    # de-duplicate repo label column names for clarity
    df = link.merge(am, on=C.KEY_COLS, how="left")
    assert df["primary_trigger"].notna().all(), "label join incomplete"

    # ---- coerce numeric where possible ----
    # NOTE: several pre_*/post_* columns are NOT numeric (see TEXT_COLS). Coercing them
    # would silently turn the classification target into all-NaN.
    for c in df.columns:
        if c in TEXT_COLS:
            df[c] = df[c].astype("string")
            continue
        if c.startswith(("pre_", "post_")) or c in (
                "risk_score", "maneuver_score", "speed_mps", "accel_mps2",
                "steer_angle_deg", "steer_torque", "cruise_speed_kmh"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # ---- feature / target definition ----
    static_feats = ["log_hz", "clip_dur_s", "speed_mps", "accel_mps2",
                    "steer_angle_deg", "steer_torque", "cruise_speed_kmh"]
    pre_feats = [c for c in df.columns
                 if c.startswith("pre_") and c not in TEXT_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    feats = [c for c in dict.fromkeys(static_feats + pre_feats) if c in df.columns]

    cat_feats = [c for c in ["log_kind", "brand", "powertrain", "car_model", "vid_kind"]
                 if c in df.columns]

    # classification target
    y_cls = "post_maneuver_type"
    # regression targets
    y_reg = [c for c in [
        "post_max_abs_steer_torque", "post_steer_rate_max_deg_per_s",
        "post_max_abs_jerk_mps3", "post_min_accel_mps2", "post_roughness_rms_mps2",
        "stabilization_5s_time_s",
    ] if c in df.columns]

    # leakage guard: no feature may be a post-window column
    bad = [c for c in feats if _is_post(c)] + [c for c in feats if c in FEATURE_DENY]
    assert not bad, f"leaky features present: {bad}"

    schema = {
        "n_rows": int(len(df)),
        "features_numeric": feats,
        "features_categorical": cat_feats,
        "target_classification": y_cls,
        "targets_regression": y_reg,
        "group_col": "dongle_id",
        "groups": {
            "driver": "dongle_id",
            "brand": "brand",
            "car_model": "car_model",
        },
        "strata": ["log_kind"],
        "excluded_from_features": sorted(FEATURE_DENY),
        "leakage_note": (
            "Features are pre-window (t in [-5,0]); targets are post-window (t in [0,5]). "
            "primary_trigger/trig_* excluded because the paper derives them from [-3,+0.5] s, "
            "which overlaps the pre window."
        ),
        "class_balance": df[y_cls].value_counts().to_dict(),
    }
    return df, schema


def main() -> None:
    print("[1/3] clip index ...")
    idx = build_clip_index()
    idx.to_parquet(C.CLIP_INDEX, index=False)
    print(f"      {len(idx):,} clips -> {C.CLIP_INDEX}")
    print(f"      models={idx.car_model.nunique()} drivers={idx.driver.nunique()} "
          f"routes={idx.groupby(['car_model','driver','route']).ngroups}")
    print(f"      log_kind: {idx.log_kind.value_counts().to_dict()}")

    print("[2/3] model table ...")
    df, schema = build_model_table()
    df.to_parquet(C.MODEL_TABLE, index=False)
    print(f"      {len(df):,} rows -> {C.MODEL_TABLE}")
    print(f"      numeric features : {len(schema['features_numeric'])}")
    print(f"      categorical      : {schema['features_categorical']}")
    print(f"      target (cls)     : {schema['target_classification']}")
    print(f"      targets (reg)    : {len(schema['targets_regression'])}")
    print(f"      class balance    : {schema['class_balance']}")

    print("[3/3] schema ...")
    C.SCHEMA_JSON.write_text(json.dumps(schema, indent=2, default=str), encoding="utf-8")
    print(f"      -> {C.SCHEMA_JSON}")

    print("\n[leakage guard] no post-window column is a feature: OK")
    print(f"[groups] clips per driver: median="
          f"{df.groupby('dongle_id').size().median():.0f} "
          f"max={df.groupby('dongle_id').size().max()} "
          f"({100*df.groupby('dongle_id').size().max()/len(df):.1f}% of data)")


if __name__ == "__main__":
    main()
