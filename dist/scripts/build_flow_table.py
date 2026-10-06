"""
Flow stage 2 - aggregate per-frame ego-motion-compensated flow scalars onto the
sliding forecast windows, then merge them onto `forecast_table`.

For every (clip, decision_t_s) row, features are aggregated over the exact closed
window `t_rel in [s - W, s]` (W read from `forecast_schema.json`). Every window ends
at `s <= 0`, so flow features can never see the takeover or the `[0, +5] s` target
window; this is asserted per row.

`flow_available == 0` rows (clips with no extracted frames) keep all flow features at
the zero sentinel - they are an explicit "nothing detected" observation, exactly like
the `vis_*` block. `model.encode()` therefore never median-imputes a non-existent
event.

Writes:
    data/derived/flow_table.parquet          (flow_* per forecast row, 5-key)
    data/derived/forecast_table_flow.parquet (forecast_table + flow_*)
    data/derived/forecast_schema_flow.json

Usage:
    python scripts/build_flow_table.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C

KEY_COLS = ["car_model", "driver", "route", "clip_id_pub"]
DEC_COL = "decision_t_s"
EPS = 1e-9
NAN = float("nan")
CLIP_START_S = -10.0

FLOW_BASE = ["flow_available", "flow_n_samples"]
FLOW_RES = [
    "flow_res_mag_mean_mean",
    "flow_res_mag_mean_max",
    "flow_res_mag_mean_p90",
    "flow_res_corridor_mag_mean",
    "flow_res_corridor_mag_max",
    "flow_res_lat_abs_max",
    "flow_res_lat_mean",
    "flow_res_long_abs_max",
    "flow_div_corridor_mean",
    "flow_div_corridor_max",
]
FLOW_IMO = [
    "flow_imo_count_mean",
    "flow_imo_count_max",
    "flow_imo_area_frac_max",
    "flow_imo_available",
]
FLOW_ONSET = [
    "flow_onset_count",
    "flow_onset_corridor_count",
    "flow_time_since_last_onset",
    "flow_onset_within_0p5s",
]
FLOW_GLOBAL = [
    "flow_global_scale_mean",
    "flow_global_rot_std",
]
FLOW_FEATURES = FLOW_BASE + FLOW_RES + FLOW_IMO + FLOW_ONSET + FLOW_GLOBAL

ONSET_RECENT_S = 0.5


# --------------------------------------------------------------------------- #
# nan-safe helpers (mirrors build_vision_table / build_forecast_table)
# --------------------------------------------------------------------------- #
def _finite(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _fmax(x):
    f = _finite(x)
    return float(f.max()) if f.size else NAN


def _fmean0(x):
    f = _finite(x)
    return float(f.mean()) if f.size else 0.0


def _fmax0(x):
    f = _finite(x)
    return float(f.max()) if f.size else 0.0


def _fq0(x, p):
    f = _finite(x)
    return float(np.percentile(f, p)) if f.size else 0.0


def _fstd0(x):
    f = _finite(x)
    return float(f.std()) if f.size else 0.0


def _longest_true_run(flag: np.ndarray, tt: np.ndarray) -> float:
    runs, start = [], None
    for i, s in enumerate(flag):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append(tt[i - 1] - tt[start])
            start = None
    if start is not None and tt.size:
        runs.append(tt[-1] - tt[start])
    return float(max(runs)) if runs else 0.0


def window_flow(a: dict, lo: float, hi: float, s: float) -> dict:
    """Aggregate one closed window [lo, hi] of a clip's per-frame arrays.

    Zero-inflated encoding: a window with no flow-valid frame and no IMO is a real
    observation, not missing data. Every feature is defined on every window -
    residual/divergence statistics fall back to 0.0, the IMO count to 0, and
    `flow_imo_available` records whether any IMO was present at all.
    """
    t = a["t_rel"]
    m = (t >= lo - EPS) & (t <= hi + EPS)
    f = {k: 0.0 for k in FLOW_FEATURES}
    f["flow_n_samples"] = int(m.sum())
    if m.sum() == 0:
        return f

    tt = t[m]

    def g(k):
        return a[k][m]

    valid = np.nan_to_num(g("flow_valid"), nan=0.0) > 0.5

    res = np.where(valid, g("flow_res_mag_mean"), NAN)
    f["flow_res_mag_mean_mean"] = _fmean0(res)
    f["flow_res_mag_mean_max"] = _fmax0(res)
    f["flow_res_mag_mean_p90"] = _fq0(res, 90)

    corr = np.where(valid, g("flow_res_corridor_mag_mean"), NAN)
    f["flow_res_corridor_mag_mean"] = _fmean0(corr)
    f["flow_res_corridor_mag_max"] = _fmax0(corr)

    lat = np.where(valid, g("flow_res_lat_mean"), NAN)
    f["flow_res_lat_mean"] = _fmean0(lat)
    f["flow_res_lat_abs_max"] = _fmax0(np.abs(lat))

    lon = np.where(valid, g("flow_res_long_mean"), NAN)
    f["flow_res_long_abs_max"] = _fmax0(np.abs(lon))

    divc = np.where(valid, g("flow_div_corridor_mean"), NAN)
    f["flow_div_corridor_mean"] = _fmean0(divc)
    f["flow_div_corridor_max"] = _fmax0(divc)

    imo = np.where(valid, g("flow_imo_count"), 0.0)
    f["flow_imo_count_mean"] = _fmean0(imo)
    f["flow_imo_count_max"] = _fmax0(imo)
    f["flow_imo_area_frac_max"] = _fmax0(g("flow_imo_area_frac"))
    f["flow_imo_available"] = 1.0 if f["flow_imo_count_max"] > 0 else 0.0

    ons = np.nan_to_num(g("flow_onset"), nan=0.0)
    onsc = np.nan_to_num(g("flow_onset_corridor"), nan=0.0)
    f["flow_onset_count"] = float(ons.sum())
    f["flow_onset_corridor_count"] = float(onsc.sum())
    recent = (tt >= s - ONSET_RECENT_S - EPS) & (ons > 0.5)
    f["flow_onset_within_0p5s"] = 1.0 if recent.any() else 0.0

    # time since the last onset anywhere up to s (>= 0 and bounded by the clip);
    # `flow_onset_count == 0` marks "no onset inside this window".
    allm = t <= s + EPS
    prior = np.where(allm, np.nan_to_num(a["flow_onset"], nan=0.0), 0.0) > 0.5
    idx = np.where(prior)[0]
    f["flow_time_since_last_onset"] = (float(s - t[idx[-1]]) if idx.size
                                       else float(s - CLIP_START_S))

    sc = np.where(valid, g("flow_global_scale"), NAN)
    f["flow_global_scale_mean"] = _fmean0(sc)
    rot = np.where(valid, g("flow_global_rot"), NAN)
    f["flow_global_rot_std"] = _fstd0(rot)
    return f


def clip_arrays(sub: pd.DataFrame) -> dict:
    sub = sub.sort_values("t_rel")
    cols = ["t_rel", "flow_valid", "flow_res_mag_mean", "flow_res_corridor_mag_mean",
            "flow_res_lat_mean", "flow_res_long_mean", "flow_div_corridor_mean",
            "flow_imo_count", "flow_imo_area_frac", "flow_onset",
            "flow_onset_corridor", "flow_global_scale", "flow_global_rot"]
    return {c: sub[c].to_numpy(float) for c in cols}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default=C.FLOW_METHOD, choices=["dis", "raft"],
                    help="optical-flow method actually used by extract_flow_frames.py")
    args = ap.parse_args()

    forecast = pd.read_parquet(C.FORECAST_TABLE)
    schema = json.loads(C.FORECAST_SCHEMA.read_text(encoding="utf-8"))
    W = float(schema["sliding"]["window_s"])
    times = [float(x) for x in schema["sliding"]["decision_times_s"]]
    n_expected = int(schema["n_rows"])

    clip_index = pd.read_parquet(C.CLIP_INDEX, columns=KEY_COLS)
    assert not clip_index.duplicated(KEY_COLS).any(), "clip_index keys are not unique"
    fkeys = forecast[KEY_COLS].drop_duplicates()
    missing = fkeys.merge(clip_index, on=KEY_COLS, how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    assert missing.empty, f"{len(missing)} forecast clips missing from clip_index"

    frames = pd.read_parquet(C.FLOW_FRAMES)
    available = set(map(tuple, frames[KEY_COLS].drop_duplicates().to_numpy()))
    clips = list(map(tuple, fkeys.to_numpy()))
    print(f"[flow-table] W={W}s  {len(clips)} clips  "
          f"{len(available)} with flow frames  {len(times)} windows/clip")

    by_key = {k: sub for k, sub in frames.groupby(KEY_COLS)}

    records = []
    max_t_used = -np.inf
    for key in clips:
        has = key in available
        arrays = clip_arrays(by_key[key]) if has else None
        for s in times:
            assert s <= EPS, "decision time must not be after the takeover"
            rec = dict(zip(KEY_COLS, key))
            rec[DEC_COL] = float(s)
            if has and arrays is not None:
                lo, hi = s - W, s
                rec.update(window_flow(arrays, lo, hi, s))
                rec["flow_available"] = 1.0
                used = arrays["t_rel"][(arrays["t_rel"] >= lo - EPS)
                                       & (arrays["t_rel"] <= hi + EPS)]
                assert used.size == 0 or float(used.max()) <= s + EPS, "window sees the future"
                if used.size:
                    max_t_used = max(max_t_used, float(used.max()))
            else:
                rec.update({k: 0.0 for k in FLOW_FEATURES})
                rec["flow_available"] = 0.0
            records.append(rec)

    ft = pd.DataFrame(records)
    ft = ft[KEY_COLS + [DEC_COL] + FLOW_FEATURES]

    assert len(ft) == n_expected, f"flow_table rows {len(ft)} != {n_expected}"
    assert not ft.duplicated(KEY_COLS + [DEC_COL]).any(), "duplicate flow_table keys"
    assert max_t_used <= 0.0 + EPS, f"leakage: used frame at t_rel={max_t_used}"

    out = forecast.merge(ft, on=KEY_COLS + [DEC_COL], how="left", validate="one_to_one")
    assert len(out) == len(forecast), "merge inflated the row count"
    assert out[FLOW_FEATURES].notna().all().all(), \
        "flow block must be fully defined (zero-inflated encoding, no NaN)"
    assert set(out["flow_available"].unique()) <= {0.0, 1.0}, "flow_available must be 0/1"
    assert set(out["flow_imo_available"].unique()) <= {0.0, 1.0}, \
        "flow_imo_available must be 0/1"
    assert (out.loc[out["flow_available"] == 0.0, "flow_n_samples"] == 0).all(), \
        "unavailable clips must have zero flow samples"
    assert (out.loc[out["flow_imo_available"] == 0.0, "flow_imo_count_max"] == 0.0).all(), \
        "no-IMO windows must use the zero sentinel"

    ft.to_parquet(C.FLOW_TABLE, index=False)
    out.to_parquet(C.FORECAST_TABLE_FLOW, index=False)

    feats = [c for c in schema["features_numeric"]] + FLOW_FEATURES
    can_cols = set(schema["features_numeric"])
    assert not (set(FLOW_FEATURES) & can_cols), "flow_* collides with a CAN feature"
    assert all(c in out.columns for c in feats), "missing feature after merge"

    groups = {
        "flow_res": FLOW_RES,
        "flow_imo": FLOW_IMO,
        "flow_onset": FLOW_ONSET,
        "flow_global": FLOW_GLOBAL,
    }
    assert sorted(sum(groups.values(), []) + FLOW_BASE) == sorted(FLOW_FEATURES), \
        "flow groups must partition the feature set"

    schema_out = {
        **schema,
        "n_rows": int(len(out)),
        "features_numeric": feats,
        "flow": {
            "method": args.method,
            "sampling_fps": C.FLOW_HZ,
            "frames_table": C.FLOW_FRAMES.name,
            "alignment": "t_rel = frame_idx / CAP_PROP_FPS - 10.0 (clip-relative video "
                         "time; takeover at t_rel = 0, same axis as build_clip_telemetry t)",
            "window": "closed [decision_t_s - W, decision_t_s], W from this schema",
            "global_fit": "cv2.estimateAffinePartial2D (RANSAC similarity) on an "
                          "every-8th-pixel grid, masking the ego hood (bottom "
                          f"{C.FLOW_HOOD_FRAC:.0%}) and a 4-px border; residual = "
                          "observed - global-model flow on all pixels",
            "imo_rule": "|residual| > max(1.5 px, 3 x median|residual|), 8-connected "
                        "components with area >= 0.1% of the frame",
            "onset_rule": "rising edge of the residual activity vs max(1.8 x rolling "
                          "median over 2 s, 1.0 px); heuristic, tuned on the smoke run",
            "divergence": "du/dx + dv/dy of the residual field (positive = expanding)",
            "clips_with_frames": int(len(available)),
            "clips_total": int(len(clips)),
            "leakage_assert": "max frame t_rel used <= decision_t_s (<= 0) for every row",
            "encoding": (
                "zero-inflated: a window with no flow-valid frame or no detected IMO is "
                "a real observation. Residual/divergence statistics are 0.0 when absent, "
                "flow_available / flow_imo_available (0/1) record presence, so no median "
                "imputation of a non-existent event can happen. All flow_* columns are "
                "defined on every row."
            ),
            "groups": groups,
            "imputation": "none required for flow_* (fully defined); model.encode() still "
                          "medians the CAN columns inside the train fold as before",
        },
        "excluded_from_features": sorted(set(schema.get("excluded_from_features", []))
                                         | {DEC_COL, "window_s", "stride_s", "lead_s"}),
    }
    C.FORECAST_SCHEMA_FLOW.write_text(json.dumps(schema_out, indent=2, default=str),
                                      encoding="utf-8")
    rate = float((out["flow_available"] == 1.0).mean())
    print(f"[flow-table] {len(out):,} rows x {len(FLOW_FEATURES)} flow features  "
          f"(available on {rate:.1%} of rows) -> {C.FORECAST_TABLE_FLOW}")
    print(f"[flow-table] flow_n_samples mean={out['flow_n_samples'].mean():.1f} "
          f"(expected ~{W * C.FLOW_HZ:.0f}); "
          f"IMO-present windows={float(out['flow_imo_available'].mean()):.1%}; "
          f"onsets total={int(out['flow_onset_count'].sum())} "
          f"(corridor {int(out['flow_onset_corridor_count'].sum())})")
    dead = [c for c in FLOW_FEATURES if out[c].isna().all()]
    print(f"[flow-table] all-NaN feature columns: {dead or 'none'}")


if __name__ == "__main__":
    main()
