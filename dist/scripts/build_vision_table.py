"""
Vision stage 2 - aggregate per-frame YOLOv8n scalars onto the sliding forecast
windows, then merge them onto `forecast_table`.

For every (clip, decision_t_s) row, features are aggregated over the exact closed
window `t_rel in [s - W, s]` (W read from `forecast_schema.json`). Every window
ends at `s <= 0`, so vision features can never see the takeover or the
`[0, +5] s` target window; this is asserted per row.

`vis_available == 0` rows (clips with no decoded frames) keep all vision features
NaN - they are never imputed here. `model.encode()` medians them inside each
training fold, which is the correct place.

Writes:
    data/derived/vision_table.parquet          (vis_* per forecast row, 5-key)
    data/derived/forecast_table_vision.parquet (forecast_table + vis_*)
    data/derived/forecast_schema_vision.json

Usage:
    python scripts/build_vision_table.py
"""
from __future__ import annotations

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

GRID_CELLS = [(r, c) for r in range(3) for c in range(3)]
GRID_VIS = ([f"vis_grid_cnt_mean_r{r}c{c}" for r, c in GRID_CELLS]
            + [f"vis_grid_area_mean_r{r}c{c}" for r, c in GRID_CELLS])
TL_VIS = ["vis_tl_red_any", "vis_tl_amber_any", "vis_tl_green_any",
          "vis_tl_unknown_any", "vis_tl_red_frac", "vis_tl_amber_frac",
          "vis_tl_green_frac"]

VISION_FEATURES = [
    "vis_available",
    "vis_n_frames",
    "vis_lead_available",
    "vis_lead_present_rate",
    "vis_lead_area_frac_mean",
    "vis_lead_area_frac_max",
    "vis_lead_area_frac_p90",
    "vis_lead_area_frac_last",
    "vis_lead_area_growth_per_s",
    "vis_lead_area_delta",
    "vis_lead_cx_offset_mean",
    "vis_lead_cx_offset_last",
    "vis_lead_abs_cx_offset_max",
    "vis_lead_y2_norm_mean",
    "vis_lead_y2_norm_max",
    "vis_lead_conf_mean",
    "vis_lead_drop_count",
    "vis_lead_longest_cont_s",
    "vis_vehicle_count_mean",
    "vis_vehicle_count_max",
    "vis_vehicle_count_delta",
    "vis_corridor_vehicle_count_mean",
    "vis_corridor_vehicle_count_max",
    "vis_corridor_clear",
    "vis_any_vehicle",
    "vis_person_max",
    "vis_bicycle_max",
    "vis_traffic_light_max",
    "vis_stop_sign_max",
    "vis_any_vru",
    "vis_any_traffic_light",
    "vis_brightness_mean",
    "vis_motion_mean",
] + GRID_VIS + TL_VIS


def _finite(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _fmax(x):
    f = _finite(x)
    return float(f.max()) if f.size else NAN


def _fmean(x):
    f = _finite(x)
    return float(f.mean()) if f.size else NAN


def _fq(x, p):
    f = _finite(x)
    return float(np.percentile(f, p)) if f.size else NAN


def _first_finite(x):
    x = np.asarray(x, float)
    idx = np.where(np.isfinite(x))[0]
    return float(x[idx[0]]) if idx.size else NAN


def _last_finite(x):
    x = np.asarray(x, float)
    idx = np.where(np.isfinite(x))[0]
    return float(x[idx[-1]]) if idx.size else NAN


def _slope(t, v):
    t = np.asarray(t, float)
    v = np.asarray(v, float)
    ok = np.isfinite(t) & np.isfinite(v)
    if ok.sum() < 3:
        return NAN
    tt, vv = t[ok], v[ok]
    if float(np.ptp(tt)) <= EPS:
        return NAN
    return float(np.polyfit(tt, vv, 1)[0])


def _fmean0(x):
    f = _finite(x)
    return float(f.mean()) if f.size else 0.0


def _fmax0(x):
    f = _finite(x)
    return float(f.max()) if f.size else 0.0


def _fq0(x, p):
    f = _finite(x)
    return float(np.percentile(f, p)) if f.size else 0.0


def _last_finite0(x):
    x = np.asarray(x, float)
    idx = np.where(np.isfinite(x))[0]
    return float(x[idx[-1]]) if idx.size else 0.0


def _slope0(t, v):
    s = _slope(t, v)
    return s if np.isfinite(s) else 0.0


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


def window_vis(a: dict, lo: float, hi: float) -> dict:
    """Aggregate one closed window [lo, hi] of a clip's per-frame arrays.

    Zero-inflated encoding: "no lead in this window" is a real observation, not a
    missing value. Every feature is defined on every window - lead geometry falls
    back to 0.0 (nothing there) and `vis_lead_available` records whether a lead was
    present at all, so `model.encode()` never median-imputes a lead that did not
    exist.
    """
    t = a["t_rel"]
    m = (t >= lo - EPS) & (t <= hi + EPS)
    f = {k: 0.0 for k in VISION_FEATURES}
    f["vis_n_frames"] = int(m.sum())
    if m.sum() == 0:
        return f

    tt = t[m]

    def g(k):
        return a[k][m]

    lead = np.nan_to_num(g("lead_present"), nan=0.0) > 0.5
    area = np.where(lead, g("lead_area_frac"), NAN)
    cx = np.where(lead, g("lead_cx_offset"), NAN)
    y2 = np.where(lead, g("lead_y2_norm"), NAN)
    conf = np.where(lead, g("lead_conf"), NAN)

    f["vis_lead_available"] = 1.0 if lead.any() else 0.0
    f["vis_lead_present_rate"] = float(lead.mean()) if lead.size else 0.0
    f["vis_lead_area_frac_mean"] = _fmean0(area)
    f["vis_lead_area_frac_max"] = _fmax0(area)
    f["vis_lead_area_frac_p90"] = _fq0(area, 90)
    f["vis_lead_area_frac_last"] = _last_finite0(area)
    f["vis_lead_area_growth_per_s"] = _slope0(tt, area)
    first, last = _first_finite(area), _last_finite(area)
    f["vis_lead_area_delta"] = (last - first) if (
        np.isfinite(first) and np.isfinite(last)) else 0.0
    f["vis_lead_cx_offset_mean"] = _fmean0(cx)
    f["vis_lead_cx_offset_last"] = _last_finite0(cx)
    f["vis_lead_abs_cx_offset_max"] = _fmax0(np.abs(cx))
    f["vis_lead_y2_norm_mean"] = _fmean0(y2)
    f["vis_lead_y2_norm_max"] = _fmax0(y2)
    f["vis_lead_conf_mean"] = _fmean0(conf)
    f["vis_lead_drop_count"] = int(np.sum(lead[:-1] & ~lead[1:]))
    f["vis_lead_longest_cont_s"] = _longest_true_run(lead, tt)

    veh = g("n_vehicles")
    f["vis_vehicle_count_mean"] = _fmean0(veh)
    f["vis_vehicle_count_max"] = _fmax0(veh)
    vfirst, vlast = _first_finite(veh), _last_finite(veh)
    f["vis_vehicle_count_delta"] = (vlast - vfirst) if (
        np.isfinite(vfirst) and np.isfinite(vlast)) else 0.0
    corr = g("n_corridor_vehicles")
    f["vis_corridor_vehicle_count_mean"] = _fmean0(corr)
    f["vis_corridor_vehicle_count_max"] = _fmax0(corr)
    f["vis_corridor_clear"] = 1.0 if _fmax0(corr) <= 0 else 0.0
    f["vis_any_vehicle"] = 1.0 if _fmax0(veh) > 0 else 0.0

    p_max, b_max = _fmax0(g("n_person")), _fmax0(g("n_bicycle"))
    tl_max, ss_max = _fmax0(g("n_traffic_light")), _fmax0(g("n_stop_sign"))
    f["vis_person_max"] = p_max
    f["vis_bicycle_max"] = b_max
    f["vis_traffic_light_max"] = tl_max
    f["vis_stop_sign_max"] = ss_max
    f["vis_any_vru"] = float((p_max > 0) or (b_max > 0))
    f["vis_any_traffic_light"] = float(tl_max > 0)

    f["vis_brightness_mean"] = _fmean0(g("brightness_mean"))
    f["vis_motion_mean"] = _fmean0(g("motion_mean"))

    for r, c in GRID_CELLS:
        f[f"vis_grid_cnt_mean_r{r}c{c}"] = _fmean0(g(f"n_veh_r{r}c{c}"))
        f[f"vis_grid_area_mean_r{r}c{c}"] = _fmean0(g(f"area_veh_r{r}c{c}"))

    for state in ("red", "amber", "green", "unknown"):
        cnt = np.nan_to_num(g(f"n_tl_{state}"), nan=0.0)
        f[f"vis_tl_{state}_any"] = 1.0 if cnt.max() > 0 else 0.0
        if state != "unknown":
            f[f"vis_tl_{state}_frac"] = float((cnt > 0).mean())
    return f


def clip_arrays(sub: pd.DataFrame) -> dict:
    sub = sub.sort_values("t_rel")
    cols = ["t_rel", "lead_present", "lead_area_frac", "lead_cx_offset", "lead_y2_norm",
            "lead_conf", "n_vehicles", "n_corridor_vehicles", "n_person", "n_bicycle",
            "n_traffic_light", "n_stop_sign", "brightness_mean", "motion_mean"]
    cols += [f"n_veh_r{r}c{c}" for r, c in GRID_CELLS]
    cols += [f"area_veh_r{r}c{c}" for r, c in GRID_CELLS]
    cols += ["n_tl_red", "n_tl_amber", "n_tl_green", "n_tl_unknown"]
    return {c: sub[c].to_numpy(float) for c in cols}


def main() -> None:
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

    frames = pd.read_parquet(C.VISION_FRAMES)
    available = set(map(tuple, frames[KEY_COLS].drop_duplicates().to_numpy()))
    clips = list(map(tuple, fkeys.to_numpy()))
    print(f"[vision-table] W={W}s  {len(clips)} clips  "
          f"{len(available)} with decoded frames  {len(times)} windows/clip")

    records = []
    max_t_used = -np.inf
    for key in clips:
        has = key in available
        sub = frames[(frames[KEY_COLS] == key).all(axis=1)] if has else None
        arrays = clip_arrays(sub) if has else None
        for s in times:
            assert s <= EPS, "decision time must not be after the takeover"
            rec = dict(zip(KEY_COLS, key))
            rec[DEC_COL] = float(s)
            if has and arrays is not None:
                lo, hi = s - W, s
                rec.update(window_vis(arrays, lo, hi))
                rec["vis_available"] = 1.0
                used = arrays["t_rel"][(arrays["t_rel"] >= lo - EPS)
                                       & (arrays["t_rel"] <= hi + EPS)]
                assert used.size == 0 or float(used.max()) <= s + EPS, "window sees the future"
                if used.size:
                    max_t_used = max(max_t_used, float(used.max()))
            else:
                rec.update({k: 0.0 for k in VISION_FEATURES})
                rec["vis_available"] = 0.0
            records.append(rec)

    vt = pd.DataFrame(records)
    vt = vt[KEY_COLS + [DEC_COL] + VISION_FEATURES]

    assert len(vt) == n_expected, f"vision_table rows {len(vt)} != {n_expected}"
    assert not vt.duplicated(KEY_COLS + [DEC_COL]).any(), "duplicate vision_table keys"
    assert max_t_used <= -0.0 + EPS, f"leakage: used frame at t_rel={max_t_used}"

    out = forecast.merge(vt, on=KEY_COLS + [DEC_COL], how="left", validate="one_to_one")
    assert len(out) == len(forecast), "merge inflated the row count"
    assert out[VISION_FEATURES].notna().all().all(), \
        "vision block must be fully defined (zero-inflated encoding, no NaN)"
    assert set(out["vis_available"].unique()) <= {0.0, 1.0}, "vis_available must be 0/1"
    assert set(out["vis_lead_available"].unique()) <= {0.0, 1.0}, \
        "vis_lead_available must be 0/1"
    assert (out.loc[out["vis_available"] == 0.0, "vis_n_frames"] == 0).all(), \
        "unavailable clips must have zero decoded frames"
    assert (out.loc[out["vis_lead_available"] == 0.0, "vis_lead_conf_mean"] == 0.0).all(), \
        "no-lead windows must use the zero sentinel"

    vt.to_parquet(C.VISION_TABLE, index=False)
    out.to_parquet(C.FORECAST_TABLE_VISION, index=False)

    feats = [c for c in schema["features_numeric"]] + VISION_FEATURES
    can_cols = set(schema["features_numeric"])
    assert not (set(VISION_FEATURES) & can_cols), "vis_* collides with a CAN feature"
    assert all(c in out.columns for c in feats), "missing feature after merge"

    lead_g = [c for c in VISION_FEATURES
              if c.startswith("vis_lead") or c in ("vis_available", "vis_n_frames")]
    scene_g = [c for c in VISION_FEATURES
               if c not in lead_g and c not in GRID_VIS and c not in TL_VIS]
    groups = {"lead": lead_g, "scene": scene_g, "grid": GRID_VIS, "light": TL_VIS}
    assert sorted(sum(groups.values(), [])) == sorted(VISION_FEATURES), \
        "vision groups must partition the feature set"

    schema_out = {
        **schema,
        "n_rows": int(len(out)),
        "features_numeric": feats,
        "vision": {
            "backbone": C.YOLO_MODEL,
            "conf": C.YOLO_CONF,
            "imgsz": C.YOLO_IMGSZ,
            "sampling_fps": C.VISION_FPS,
            "frames_table": C.VISION_FRAMES.name,
            "alignment": "t_rel = frame_idx / CAP_PROP_FPS - 10.0 (clip-relative video time; "
                         "takeover at t_rel = 0, same axis as build_clip_telemetry t)",
            "window": "closed [decision_t_s - W, decision_t_s], W from this schema",
            "lead_rule": "vehicle box centred in the ego-lane corridor [0.25W, 0.75W] with "
                         "the largest area; the previous frame's lead is retained when "
                         "IoU >= 0.1 (temporal continuity). No near-field y2 gate, so "
                         "distant leads are included.",
            "clips_with_frames": int(len(available)),
            "clips_total": int(len(clips)),
            "leakage_assert": "max frame t_rel used <= decision_t_s (<= 0) for every row",
            "encoding": (
                "zero-inflated: a window with no detected lead is a real observation, not "
                "missing data. Lead geometry is 0.0 when absent and vis_lead_available "
                "(0/1) records presence, so no median imputation of a non-existent lead "
                "can happen. All vis_* columns are defined on every row."
            ),
            "grid": "3x3 occupancy (rows far->near by box centre, cols left->right): "
                    "per-cell vehicle count and summed box area, averaged over the window. "
                    "Defined for every cell (zero when empty).",
            "light": "traffic-light state read from the colour of each detected light box "
                     "(red/amber/green hue majority, conservative 'unknown'); window features "
                     "are any/frac per state, zero-inflated.",
            "groups": groups,
            "imputation": "none required for vis_* (fully defined); model.encode() still "
                          "medians the CAN columns inside the train fold as before",
        },
        "excluded_from_features": sorted(set(schema.get("excluded_from_features", []))
                                         | {DEC_COL, "window_s", "stride_s", "lead_s"}),
    }
    C.FORECAST_SCHEMA_VISION.write_text(json.dumps(schema_out, indent=2, default=str),
                                        encoding="utf-8")
    rate = float((out["vis_available"] == 1.0).mean())
    print(f"[vision-table] {len(out):,} rows x {len(VISION_FEATURES)} vis features  "
          f"(available on {rate:.1%} of rows) -> {C.FORECAST_TABLE_VISION}")
    print(f"[vision-table] vis_n_frames mean={out['vis_n_frames'].mean():.1f} "
          f"(expected ~{W * C.VISION_FPS:.0f}); "
          f"lead-present rate={out['vis_lead_present_rate'].mean():.3f}")
    dead = [c for c in VISION_FEATURES if out[c].isna().all()]
    print(f"[vision-table] all-NaN feature columns: {dead or 'none'}")


if __name__ == "__main__":
    main()
