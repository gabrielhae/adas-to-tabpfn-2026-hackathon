"""
M4-minimal - forecast training data from a SLIDING WINDOW.

One training row per (clip, decision time):

    decision time s  ->  feature window [s - W, s]      (fixed length W)
                         lead time      delta = -s       (seconds before takeover)
    target           ->  post-window outcomes [0, +5]    (same for every row of a clip)

So a fixed-length window slides across the pre-takeover timeline; each position is a
separate training example for the SAME future outcome, at a known lead time. The
closest decision is at s = -min_lead, so features never touch the takeover (no t=0
static, no run-up into [0, +5]).

Rows from one clip are correlated; CV still groups by dongle_id, which subsumes clips
(all of a driver's clips - and all their windows - stay in one fold).

Usage:
    python scripts/build_forecast_table.py            # W=3.5s, stride=0.5s, windows end 0 .. -6.5s
    python scripts/build_forecast_table.py --min-lead 0.5   # stop just before onset (no t=0 sample)
    python scripts/build_forecast_table.py --window 5 --stride 0.5 --min-lead 2
    python scripts/build_forecast_table.py --min-lead 3 --max-lead 3   # single fixed lead
    python scripts/build_forecast_table.py --limit 5  # smoke test

NOTE on the last window: with --min-lead 0 the closest window ends exactly at t=0, so its
static scalars are sampled at the onset and it shares the single sample at t=0 with the
target window. Rows with lead 0 are therefore "nowcast"/onset-adjacent, not early warning;
rows with lead >= 0.5 are strictly before the takeover.

Writes:
    data/derived/forecast_table.parquet
    data/derived/forecast_schema.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C
from adas_to.telemetry import build_clip_telemetry

HZ = C.RESAMPLE_HZ
DT = 1.0 / HZ
SMOOTH_S = 0.3
EPS = 1e-9
NAN = float("nan")
CLIP_START_S = -C.CLIP_SECONDS / 2.0          # -10.0 (relative axis starts here)

CLOSING_MIN = C.ANALYSIS["closing_min_mps"]
DREL_MIN = C.ANALYSIS["drel_min_m"]
TTC_CAP = C.ANALYSIS["ttc_cap_s"]
MIN_V_THW = C.ANALYSIS["min_speed_thw_mps"]

KEY_COLS = ["car_model", "driver", "route", "clip_id_pub"]
META_COLS = ["decision_t_s", "lead_s", "window_s", "stride_s"]


# --------------------------------------------------------------------------- #
# nan-safe helpers
# --------------------------------------------------------------------------- #
def _finite(x) -> np.ndarray:
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def _fmax(x):
    f = _finite(x)
    return float(f.max()) if f.size else NAN


def _fmin(x):
    f = _finite(x)
    return float(f.min()) if f.size else NAN


def _fmean(x):
    f = _finite(x)
    return float(f.mean()) if f.size else NAN


def _fq(x, p):
    f = _finite(x)
    return float(np.percentile(f, p)) if f.size else NAN


def _last_finite(x):
    x = np.asarray(x, float)
    idx = np.where(np.isfinite(x))[0]
    return float(x[idx[-1]]) if idx.size else NAN


def _t_of(arr, t, which):
    a = np.asarray(arr, float)
    idx = np.where(np.isfinite(a))[0]
    if not idx.size:
        return NAN
    return float(t[idx[which(a[idx])]])


def _smooth(v: np.ndarray) -> np.ndarray:
    n = len(v)
    if n < 5:
        return v.copy()
    win = max(3, int(round(SMOOTH_S * HZ)))
    if win % 2 == 0:
        win += 1
    win = min(win, n if n % 2 == 1 else n - 1)
    poly = min(2, win - 1)
    try:
        return savgol_filter(v, win, poly)
    except Exception:
        return pd.Series(v).rolling(3, center=True, min_periods=1).mean().values


# --------------------------------------------------------------------------- #
# load once per clip, then slice any window
# --------------------------------------------------------------------------- #
def load_series(clip_dir: Path) -> tuple[np.ndarray, dict]:
    tel = build_clip_telemetry(Path(clip_dir))
    t = np.asarray(tel["t"], float)
    series = {k: np.asarray(v, float) for k, v in tel["series"].items()}
    return t, series


def window_features(series: dict, t: np.ndarray, lo: float, hi: float) -> dict:
    """Aggregate features over the closed window [lo, hi] (seconds relative to event)."""
    n = len(t)
    m = (t >= lo - EPS) & (t <= hi + EPS)
    if m.sum() < 2:
        raise ValueError("window too short")
    tt = t[m]

    def gm(k):
        v = series.get(k)
        return np.full(n, NAN)[m] if v is None else v[m]

    vEgo, aEgo = gm("vEgo"), gm("aEgo")
    sa, st = gm("steeringAngleDeg"), gm("steeringTorque")
    drel, vrel, lead = gm("leadOne.dRel"), gm("leadOne.vRel"), gm("leadOne.status")
    curv, dcurv = gm("curvature"), gm("desiredCurvature")
    lp, rp = gm("laneLineMeta.leftProb"), gm("laneLineMeta.rightProb")
    fcw = gm("fcw")

    f: dict = {}

    # static scalars = last known value AT the decision time (hi), never at t=0
    f["speed_mps"] = _last_finite(vEgo)
    f["accel_mps2"] = _last_finite(aEgo)
    f["steer_angle_deg"] = _last_finite(sa)
    f["steer_torque"] = _last_finite(st)
    f["cruise_speed_kmh"] = _last_finite(gm("cruiseState.speed"))

    # ---- longitudinal ----
    f["pre_speed_mean_mps"] = _fmean(vEgo)
    vf = _finite(vEgo)
    f["pre_speed_delta_mps"] = float(vf[-1] - vf[0]) if vf.size >= 2 else NAN
    f["pre_min_accel_mps2"] = _fmin(aEgo)
    f["pre_max_accel_mps2"] = _fmax(aEgo)
    f["pre_accel_p5_mps2"] = _fq(aEgo, 5)
    f["pre_accel_p50_mps2"] = _fq(aEgo, 50)
    f["pre_accel_p95_mps2"] = _fq(aEgo, 95)
    f["pre_time_of_peak_decel_s"] = _t_of(aEgo, tt, np.argmin)

    aok = np.where(np.isfinite(aEgo))[0]
    if aok.size >= 5:
        av, tv = aEgo[aok], tt[aok]
        jerk = np.abs(np.diff(_smooth(av)) / DT)
        tj = (tv[:-1] + tv[1:]) / 2
        f["pre_max_abs_jerk_mps3"] = float(jerk.max())
        f["pre_jerk_p50_mps3"] = float(np.percentile(jerk, 50))
        f["pre_jerk_p95_mps3"] = float(np.percentile(jerk, 95))
        f["pre_time_of_peak_jerk_s"] = float(tj[jerk.argmax()])
    else:
        f["pre_max_abs_jerk_mps3"] = NAN
        f["pre_jerk_p50_mps3"] = NAN
        f["pre_jerk_p95_mps3"] = NAN
        f["pre_time_of_peak_jerk_s"] = NAN

    # ---- lateral ----
    f["pre_max_abs_steer_angle_deg"] = _fmax(np.abs(sa))
    f["pre_max_abs_steer_torque"] = _fmax(np.abs(st))
    f["pre_max_abs_curvature"] = _fmax(np.abs(curv))
    f["pre_max_abs_desired_curvature"] = _fmax(np.abs(dcurv))

    sok = np.where(np.isfinite(sa))[0]
    if sok.size >= 5:
        sv, tv = sa[sok], tt[sok]
        rate = np.abs(np.diff(_smooth(sv)) / DT)
        tr = (tv[:-1] + tv[1:]) / 2
        f["pre_steer_rate_max_deg_per_s"] = float(rate.max())
        f["pre_steer_rate_p95_deg_per_s"] = float(np.percentile(rate, 95))
        f["pre_time_of_peak_steer_rate_s"] = float(tr[rate.argmax()])
    else:
        f["pre_steer_rate_max_deg_per_s"] = NAN
        f["pre_steer_rate_p95_deg_per_s"] = NAN
        f["pre_time_of_peak_steer_rate_s"] = NAN

    # ---- lead / safety margins (lead-present only) ----
    leadok = np.nan_to_num(lead, nan=0.0) > 0.5
    f["pre_n_lead_samples"] = int(leadok.sum())
    f["pre_lead_present_rate"] = float(leadok.mean()) if len(leadok) else NAN
    f["pre_lead_drop_count"] = int(np.sum(leadok[:-1] & ~leadok[1:]))
    runs, start = [], None
    for i, s in enumerate(leadok):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append(tt[i - 1] - tt[start])
            start = None
    if start is not None and tt.size:
        runs.append(tt[-1] - tt[start])
    f["pre_longest_cont_lead_s"] = float(max(runs)) if runs else 0.0

    drel_l = np.where(leadok, drel, NAN)
    vrel_l = np.where(leadok, vrel, NAN)
    f["pre_min_drel_m"] = _fmin(drel_l)
    f["pre_p5_drel_m"] = _fq(drel_l, 5)

    with np.errstate(invalid="ignore", divide="ignore"):
        thw = np.where(vEgo > MIN_V_THW, drel_l / vEgo, NAN)
        thw = np.where(np.isfinite(thw) & (thw > 0), thw, NAN)
        f["pre_thw_min_s"] = _fmin(thw)
        f["pre_thw_p5_s"] = _fq(thw, 5)
        f["pre_thw_p50_s"] = _fq(thw, 50)
        f["pre_thw_p95_s"] = _fq(thw, 95)

        closing = np.where(vrel_l < -CLOSING_MIN, -vrel_l, NAN)
        ttc = np.where(drel_l > DREL_MIN, drel_l / closing, NAN)
        ttc = np.where(np.isfinite(ttc) & (ttc > 0), ttc, NAN)
        f["pre_ttc_min_raw_s"] = _fmin(ttc)
        ttc_cap = np.clip(ttc, 0, TTC_CAP)
        f["pre_ttc_min_capped_s"] = _fmin(ttc_cap)
        f["pre_ttc_p5_s"] = _fq(ttc_cap, 5)
        f["pre_ttc_p50_s"] = _fq(ttc_cap, 50)
        f["pre_ttc_p95_s"] = _fq(ttc_cap, 95)
        f["pre_time_of_min_ttc_s"] = _t_of(ttc, tt, np.argmin)

        drac = np.where(drel_l > DREL_MIN, closing ** 2 / (2 * drel_l), NAN)
        f["pre_drac_max_raw_mps2"] = _fmax(drac)
        drac_cap = np.clip(drac, 0, 50.0)
        f["pre_drac_max_capped_mps2"] = _fmax(drac_cap)
        f["pre_drac_p50_mps2"] = _fq(drac_cap, 50)
        f["pre_drac_p95_mps2"] = _fq(drac_cap, 95)
        f["pre_time_of_max_drac_s"] = _t_of(drac, tt, np.argmax)

    # ---- exposure (seconds / severity-weighted seconds) ----
    for th in (1.5, 2.0, 3.0):
        f[f"pre_time_below_ttc_{th}s"] = float((np.isfinite(ttc_cap) & (ttc_cap < th)).sum() * DT)
        excess = np.where(np.isfinite(ttc_cap), np.maximum(th - ttc_cap, 0.0), 0.0)
        f[f"pre_severity_integral_ttc_{th}s"] = float(excess.sum() * DT)
    for th in (0.8, 1.0, 1.5):
        f[f"pre_time_below_thw_{th}s"] = float((np.isfinite(thw) & (thw < th)).sum() * DT)
    for th in (3.0, 4.0):
        f[f"pre_time_above_drac_{th}mps2"] = float(
            (np.isfinite(drac_cap) & (drac_cap > th)).sum() * DT)

    # ---- perception ----
    f["pre_has_lane_probs"] = float(_finite(lp).size > 0 or _finite(rp).size > 0)
    f["pre_lane_left_prob_mean"] = _fmean(lp)
    f["pre_lane_right_prob_mean"] = _fmean(rp)
    f["pre_fcw_present"] = float(np.nan_to_num(fcw, nan=0.0).max() > 0.5) if n else 0.0

    return f


def decision_times(window: float, stride: float, min_lead: float, max_lead: float) -> np.ndarray:
    """Decision times s (feature window = [s-W, s]) with lead = -s in [min,max]."""
    first = CLIP_START_S + window          # window must fit inside the clip
    last = -min_lead
    first = max(first, -max_lead)          # respect the farthest allowed lead
    if first > last + EPS:
        return np.array([])
    k = int(np.floor((last - first) / stride + 1e-9))
    return np.round(first + stride * np.arange(k + 1), 6)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=3.5, help="feature window length W (s)")
    ap.add_argument("--stride", type=float, default=0.5, help="decision-time step (s)")
    ap.add_argument("--min-lead", type=float, default=0.0,
                    help="closest decision to takeover (s); 0 = window ends at the onset")
    ap.add_argument("--max-lead", type=float, default=None,
                    help="farthest decision (default 10 - W)")
    ap.add_argument("--limit", type=int, default=0, help="only build N clips (smoke test)")
    args = ap.parse_args()

    W, stride, min_lead = args.window, args.stride, args.min_lead
    max_lead = args.max_lead if args.max_lead is not None else (C.CLIP_SECONDS / 2 - W)
    assert 0 < W < 10 and stride > 0 and 0 <= min_lead <= max_lead

    times = decision_times(W, stride, min_lead, max_lead)
    assert times.size, "no decision times for this window/lead configuration"

    mt = pd.read_parquet(C.MODEL_TABLE)
    idx = pd.read_parquet(C.CLIP_INDEX)
    schema = json.loads(C.SCHEMA_JSON.read_text(encoding="utf-8"))
    base = mt.merge(idx[KEY_COLS + ["clip_dir"]], on=KEY_COLS, how="left")
    assert base["clip_dir"].notna().all(), "some labelled clips have no clip_dir on disk"
    if args.limit:
        base = base.head(args.limit).copy()
    base = base.reset_index(drop=True)

    print(f"[sliding] window W={W}s stride={stride}s  leads {min_lead}..{max_lead}s  "
          f"-> {times.size} windows/clip   window ends at t={times[-1]}s .. {times[0]}s")

    records, owner, failed = [], [], 0
    for pos, (_, r) in enumerate(base.iterrows()):
        try:
            t, series = load_series(Path(r.clip_dir))
            for s in times:
                f = window_features(series, t, s - W, s)
                f["decision_t_s"] = float(s)
                f["lead_s"] = float(-s)
                f["window_s"] = W
                f["stride_s"] = stride
                records.append(f)
                owner.append(pos)
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"      skip {r.car_model}/{r.driver}/{r.route}/{r.clip_id_pub}: {e}")
        if (pos + 1) % 200 == 0:
            print(f"      {pos + 1}/{len(base)} clips ...")

    fdf = pd.DataFrame(records)
    out = base.drop(columns=["clip_dir"]).iloc[owner].reset_index(drop=True)
    for c in fdf.columns:
        out[c] = fdf[c].to_numpy()

    recomputed = set(c for c in fdf.columns if c not in META_COLS)
    out = out.drop(columns=[c for c in out.columns
                            if c.startswith("pre_") and c not in recomputed])

    feats_num = [c for c in schema["features_numeric"] if c in out.columns]
    schema_out = {
        **schema,
        "n_rows": int(len(out)),
        "n_clips": int(base.shape[0] - failed),
        "features_numeric": feats_num,
        "feature_window": [CLIP_START_S, -min_lead],
        "target_window": list(C.SAFE_POST_WINDOW),
        "sliding": {
            "window_s": W, "stride_s": stride,
            "lead_min_s": min_lead, "lead_max_s": max_lead,
            "n_windows_per_clip": int(times.size),
            "decision_times_s": [float(x) for x in times],
            "grouping": "GroupKFold(dongle_id) subsumes clip -> no window leaks a fold",
        },
        "excluded_from_features": sorted(set(schema.get("excluded_from_features", []))
                                         | set(META_COLS)),
        "leakage_note": (
            f"Sliding forecast: features aggregated over a fixed {W}s window ending at "
            f"t=s (lead = -s in [{min_lead}, {max_lead}]s). No feature is computed at or "
            "after the takeover; targets are the post-window [0, +5] outcomes."
        ),
        "class_balance": out[schema["target_classification"]].value_counts().to_dict(),
    }

    out.to_parquet(C.FORECAST_TABLE, index=False)
    C.FORECAST_SCHEMA.write_text(json.dumps(schema_out, indent=2, default=str),
                                 encoding="utf-8")
    print(f"[sliding] {len(out):,} rows from {base.shape[0] - failed:,} clips "
          f"({failed} failed) x {len(feats_num)} numeric features -> {C.FORECAST_TABLE}")
    print(f"[sliding] rows per lead: "
          f"{out['lead_s'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
