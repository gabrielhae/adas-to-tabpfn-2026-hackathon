"""
Resample one ADAS-TO clip onto a common 20 Hz grid.

Why 20 Hz (from ADAS-TO/Code/configs/analysis_thresholds.yaml):
    "DO NOT resample above 20 Hz: it would fabricate information for qlog clips and for
     all radar/longPlan/model topics."
Measured native rates in the public sample:
    rlog: carState/ctrl/carControl/carOutput 100 Hz; radar/longPlan/model 20 Hz
    qlog: carState/ctrl/carControl/carOutput  10 Hz; radar 4 Hz; longPlan 2.7 Hz; model 1.3 Hz
So qlog `drivingModelData` at 1.3 Hz is heavily interpolated even at 20 Hz -> we flag it.

Event alignment (verified): the takeover sits at `csv_min + 10.0 s` in the CSV time base
(98.2% of 110 probed clips within 0.25 s). `meta.json`'s `video_time_s` is NOT reliable
for intra-clip sync (per-route clock offset), so everything is keyed off the CSV anchor.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

CLIP_SECONDS = C.CLIP_SECONDS
EVENT_OFFSET = C.EVENT_OFFSET_S

# topic -> (continuous cols, boolean cols)
TOPICS = {
    "carState": (
        ["vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "cruiseState.speed"],
        ["steeringPressed", "brakePressed", "gasPressed", "cruiseState.enabled", "standstill"],
    ),
    "controlsState": (
        ["curvature", "desiredCurvature", "vCruise"],
        ["enabled", "active"],
    ),
    "carControl": (
        ["actuators.accel", "actuators.torque", "actuators.curvature"],
        ["latActive", "longActive"],
    ),
    "radarState": (
        ["leadOne.dRel", "leadOne.vRel", "leadOne.vLead", "leadOne.aLeadK",
         "leadTwo.dRel", "leadTwo.vRel"],
        ["leadOne.status", "leadTwo.status"],
    ),
    "longitudinalPlan": (
        ["aTarget"],
        ["hasLead", "fcw", "shouldStop"],
    ),
    "drivingModelData": (
        ["laneLineMeta.leftProb", "laneLineMeta.rightProb",
         "action.desiredCurvature", "action.desiredAcceleration"],
        [],
    ),
}


def _to_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "1.0", "yes"])


def _read_topic(path: Path, cont: list[str], booleans: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    if "time_s" not in df.columns or len(df) == 0:
        return None
    out = pd.DataFrame({"time_s": pd.to_numeric(df["time_s"], errors="coerce")})
    out = out.dropna().sort_values("time_s").reset_index(drop=True)
    idx = df.loc[out.index] if False else None
    # re-align by position after sorting
    df = df.assign(_t=pd.to_numeric(df["time_s"], errors="coerce")).dropna(subset=["_t"])
    df = df.sort_values("_t").reset_index(drop=True)
    out = pd.DataFrame({"time_s": df["_t"].to_numpy(float)})
    for c in cont:
        out[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan
    for c in booleans:
        out[c] = _to_bool(df[c]).astype(float) if c in df.columns else np.nan
    return out


def _clean(x):
    """Recursively replace non-finite floats with None.

    JSON has no representation for NaN/Infinity; Python's json module emits bare
    `NaN`/`Infinity` which browsers reject, and Starlette raises on inf.
    """
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


def build_clip_telemetry(clip_dir: Path, meta: dict | None = None) -> dict:
    """Return a JSON-ready dict: common 20 Hz grid + all series + derived TTC/THW."""
    clip_dir = Path(clip_dir)
    if meta is None:
        try:
            meta = json.loads((clip_dir / "meta.json").read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    raw = {}
    for topic, (cont, booleans) in TOPICS.items():
        df = _read_topic(clip_dir / f"{topic}.csv", cont, booleans)
        if df is not None and len(df):
            raw[topic] = df

    if not raw:
        raise ValueError(f"no readable topics in {clip_dir}")

    cs = raw.get("carState")
    if cs is None or len(cs) < 5:
        raise ValueError(f"carState missing/too short in {clip_dir}")

    t0 = float(cs["time_s"].min())
    # verified anchor: takeover is 10 s after the first CSV timestamp
    event_t = t0 + EVENT_OFFSET
    grid = np.arange(t0, t0 + CLIP_SECONDS + 1e-9, 1.0 / C.RESAMPLE_HZ)

    series: dict[str, list] = {}
    native_hz: dict[str, float] = {}

    for topic, df in raw.items():
        t = df["time_s"].to_numpy(float)
        dt = np.median(np.diff(t)) if len(t) > 2 else np.nan
        native_hz[topic] = float(1.0 / dt) if dt and dt > 0 else float("nan")
        cont, booleans = TOPICS[topic]
        for c in cont:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            good = np.isfinite(v)
            if good.sum() < 2:
                continue
            series[c] = np.interp(grid, t[good], v[good], left=np.nan, right=np.nan).round(4).tolist()
        for c in booleans:
            v = pd.to_numeric(df[c], errors="coerce").to_numpy(float)
            good = np.isfinite(v)
            if good.sum() < 1:
                continue
            # forward-fill semantics for booleans
            idx = np.searchsorted(t[good], grid, side="right") - 1
            idx = np.clip(idx, 0, len(v[good]) - 1)
            series[c] = v[good][idx].round(3).tolist()

    # ---- derived: TTC / THW / closing ----
    v = np.array(series.get("vEgo", [np.nan] * len(grid)), float)
    drel = np.array(series.get("leadOne.dRel", [np.nan] * len(grid)), float)
    vrel = np.array(series.get("leadOne.vRel", [np.nan] * len(grid)), float)
    lead = np.array(series.get("leadOne.status", [0.0] * len(grid)), float)

    # finite-only view of the inputs so division can never emit inf
    v_f = np.where(np.isfinite(v), v, np.nan)
    drel_f = np.where(np.isfinite(drel), drel, np.nan)
    vrel_f = np.where(np.isfinite(vrel), vrel, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        closing = np.where(vrel_f < -C.ANALYSIS["closing_min_mps"], -vrel_f, np.nan)
        # guard the denominator explicitly; np.where evaluates both branches
        denom = np.where(np.isfinite(closing) & (closing > 0), closing, np.nan)
        ttc = np.where(drel_f > C.ANALYSIS["drel_min_m"], drel_f / denom, np.nan)
        thw_den = np.where(np.isfinite(v_f) & (v_f > C.ANALYSIS["min_speed_thw_mps"]), v_f, np.nan)
        thw = np.where(np.isfinite(drel_f), drel_f / thw_den, np.nan)
    ttc = np.where(np.isfinite(ttc) & (ttc > 0), np.clip(ttc, 0, C.ANALYSIS["ttc_cap_s"]), np.nan)
    thw = np.where(np.isfinite(thw) & (thw > 0), thw, np.nan)
    leadmask = lead > 0.5
    series["ttc"] = np.where(leadmask, ttc, np.nan).round(3).tolist()
    series["thw"] = np.where(leadmask, thw, np.nan).round(3).tolist()
    series["closing_mps"] = np.where(leadmask, closing, np.nan).round(3).tolist()

    # ADAS engagement (paper definition: controlsState.enabled OR cruiseState.enabled)
    ctrl_en = np.array(series.get("enabled", [0.0] * len(grid)), float)
    cruise_en = np.array(series.get("cruiseState.enabled", [0.0] * len(grid)), float)
    engaged = ((ctrl_en > 0.5) | (cruise_en > 0.5)).astype(float)
    series["adas_engaged"] = engaged.tolist()

    return _clean({
        "t0_csv": t0,
        "event_t": event_t,
        "event_index": int(np.argmin(np.abs(grid - event_t))),
        "grid_start_s": float(grid[0]),
        "grid_end_s": float(grid[-1]),
        "n_samples": int(len(grid)),
        "resample_hz": C.RESAMPLE_HZ,
        "t": np.round(grid - event_t, 3).tolist(),   # relative to takeover: -10 .. +10
        "series": series,
        "native_hz": native_hz,
        "log_kind": meta.get("log_kind"),
        "log_hz": meta.get("log_hz"),
        "adas_engaged_pct": float(engaged.mean() * 100),
        "warnings": _warnings(series, native_hz, meta),
    })


def _warnings(series: dict, native_hz: dict, meta: dict) -> list[str]:
    w = []
    dm = native_hz.get("drivingModelData", float("nan"))
    if dm == dm and dm < 8:
        w.append(f"drivingModelData native rate is {dm:.1f} Hz; lane-probability traces are "
                 f"heavily interpolated on the 20 Hz grid.")
    lp = native_hz.get("longitudinalPlan", float("nan"))
    if lp == lp and lp < 8:
        w.append(f"longitudinalPlan native rate is {lp:.1f} Hz; planner traces are interpolated.")
    if not any(np.isfinite(np.array(series.get("leadOne.dRel", [np.nan])))):
        w.append("No lead vehicle reported by radar for this clip.")
    if (meta.get("clip_dur_s") or 20) < 19.0:
        w.append(f"meta clip_dur_s={meta.get('clip_dur_s')}; clip is shorter than the nominal 20 s.")
    return w
