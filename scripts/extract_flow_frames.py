"""
Flow stage 1 - ego-motion-compensated optical flow and independently-moving-object
(IMO) detection, one row per (clip, retained frame).

Decodes each clip's `takeover.mp4` ONCE, sequentially, keeping every
`round(fps / FLOW_HZ)`-th frame (10 Hz by default) with `t_rel in [-10, 0]`, then
estimates dense optical flow between consecutive retained grey frames:

    1. flow        cv2.DISOpticalFlow (default) or torchvision RAFT (`--method raft`)
    2. global fit  cv2.estimateAffinePartial2D (RANSAC similarity) on an every-8th-pixel
                   grid, masking the ego hood (bottom 15%) and a 4-px border
    3. residual    observed flow - the global (ego-motion) model flow, on all pixels
    4. IMO         |residual| > max(1.5 px, 3 x median|residual|), connected components
                   with min area 0.1% of the frame
    5. divergence  du/dx + dv/dy of the residual field (positive = expanding)
    6. onset       rising edge of the residual activity, vs a k x rolling-median baseline

Box geometry describes objects; it cannot express "something started moving". Residual
flow after ego-motion compensation isolates what moves relative to the world, and a
rising edge in that energy is an event onset. This family is scored standalone against
the CAN baseline - it is NOT stacked onto the existing `vis_*` block.

`t_rel = frame_idx / fps - 10.0` is clip-relative video time with the takeover at 0,
matching `build_clip_telemetry`'s `t`. Only `t_rel in [-10, 0]` rows are kept, which
covers every forecast window (`[s - W, s]`, `s <= 0`).

Frames whose global fit fails (`flow_valid == 0`) use the project's zero-inflated
encoding: every residual/IMO/onset feature is 0.0, a real "nothing detected" sentinel,
so `model.encode()` never median-imputes a non-existent event.

Writes:
    data/derived/flow_frames.parquet   (working format)
    data/derived/flow_frames.csv       (human/portable export)

Usage:
    python scripts/extract_flow_frames.py --limit 5          # smoke test
    python scripts/extract_flow_frames.py --resume           # skip done clips
    python scripts/extract_flow_frames.py --method raft
    python scripts/extract_flow_frames.py --fps 10 --device 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adas_to import config as C

KEY_COLS = ["car_model", "driver", "route", "clip_id_pub"]
NAN = float("nan")
EPS = 1e-9

CORRIDOR_LEFT = 0.25
CORRIDOR_RIGHT = 0.75
BORDER_PX = 4
GRID_STEP = 8

IMO_MIN_AREA_FRAC = 0.001        # 0.1% of the frame
IMO_ABS_THRESH_PX = 1.5          # absolute floor for a "moving" pixel
IMO_MED_MULT = 3.0               # ... or 3x the median |residual|

ONSET_K = 1.8                    # rising edge vs k x rolling median
ONSET_FLOOR_PX = 1.0             # absolute floor for the rising edge
ONSET_LOOKBACK_S = 2.0           # rolling-median window

# --- global-fit constants (config mirrors --method) --------------------------
GLOBAL_MIN_POINTS = 50
GLOBAL_RANSAC_PX = C.FLOW_RANSAC_REPROJ_PX
HOOD_FRAC = C.FLOW_HOOD_FRAC

RAFT_W, RAFT_H = C.FLOW_RAFT_W, C.FLOW_RAFT_H

FRAME_COLS = [
    "car_model", "driver", "route", "clip_id_pub", "vid_kind",
    "t_rel", "frame_idx", "img_w", "img_h",
    "flow_valid", "flow_ransac_inlier_frac",
    "flow_global_tx", "flow_global_ty", "flow_global_scale", "flow_global_rot",
    "flow_res_valid_frac",
    "flow_res_mag_mean", "flow_res_mag_p95", "flow_res_mag_max",
    "flow_res_max_x_frac", "flow_res_max_y_frac",
    "flow_res_corridor_mag_mean", "flow_res_corridor_mag_p95",
    "flow_res_lat_mean", "flow_res_long_mean",
    "flow_div_max", "flow_div_corridor_mean",
    "flow_imo_count", "flow_imo_area_frac", "flow_imo_corridor_count",
    "flow_imo_max_area_frac", "flow_imo_mean_area_frac",
    "flow_onset", "flow_onset_corridor",
]


# --------------------------------------------------------------------------- #
# decoding
# --------------------------------------------------------------------------- #
def decode_frames(video: Path, fps_target: float):
    """Sequentially decode one clip, returning retained (frame_idx, t_rel, bgr).

    Every `round(fps / fps_target)`-th frame with `t_rel in [-10, 0]` is kept. The
    loop stops as soon as `t_rel > 0`, since no feature or target uses post-takeover
    frames.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("VideoCapture could not open the file")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            fps = 20.0
        step = max(1, int(round(fps / fps_target)))
        out = []
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            idx += 1
            if idx % step != 0:
                continue
            t_rel = idx / fps - 10.0
            if t_rel > 0.0:
                break
            if t_rel < -10.0:
                continue
            out.append((idx, float(t_rel), frame))
    finally:
        cap.release()
    return out, fps


# --------------------------------------------------------------------------- #
# optical flow
# --------------------------------------------------------------------------- #
def _make_dis():
    import cv2
    return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)


_RAFT_CACHE: dict = {}


def _raft_model(device):
    """Load (once per device) the torchvision RAFT-Large model."""
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
    key = str(device)
    if key not in _RAFT_CACHE:
        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        _RAFT_CACHE[key] = model.to(device).eval()
    return _RAFT_CACHE[key]


def _to_raft_tensor(gray, device):
    """Grey uint8 frame -> float [3, H, W] in [0, 1] at the RAFT working size."""
    import cv2
    import torch
    x = cv2.resize(gray, (RAFT_W, RAFT_H), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x.astype(np.float32) / 255.0)
    t = t.unsqueeze(0).repeat(3, 1, 1)      # [3, H, W]
    return t.unsqueeze(0).to(device)        # [1, 3, H, W]


def raft_flows(grays, device, batch: int = 16):
    """Dense flow for every consecutive pair, evaluated at RAFT_W x RAFT_H and
    rescaled to the original pixel grid. Returns a list of (u, v) float32 arrays
    (empty list if there are fewer than two frames)."""
    import cv2
    import torch

    if len(grays) < 2:
        return []
    torch.backends.cudnn.benchmark = True
    model = _raft_model(device)
    h0, w0 = grays[0].shape[:2]
    sx, sy = w0 / RAFT_W, h0 / RAFT_H
    a_all = [_to_raft_tensor(g, device) for g in grays[:-1]]
    b_all = [_to_raft_tensor(g, device) for g in grays[1:]]
    flows = []
    with torch.no_grad():
        for i in range(0, len(a_all), batch):
            preds = model(torch.cat(a_all[i:i + batch]), torch.cat(b_all[i:i + batch]))
            f = preds[-1].permute(0, 2, 3, 1).cpu().numpy()  # [B, h, w, 2]
            for k in range(f.shape[0]):
                u = cv2.resize(f[k, ..., 0], (w0, h0),
                               interpolation=cv2.INTER_LINEAR) * sx
                v = cv2.resize(f[k, ..., 1], (w0, h0),
                               interpolation=cv2.INTER_LINEAR) * sy
                flows.append((u.astype(np.float32), v.astype(np.float32)))
    if str(device) != "cpu":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    return flows


def dis_flows(grays):
    """Dense flow for every consecutive pair with DIS; returns list of (u, v)."""
    if len(grays) < 2:
        return []
    dis = _make_dis()
    flows = []
    for i in range(1, len(grays)):
        if grays[i].shape != grays[i - 1].shape:
            flows.append(None)
            continue
        f = dis.calc(grays[i - 1], grays[i], None)
        flows.append((f[..., 0].astype(np.float32), f[..., 1].astype(np.float32)))
    return flows


# --------------------------------------------------------------------------- #
# ego-motion fit and residual statistics
# --------------------------------------------------------------------------- #
def _valid_and_corridor(h: int, w: int):
    mask = np.ones((h, w), bool)
    b = BORDER_PX
    mask[:b, :] = False
    mask[-b:, :] = False
    mask[:, :b] = False
    mask[:, -b:] = False
    hood = int(round(HOOD_FRAC * h))
    if hood > 0:
        mask[h - hood:, :] = False
    xs = np.arange(w)
    corridor = np.zeros((h, w), bool)
    cl = int(np.searchsorted(xs, CORRIDOR_LEFT * w))
    cr = int(np.searchsorted(xs, CORRIDOR_RIGHT * w))
    corridor[:, cl:cr] = True
    corridor &= mask
    return mask, corridor


def fit_global(u: np.ndarray, v: np.ndarray, valid: np.ndarray, corridor: np.ndarray):
    """RANSAC similarity fit (tx, ty, scale, rot) from a subsampled point grid.

    Returns (M, inlier_frac) or (None, 0.0) when the fit fails.
    """
    import cv2

    h, w = u.shape
    ys = np.arange(BORDER_PX, max(BORDER_PX + 1, h - BORDER_PX), GRID_STEP)
    xs = np.arange(BORDER_PX, max(BORDER_PX + 1, w - BORDER_PX), GRID_STEP)
    xx, yy = np.meshgrid(xs, ys)
    m = valid[yy, xx]
    if m.sum() < GLOBAL_MIN_POINTS:
        return None, 0.0
    sx = xx[m].astype(np.float32)
    sy = yy[m].astype(np.float32)
    src = np.stack([sx, sy], axis=1)
    dst = np.stack([sx + u[yy, xx][m], sy + v[yy, xx][m]], axis=1)
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=GLOBAL_RANSAC_PX,
        maxIters=2000, confidence=0.99, refineIters=10)
    if M is None:
        return None, 0.0
    frac = float(np.asarray(inliers).sum()) / float(len(src)) if inliers is not None else 0.0
    return M.astype(np.float64), frac


def model_flow(M, h: int, w: int):
    """Flow implied by the global similarity model at every pixel."""
    ys, xs = np.mgrid[0:h, 0:w]
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    px = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    py = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return (px - x).astype(np.float32), (py - y).astype(np.float32)


def _divergence(ru: np.ndarray, rv: np.ndarray) -> np.ndarray:
    """du/dx + dv/dy of the residual field."""
    dudx = np.gradient(ru, axis=1)
    dvdy = np.gradient(rv, axis=0)
    return (dudx + dvdy).astype(np.float32)


def frame_flow_scalars(u, v, valid, corridor):
    """Residual statistics, divergence and IMO components for one frame pair.

    Zero-inflated: a failed global fit returns all-0.0 flow features (a real "nothing
    detected" sentinel), never NaN.
    """
    import cv2

    f = {k: 0.0 for k in FRAME_FLOW_KEYS}
    if u is None or v is None or u.shape != v.shape:
        return f, None, None
    h, w = u.shape

    M, inlier = fit_global(u, v, valid, corridor)
    if M is None:
        return f, None, None

    mu, mv = model_flow(M, h, w)
    ru = u - mu
    rv = v - mv
    mag = np.sqrt(ru * ru + rv * rv)
    valid_f = valid & np.isfinite(mag)

    f["flow_valid"] = 1.0
    f["flow_ransac_inlier_frac"] = float(inlier)
    a = float(M[0, 0])
    b = float(M[1, 0])
    f["flow_global_tx"] = float(M[0, 2])
    f["flow_global_ty"] = float(M[1, 2])
    f["flow_global_scale"] = float(np.hypot(a, b))
    f["flow_global_rot"] = float(np.arctan2(b, a))
    f["flow_res_valid_frac"] = float(valid_f.mean()) if valid_f.size else 0.0

    mv_valid = mag[valid_f]
    if mv_valid.size == 0:
        return f, None, None
    f["flow_res_mag_mean"] = float(mv_valid.mean())
    f["flow_res_mag_p95"] = float(np.percentile(mv_valid, 95))
    f["flow_res_mag_max"] = float(mv_valid.max())
    flat = int(np.argmax(np.where(valid_f, mag, -np.inf)))
    f["flow_res_max_x_frac"] = float((flat % w) / max(1, w - 1))
    f["flow_res_max_y_frac"] = float((flat // w) / max(1, h - 1))

    cmag = mag[corridor & np.isfinite(mag)]
    if cmag.size:
        f["flow_res_corridor_mag_mean"] = float(cmag.mean())
        f["flow_res_corridor_mag_p95"] = float(np.percentile(cmag, 95))
    f["flow_res_lat_mean"] = float(ru[valid_f].mean())
    f["flow_res_long_mean"] = float(rv[valid_f].mean())

    div = _divergence(ru, rv)
    dv = div[valid_f & np.isfinite(div)]
    if dv.size:
        f["flow_div_max"] = float(dv.max())
    dcorr = div[corridor & np.isfinite(div)]
    if dcorr.size:
        f["flow_div_corridor_mean"] = float(dcorr.mean())

    # --- IMO connected components ------------------------------------------
    med = float(np.median(mv_valid))
    thr = max(IMO_ABS_THRESH_PX, IMO_MED_MULT * med)
    binary = np.zeros((h, w), np.uint8)
    binary[valid_f & (mag > thr)] = 1
    if binary.any():
        n, _, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = IMO_MIN_AREA_FRAC * h * w
        total = 0.0
        areas = []
        corridor_count = 0
        for lab in range(1, n):
            area = float(stats[lab, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            total += area
            areas.append(area)
            cx = float(cents[lab][0])
            if CORRIDOR_LEFT * w <= cx <= CORRIDOR_RIGHT * w:
                corridor_count += 1
        f["flow_imo_count"] = float(len(areas))
        f["flow_imo_area_frac"] = float(total / (h * w))
        f["flow_imo_corridor_count"] = float(corridor_count)
        if areas:
            f["flow_imo_max_area_frac"] = float(max(areas) / (h * w))
            f["flow_imo_mean_area_frac"] = float(np.mean(areas) / (h * w))
    return f, mag, valid_f


# every feature key emitted per frame (all zero-inflated to 0.0 when unavailable)
FRAME_FLOW_KEYS = [
    "flow_valid", "flow_ransac_inlier_frac",
    "flow_global_tx", "flow_global_ty", "flow_global_scale", "flow_global_rot",
    "flow_res_valid_frac",
    "flow_res_mag_mean", "flow_res_mag_p95", "flow_res_mag_max",
    "flow_res_max_x_frac", "flow_res_max_y_frac",
    "flow_res_corridor_mag_mean", "flow_res_corridor_mag_p95",
    "flow_res_lat_mean", "flow_res_long_mean",
    "flow_div_max", "flow_div_corridor_mean",
    "flow_imo_count", "flow_imo_area_frac", "flow_imo_corridor_count",
    "flow_imo_max_area_frac", "flow_imo_mean_area_frac",
]


def onset_flags(activity: np.ndarray, t_rel: np.ndarray, lookback_s: float) -> np.ndarray:
    """Rising edges of `activity` above max(k x rolling median, absolute floor).

    A heuristic, documented in the README. The rolling median is robust to the very
    spike being tested, so a genuine onset needs both a local rise and to clear the
    absolute floor.
    """
    a = np.nan_to_num(np.asarray(activity, float), nan=0.0)
    n = a.size
    dt = float(np.median(np.diff(t_rel))) if n > 1 else 0.1
    look = max(1, int(round(lookback_s / max(EPS, dt))))
    above = np.zeros(n, bool)
    for i in range(n):
        lo = max(0, i - look)
        base = float(np.median(a[lo:i + 1]))
        thr = max(ONSET_K * base, ONSET_FLOOR_PX)
        above[i] = a[i] > thr
    onset = above & ~np.concatenate([[False], above[:-1]])
    return onset.astype(float)


def process_clip(clip, *, method, device, fps_target):
    import cv2

    frames, fps = decode_frames(Path(clip.video), fps_target)
    if not frames:
        raise RuntimeError("no frames decoded")
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for _, _, f in frames]

    if method == "raft":
        dev = device
        if dev is None:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        flows = raft_flows(grays, dev)
    else:
        flows = dis_flows(grays)

    h, w = grays[0].shape[:2]
    valid, corridor = _valid_and_corridor(h, w)

    rows = []
    for i, (idx, t_rel, _) in enumerate(frames):
        if i == 0:
            flow = None
        else:
            flow = flows[i - 1]
        if flow is None:
            u = v = None
        else:
            u, v = flow
        f, _, _ = frame_flow_scalars(u, v, valid, corridor)
        f.update({
            "car_model": clip.car_model, "driver": clip.driver,
            "route": clip.route, "clip_id_pub": clip.clip_id_pub,
            "vid_kind": clip.vid_kind, "t_rel": float(t_rel),
            "frame_idx": int(idx), "img_w": int(w), "img_h": int(h),
        })
        rows.append(f)

    # --- onset detection (second pass over the clip's own activity) ----------
    t = np.asarray([r["t_rel"] for r in rows], float)
    act = np.asarray([r["flow_res_mag_mean"] for r in rows], float)
    act_c = np.asarray([r["flow_res_corridor_mag_mean"] for r in rows], float)
    on = onset_flags(act, t, ONSET_LOOKBACK_S)
    on_c = onset_flags(act_c, t, ONSET_LOOKBACK_S)
    for i, r in enumerate(rows):
        r["flow_onset"] = float(on[i])
        r["flow_onset_corridor"] = float(on_c[i])
    return rows, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process N clips (smoke test)")
    ap.add_argument("--fps", type=float, default=C.FLOW_HZ, help="flow sampling rate (Hz)")
    ap.add_argument("--method", default=C.FLOW_METHOD, choices=["dis", "raft"])
    ap.add_argument("--device", default=None, help="'0'/'cuda' for GPU, 'cpu' for CPU")
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already present in flow_frames.parquet")
    args = ap.parse_args()

    forecast = pd.read_parquet(C.FORECAST_TABLE, columns=KEY_COLS)
    wanted = set(map(tuple, forecast.drop_duplicates()[KEY_COLS].to_numpy()))
    idx = pd.read_parquet(C.CLIP_INDEX)
    idx = idx[idx[KEY_COLS].apply(tuple, axis=1).isin(wanted)].reset_index(drop=True)

    existing = None
    if args.resume and C.FLOW_FRAMES.exists():
        existing = pd.read_parquet(C.FLOW_FRAMES)
        done = set(map(tuple, existing[KEY_COLS].drop_duplicates().to_numpy()))
        idx = idx[~idx[KEY_COLS].apply(tuple, axis=1).isin(done)].reset_index(drop=True)
        print(f"[flow] resume: {len(done)} clips already extracted, {len(idx)} remaining")
    if args.limit:
        idx = idx.head(args.limit).reset_index(drop=True)
    if idx.empty:
        print("[flow] nothing to do")
        return

    print(f"[flow] {len(idx)} clips @ {args.fps} Hz  method={args.method}  "
          f"device={args.device or 'auto'}")

    t0 = time.time()
    records, failed = [], []
    for pos, clip in enumerate(idx.itertuples(index=False)):
        try:
            rows, fps = process_clip(clip, method=args.method, device=args.device,
                                     fps_target=args.fps)
            records.extend(rows)
        except Exception as e:
            failed.append((clip.car_model, clip.driver, clip.route,
                           clip.clip_id_pub, str(e)))
        if (pos + 1) % 25 == 0 or pos + 1 == len(idx):
            print(f"      {pos + 1}/{len(idx)} clips  ({time.time() - t0:.0f}s)")

    new = pd.DataFrame(records)
    if new.empty:
        raise SystemExit("[flow] no frames extracted; nothing written")
    new = new[FRAME_COLS]
    if existing is not None and not existing.empty:
        out = pd.concat([existing[FRAME_COLS], new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(KEY_COLS + ["frame_idx"]).reset_index(drop=True)

    out.to_parquet(C.FLOW_FRAMES, index=False)
    out.to_csv(C.FLOW_FRAMES_CSV, index=False)

    valid_rate = float(out["flow_valid"].mean())
    print(f"[flow] {len(out):,} frames from "
          f"{out[KEY_COLS].drop_duplicates().shape[0]} clips ({len(failed)} failed) "
          f"-> {C.FLOW_FRAMES}")
    print(f"[flow] flow-valid rate={valid_rate:.3f}  "
          f"mean |res|={out['flow_res_mag_mean'].mean():.3f}px  "
          f"mean IMO count={out['flow_imo_count'].mean():.3f}  "
          f"onsets={int(out['flow_onset'].sum())} "
          f"(corridor {int(out['flow_onset_corridor'].sum())})")
    for f in failed[:10]:
        print(f"      skip {f[0]}/{f[1]}/{f[2]}/{f[3]}: {f[4]}")


if __name__ == "__main__":
    main()
