"""
Vision QA - proves the per-frame boxes are real, correctly aligned, and that the
box picked as "lead" is actually the car ahead.

Two checks:

1. Radar agreement. For N sampled (clip, decision) rows, compare the vision lead
   geometry against radar `leadOne.dRel` from `build_clip_telemetry` on the same
   relative time axis. Larger `lead_area_frac` means the lead is nearer, so
   `1/sqrt(area_frac)` should correlate POSITIVELY with `dRel`. A correctly
   aligned detector gives a positive per-clip Spearman; a mis-aligned or
   mis-identified lead does not.

2. Contact sheet. `results/vision_qa_grid.png`: 4 clips x 5 frames at
   `t_rel = {-8,-6,-4,-2,0}`, with the chosen lead box in red and the ego-lane
   corridor drawn, for a human eyeball check that the red box is the car ahead.

Writes:
    results/vision_alignment.json
    results/vision_qa_grid.png
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
from adas_to.telemetry import build_clip_telemetry

KEY_COLS = ["car_model", "driver", "route", "clip_id_pub"]
GRID_TIMES = [-8.0, -6.0, -4.0, -2.0, 0.0]

CORRIDOR_LEFT = 0.25
CORRIDOR_RIGHT = 0.75
NEAR_FIELD_Y2 = 0.55
CORRIDOR_MIN_Y2 = 0.30


def sample_rows(n: int, seed: int) -> pd.DataFrame:
    ft = pd.read_parquet(C.FORECAST_TABLE,
                         columns=KEY_COLS + ["decision_t_s", "lead_s", "window_s"])
    frames = pd.read_parquet(C.VISION_FRAMES, columns=KEY_COLS)
    have = frames[KEY_COLS].drop_duplicates()
    ft = ft.merge(have, on=KEY_COLS, how="inner")
    if ft.empty:
        raise SystemExit("no extracted clips overlap the forecast table")
    rng = np.random.default_rng(seed)
    take = min(n, len(ft))
    idx = rng.choice(len(ft), size=take, replace=False)
    return ft.iloc[idx].reset_index(drop=True)


def radar_drel(clip_dir: str):
    tel = build_clip_telemetry(Path(clip_dir))
    t = np.asarray(tel["t"], float)
    d = np.asarray(tel["series"].get("leadOne.dRel", [np.nan] * len(t)), float)
    return t, d


def _rho_for(t_rel, area, row, hi_mask):
    from scipy.stats import spearmanr
    m = hi_mask & np.isfinite(area) & (area > 0)
    if m.sum() < 5:
        return int(m.sum()), None
    rt, rd = radar_drel(row.clip_dir)
    good = np.isfinite(rd)
    if good.sum() < 2:
        return int(m.sum()), None
    drel_at = np.interp(t_rel[m], rt[good], rd[good])
    valid = np.isfinite(drel_at)
    if valid.sum() < 5:
        return int(valid.sum()), None
    if floats_equal(area[m][valid]) or floats_equal(drel_at[valid]):
        return int(valid.sum()), None
    rho = spearmanr(1.0 / np.sqrt(area[m][valid]), drel_at[valid]).statistic
    return int(valid.sum()), (float(rho) if np.isfinite(rho) else None)


def floats_equal(x) -> bool:
    x = np.asarray(x, float)
    return bool(x.size) and float(np.nanmax(x) - np.nanmin(x)) == 0.0


def per_clip_corr(row, frames_clip: pd.DataFrame) -> dict:
    t_rel = frames_clip["t_rel"].to_numpy(float)
    area = frames_clip["lead_area_frac"].to_numpy(float)
    hi = row.decision_t_s
    lo = row.decision_t_s - row.window_s
    n, rho = _rho_for(t_rel, area, row, (t_rel >= lo - 1e-6) & (t_rel <= hi + 1e-6))
    n_pre, rho_pre = _rho_for(t_rel, area, row, t_rel <= hi + 1e-6)
    return {"n": n, "rho": rho, "n_pre": n_pre, "rho_pre": rho_pre}


def grab_frames(video: Path, want_idx: dict):
    import cv2
    cap = cv2.VideoCapture(str(video))
    out = {}
    if not cap.isOpened():
        return out
    try:
        for t, fidx in want_idx.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ok, frame = cap.read()
            if ok and frame is not None:
                out[t] = (frame, int(fidx))
    finally:
        cap.release()
    return out


def draw_contact_sheet(rows: pd.DataFrame, frames: pd.DataFrame, out_png: Path) -> list:
    import cv2

    picked = []
    for _, r in rows.iterrows():
        sub = frames[(frames[KEY_COLS] == r[KEY_COLS].values).all(axis=1)]
        if not sub.empty:
            picked.append((r, sub))
    picked = sorted(picked, key=lambda x: -x[1]["lead_present"].mean())[:4]
    if not picked:
        return []

    idx = pd.read_parquet(C.CLIP_INDEX, columns=KEY_COLS + ["clip_dir", "video"])
    tiles = []
    for r, sub in picked:
        key = tuple(r[KEY_COLS].tolist())
        meta = idx[idx[KEY_COLS].apply(tuple, axis=1) == key]
        if meta.empty:
            continue
        video = Path(meta.iloc[0]["video"])
        want = {}
        for t in GRID_TIMES:
            j = (sub["t_rel"] - t).abs().idxmin()
            want[t] = int(sub.loc[j, "frame_idx"])
        got = grab_frames(video, want)
        row_tiles = []
        for t in GRID_TIMES:
            if t not in got:
                continue
            frame, fidx = got[t]
            h, w = frame.shape[:2]
            frow = sub[sub["frame_idx"] == fidx]
            if not frow.empty and float(frow.iloc[0]["lead_present"]) > 0.5:
                cx = (0.5 + float(frow.iloc[0]["lead_cx_offset"])) * w
                y2 = float(frow.iloc[0]["lead_y2_norm"]) * h
                area = float(frow.iloc[0]["lead_area_frac"]) * w * h
                bw = max(8.0, area / max(1.0, y2 * 0.5))
                bh = max(8.0, y2 * 0.5)
                cv2.rectangle(frame, (int(cx - bw / 2), int(y2 - bh)),
                              (int(cx + bw / 2), int(y2)), (0, 0, 255), 2)
            cv2.rectangle(frame, (int(CORRIDOR_LEFT * w), int(CORRIDOR_MIN_Y2 * h)),
                          (int(CORRIDOR_RIGHT * w), h - 1), (255, 0, 0), 1)
            cv2.line(frame, (0, int(NEAR_FIELD_Y2 * h)), (w, int(NEAR_FIELD_Y2 * h)),
                     (0, 255, 255), 1)
            cv2.putText(frame, f"t={t:+.0f}s", (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)
            row_tiles.append(frame)
        if row_tiles:
            tiles.append(row_tiles)
    if not tiles:
        return []
    width = max(len(t) for t in tiles)
    h = tiles[0][0].shape[0]
    canvas = np.zeros((h * len(tiles), tiles[0][0].shape[1] * width, 3), np.uint8)
    for i, row_tiles in enumerate(tiles):
        for j, tile in enumerate(row_tiles):
            canvas[i * h:(i + 1) * h, j * tile.shape[1]:(j + 1) * tile.shape[1]] = tile
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    return [tuple(r[KEY_COLS].tolist()) for r, _ in picked]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=40, help="number of (clip, window) rows")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = pd.read_parquet(C.VISION_FRAMES)
    ft = pd.read_parquet(C.FORECAST_TABLE, columns=KEY_COLS + ["decision_t_s", "window_s"])
    rows = sample_rows(args.n, args.seed)
    clip_dir = pd.read_parquet(C.CLIP_INDEX, columns=KEY_COLS + ["clip_dir"])
    rows = rows.merge(clip_dir, on=KEY_COLS, how="left")

    results = []
    for _, r in rows.iterrows():
        sub = frames[(frames[KEY_COLS] == r[KEY_COLS].values).all(axis=1)]
        results.append({**{k: r[k] for k in KEY_COLS},
                        "decision_t_s": float(r.decision_t_s),
                        **per_clip_corr(r, sub)})

    rhos = [x["rho"] for x in results if x["rho"] is not None]
    positive = int(sum(1 for x in rhos if x > 0))
    rho_pre = [x["rho_pre"] for x in results if x.get("rho_pre") is not None]
    payload = {
        "n_sampled": len(results),
        "n_with_correlation": len(rhos),
        "median_spearman": float(np.median(rhos)) if rhos else None,
        "positive_fraction": (positive / len(rhos)) if rhos else None,
        "pass": bool(rhos and np.median(rhos) > 0.3 and positive / len(rhos) > 0.5),
        "median_spearman_pre_window": float(np.median(rho_pre)) if rho_pre else None,
        "n_pre_window": len(rho_pre),
        "per_row": results,
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "vision_alignment.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"[qa] sampled {payload['n_sampled']} rows; "
          f"{payload['n_with_correlation']} with a usable correlation")
    print(f"[qa] median Spearman(1/sqrt(area), dRel) = {payload['median_spearman']}")
    print(f"[qa] positive fraction = {payload['positive_fraction']}  PASS={payload['pass']}")
    print(f"[qa] wrote {C.RESULTS / 'vision_alignment.json'}")

    picked = draw_contact_sheet(rows, frames, C.RESULTS / "vision_qa_grid.png")
    if picked:
        print(f"[qa] wrote {C.RESULTS / 'vision_qa_grid.png'} for {len(picked)} clips")
    else:
        print("[qa] no contact sheet produced")


if __name__ == "__main__":
    main()
