"""
Flow QA - blocking gate before the full experiment run.

Three checks:

1. Radar agreement. For N sampled (clip, decision) rows, compare the residual
   divergence in the ego-lane corridor against radar `leadOne.vRel` from
   `build_clip_telemetry` on the same relative time axis. Closing traffic has
   `vRel < 0` and makes the residual field expand, so the per-clip Spearman should
   be NEGATIVE; a majority negative is the alignment signal.

2. Jitter confound. Residual flow after the RANSAC global fit should not be driven
   by camera bump / rotation noise. We report the onset rate among the top-decile
   |global rotation| frames vs the rest.

3. Contact sheet. `results/flow_qa_grid.png`: sampled frames annotated with the
   fitted global flow (green arrows), the residual flow (red arrows) and the
   detected IMO components (blue boxes), plus the ego-hood mask line. Expect
   compact boxes on cut-ins / pedestrians / oncoming traffic, not the hood or
   uniform background.

Asserts no future-frame leakage and that every `flow_*` column is defined.

Writes:
    results/flow_alignment.json
    results/flow_qa_grid.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2  # noqa: E402

from adas_to import config as C  # noqa: E402
from adas_to.telemetry import build_clip_telemetry  # noqa: E402
import extract_flow_frames as X  # noqa: E402

KEY_COLS = ["car_model", "driver", "route", "clip_id_pub"]
GRID_TIMES = [-3.0, -2.0, -1.0, 0.0]
TILE_W, TILE_H = 526, 330


def floats_equal(x) -> bool:
    x = np.asarray(x, float)
    return bool(x.size) and float(np.nanmax(x) - np.nanmin(x)) == 0.0


def radar_series(clip_dir: str):
    """Radar lead relative speed and presence, on the event-relative time axis."""
    tel = build_clip_telemetry(Path(clip_dir))
    t = np.asarray(tel["t"], float)
    v = np.asarray(tel["series"].get("leadOne.vRel", [np.nan] * len(t)), float)
    st = np.asarray(tel["series"].get("leadOne.status", [np.nan] * len(t)), float)
    return t, v, st


def sample_rows(n: int, seed: int) -> pd.DataFrame:
    ft = pd.read_parquet(C.FORECAST_TABLE,
                         columns=KEY_COLS + ["decision_t_s", "window_s"])
    frames = pd.read_parquet(C.FLOW_FRAMES, columns=KEY_COLS)
    have = frames[KEY_COLS].drop_duplicates()
    ft = ft.merge(have, on=KEY_COLS, how="inner")
    if ft.empty:
        raise SystemExit("no extracted clips overlap the forecast table")
    rng = np.random.default_rng(seed)
    take = min(n, len(ft))
    idx = rng.choice(len(ft), size=take, replace=False)
    return ft.iloc[idx].reset_index(drop=True)


def _corr(div: np.ndarray, vrel: np.ndarray, mask: np.ndarray):
    from scipy.stats import spearmanr
    m = mask & np.isfinite(div) & np.isfinite(vrel)
    if m.sum() < 5:
        return int(m.sum()), None
    if floats_equal(div[m]) or floats_equal(vrel[m]):
        return int(m.sum()), None
    rho = spearmanr(div[m], vrel[m]).statistic
    return int(m.sum()), (float(rho) if np.isfinite(rho) else None)


def per_row_corr(row, frames_clip: pd.DataFrame) -> dict:
    t = frames_clip["t_rel"].to_numpy(float)
    div = frames_clip["flow_div_corridor_mean"].to_numpy(float)
    valid = frames_clip["flow_valid"].to_numpy(float) > 0.5
    rt, rv, rst = radar_series(row.clip_dir)
    good = np.isfinite(rv)
    if good.sum() >= 2:
        vrel = np.interp(t, rt[good], rv[good])
        lead = np.interp(t, rt, np.nan_to_num(rst, nan=0.0)) > 0.5
    else:
        vrel = np.full(t.shape, np.nan)
        lead = np.zeros(t.shape, bool)
    hi = row.decision_t_s
    lo = row.decision_t_s - row.window_s
    n, rho = _corr(div, vrel, valid & lead & (t >= lo - 1e-6) & (t <= hi + 1e-6))
    n_pre, rho_pre = _corr(div, vrel, valid & lead & (t <= hi + 1e-6))
    return {"n": n, "rho": rho, "n_pre": n_pre, "rho_pre": rho_pre,
            "n_lead_frames": int((valid & lead).sum())}


def render_frame(video: Path, frame_idx: int, tile_t: float, method: str):
    """Decode the pair ending at `frame_idx`, fit global motion, and draw the
    residual field + IMO boxes on the frame."""
    frames, _ = X.decode_frames(video, C.FLOW_HZ)
    pos = next((i for i, (idx, _, _) in enumerate(frames) if idx == frame_idx), None)
    if pos is None or pos == 0:
        return None, None
    bgr = frames[pos][2]
    prev = cv2.cvtColor(frames[pos - 1][2], cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if prev.shape != cur.shape:
        return None, None
    if method == "raft":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        fl = X.raft_flows([prev, cur], dev)
        if not fl or fl[0] is None:
            return None, None
        u, v = fl[0]
    else:
        fl = X.dis_flows([prev, cur])
        if not fl or fl[0] is None:
            return None, None
        u, v = fl[0]
    h, w = cur.shape[:2]
    valid, corridor = X._valid_and_corridor(h, w)
    M, inlier = X.fit_global(u, v, valid, corridor)
    if M is None:
        return None, None
    mu, mv = X.model_flow(M, h, w)
    ru, rv = u - mu, v - mv
    mag = np.sqrt(ru * ru + rv * rv)

    step = 24
    ys = np.arange(step, h - step, step)
    xs = np.arange(step, w - step, step)
    for y in ys:
        for x in xs:
            if not valid[y, x]:
                continue
            cv2.arrowedLine(bgr, (x, y), (int(x + mu[y, x]), int(y + mv[y, x])),
                            (0, 200, 0), 1, tipLength=0.3)
    thr = max(X.IMO_ABS_THRESH_PX, X.IMO_MED_MULT * float(np.median(mag[valid])))
    for y in ys:
        for x in xs:
            if valid[y, x] and mag[y, x] > thr:
                cv2.arrowedLine(bgr, (x, y), (int(x + ru[y, x]), int(y + rv[y, x])),
                                (0, 0, 255), 1, tipLength=0.3)

    binary = np.zeros((h, w), np.uint8)
    binary[valid & (mag > thr)] = 1
    n_imo = 0
    if binary.any():
        n, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = X.IMO_MIN_AREA_FRAC * h * w
        for lab in range(1, n):
            if float(stats[lab, cv2.CC_STAT_AREA]) < min_area:
                continue
            x0, y0 = int(stats[lab, cv2.CC_STAT_LEFT]), int(stats[lab, cv2.CC_STAT_TOP])
            bw, bh = int(stats[lab, cv2.CC_STAT_WIDTH]), int(stats[lab, cv2.CC_STAT_HEIGHT])
            cv2.rectangle(bgr, (x0, y0), (x0 + bw, y0 + bh), (255, 0, 0), 2)
            n_imo += 1
    cv2.line(bgr, (0, int((1 - X.HOOD_FRAC) * h)), (w, int((1 - X.HOOD_FRAC) * h)),
             (0, 255, 255), 1)
    cv2.putText(bgr, f"t={tile_t:+.0f}s inl={inlier:.2f} IMO={n_imo}", (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return bgr, n_imo


def draw_contact_sheet(rows: pd.DataFrame, frames: pd.DataFrame, out_png: Path,
                       method: str) -> list:
    picked = []
    for _, r in rows.iterrows():
        sub = frames[(frames[KEY_COLS] == r[KEY_COLS].values).all(axis=1)]
        if not sub.empty:
            picked.append((r, sub))
    if not picked:
        return []
    # prefer clips with the strongest flow activity so the sheet is informative
    picked = sorted(picked, key=lambda x: -float(x[1]["flow_res_mag_mean"].mean()))[:4]
    idx = pd.read_parquet(C.CLIP_INDEX, columns=KEY_COLS + ["clip_dir", "video"])
    sheet, summary = [], []
    for r, sub in picked:
        key = tuple(r[KEY_COLS].tolist())
        meta = idx[idx[KEY_COLS].apply(tuple, axis=1) == key]
        if meta.empty:
            continue
        video = Path(meta.iloc[0]["video"])
        tiles = []
        for t in GRID_TIMES:
            j = (sub["t_rel"] - t).abs().idxmin()
            fidx = int(sub.loc[j, "frame_idx"])
            tile, n_imo = render_frame(video, fidx, float(sub.loc[j, "t_rel"]), method)
            if tile is None:
                continue
            tile = cv2.resize(tile, (TILE_W, TILE_H))
            tiles.append(tile)
            summary.append({"clip": key, "t_rel": float(sub.loc[j, "t_rel"]),
                            "n_imo": int(n_imo)})
        if tiles:
            sheet.append(tiles)
    if not sheet:
        return []
    width = max(len(t) for t in sheet)
    canvas = np.zeros((TILE_H * len(sheet), TILE_W * width, 3), np.uint8)
    for i, tiles in enumerate(sheet):
        for j, tile in enumerate(tiles):
            canvas[i * TILE_H:(i + 1) * TILE_H, j * TILE_W:(j + 1) * TILE_W] = tile
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=40, help="number of (clip, window) rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", default=C.FLOW_METHOD, choices=["dis", "raft"],
                    help="optical-flow method used by extract_flow_frames.py")
    args = ap.parse_args()

    frames = pd.read_parquet(C.FLOW_FRAMES)
    ft_out = pd.read_parquet(C.FORECAST_TABLE_FLOW)
    schema = json.loads(C.FORECAST_SCHEMA_FLOW.read_text(encoding="utf-8"))
    flow_cols = [c for c in schema["features_numeric"] if c.startswith("flow_")]

    # --- assertions (no leakage, fully defined) -----------------------------
    assert float(frames["t_rel"].max()) <= 0.0 + 1e-6, "flow frames see the future"
    assert ft_out[flow_cols].notna().all().all(), "flow_* must be fully defined"
    assert len(ft_out) == int(schema["n_rows"]), "forecast_table_flow row count"

    rows = sample_rows(args.n, args.seed)
    clip_dir = pd.read_parquet(C.CLIP_INDEX, columns=KEY_COLS + ["clip_dir"])
    rows = rows.merge(clip_dir, on=KEY_COLS, how="left")

    results = []
    for _, r in rows.iterrows():
        sub = frames[(frames[KEY_COLS] == r[KEY_COLS].values).all(axis=1)]
        results.append({**{k: r[k] for k in KEY_COLS},
                        "decision_t_s": float(r.decision_t_s),
                        **per_row_corr(r, sub)})

    rhos = [x["rho"] for x in results if x["rho"] is not None]
    negative = int(sum(1 for x in rhos if x < 0))
    rho_pre = [x["rho_pre"] for x in results if x.get("rho_pre") is not None]

    # --- jitter confound -----------------------------------------------------
    onset = frames["flow_onset"].to_numpy(float) > 0.5
    rot = np.abs(frames["flow_global_rot"].to_numpy(float))
    valid = frames["flow_valid"].to_numpy(float) > 0.5
    q90 = float(np.nanpercentile(rot[valid], 90)) if valid.any() else np.inf
    high = valid & (rot >= q90)
    rate_high = float(onset[high].mean()) if high.any() else 0.0
    rate_low = float(onset[valid & ~high].mean()) if (valid & ~high).any() else 0.0

    payload = {
        "n_sampled": len(results),
        "n_with_correlation": len(rhos),
        "median_spearman": float(np.median(rhos)) if rhos else None,
        "negative_fraction": (negative / len(rhos)) if rhos else None,
        "pass": bool(rhos and np.median(rhos) < -0.2 and negative / len(rhos) > 0.5),
        "median_spearman_pre_window": float(np.median(rho_pre)) if rho_pre else None,
        "n_pre_window": len(rho_pre),
        "jitter": {
            "onset_rate_top_decile_abs_rot": rate_high,
            "onset_rate_rest": rate_low,
            "rot_p90": q90,
            "pass": bool(rate_high <= max(3.0 * rate_low, rate_low + 0.02)),
        },
        "per_row": results,
    }
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "flow_alignment.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(f"[qa] sampled {payload['n_sampled']} rows; "
          f"{payload['n_with_correlation']} with a usable correlation")
    print(f"[qa] median Spearman(div_corridor, leadOne.vRel) = {payload['median_spearman']}")
    print(f"[qa] negative fraction = {payload['negative_fraction']}  "
          f"PASS={payload['pass']}")
    print(f"[qa] jitter: onset rate top-decile |rot|={rate_high:.3f} vs "
          f"rest={rate_low:.3f}  PASS={payload['jitter']['pass']}")
    print(f"[qa] wrote {C.RESULTS / 'flow_alignment.json'}")

    summary = draw_contact_sheet(rows, frames, C.RESULTS / "flow_qa_grid.png",
                                 args.method)
    if summary:
        sizes = [s["n_imo"] for s in summary]
        print(f"[qa] wrote {C.RESULTS / 'flow_qa_grid.png'} ({len(summary)} tiles); "
              f"IMO/tile mean={np.mean(sizes):.2f} max={max(sizes)}")
    else:
        print("[qa] no contact sheet produced")


if __name__ == "__main__":
    main()
