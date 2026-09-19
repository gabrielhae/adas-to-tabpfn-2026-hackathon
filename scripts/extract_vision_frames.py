"""
Vision stage 1 - per-frame YOLOv8n scene scalars for the sliding forecast.

Decodes each clip's `takeover.mp4` ONCE, sequentially, keeping every
`round(fps / VISION_FPS)`-th frame (5 Hz by default), then runs YOLOv8n on the
retained frames in batches. Emits one row per (clip, frame):

    car_model, driver, route, clip_id_pub, vid_kind, t_rel, frame_idx,
    img_w, img_h, n_vehicles, n_corridor_vehicles, n_person, n_bicycle,
    n_traffic_light, n_stop_sign, lead_present, lead_conf, lead_area_frac,
    lead_cx_offset, lead_y2_norm, brightness_mean, motion_mean

`t_rel = frame_idx / fps - 10.0` is clip-relative video time with the takeover at
0, matching `build_clip_telemetry`'s `t` (both are `t_video - 10.0`). Only
`t_rel in [-10, +10]` rows are kept.

Writes:
    data/derived/vision_frames.parquet   (working format)
    data/derived/vision_frames.csv       (human/portable export)

Usage:
    python scripts/extract_vision_frames.py --limit 5        # smoke test
    python scripts/extract_vision_frames.py --resume         # skip done clips
    python scripts/extract_vision_frames.py --fps 5 --device 0
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

VEHICLE_CLASSES = {2, 3, 5, 7}          # car, motorcycle, bus, truck
PERSON_CLASS = 0
BICYCLE_CLASS = 1
TRAFFIC_LIGHT_CLASS = 9
STOP_SIGN_CLASS = 11

CORRIDOR_LEFT = 0.25
CORRIDOR_RIGHT = 0.75
NEAR_FIELD_Y2 = 0.55
CORRIDOR_MIN_Y2 = 0.30
LEAD_IOU_CONTINUITY = 0.1

FRAME_COLS = [
    "car_model", "driver", "route", "clip_id_pub", "vid_kind",
    "t_rel", "frame_idx", "img_w", "img_h",
    "n_vehicles", "n_corridor_vehicles", "n_person", "n_bicycle",
    "n_traffic_light", "n_stop_sign",
    "lead_present", "lead_conf", "lead_area_frac", "lead_cx_offset", "lead_y2_norm",
    "brightness_mean", "motion_mean",
]


def resolve_weights(name: str) -> str:
    p = Path(name)
    if p.is_absolute() or p.parent != Path("."):
        return str(p)
    return str(C.VISION_DIR / name)


def decode_frames(video: Path, fps_target: float):
    """Sequentially decode one clip, returning (frame_idx, t_rel, bgr) retained frames."""
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
            if t_rel < -10.0 or t_rel > 10.0:
                continue
            out.append((idx, float(t_rel), frame))
    finally:
        cap.release()
    return out, fps


def detect_batch(model, frames, *, conf: float, imgsz: int, device, batch: int = 16):
    """Return a list (one per frame) of detection dicts with absolute xyxy boxes."""
    per_frame = []
    for i in range(0, len(frames), batch):
        chunk = frames[i:i + batch]
        results = model.predict(chunk, imgsz=imgsz, conf=conf, device=device,
                                verbose=False)
        for r in results:
            dets = []
            b = r.boxes
            if b is not None and len(b) > 0:
                xyxy = b.xyxy.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                cf = b.conf.cpu().numpy()
                for (x1, y1, x2, y2), c, p in zip(xyxy, cls, cf):
                    dets.append({"cls": int(c), "conf": float(p),
                                 "x1": float(x1), "y1": float(y1),
                                 "x2": float(x2), "y2": float(y2)})
            per_frame.append(dets)
    return per_frame


def _iou(a: dict, b: dict) -> float:
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]) + \
        (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def frame_scalars(dets, frame, prev_gray, t_rel, frame_idx, vid_kind, prev_lead=None):
    import cv2

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    motion = NAN if prev_gray is None else float(
        np.abs(gray.astype(np.int16) - prev_gray.astype(np.int16)).mean())

    n_person = n_bicycle = n_tl = n_stop = 0
    vehicles = []
    for d in dets:
        c = d["cls"]
        if c in VEHICLE_CLASSES:
            vehicles.append(d)
        elif c == PERSON_CLASS:
            n_person += 1
        elif c == BICYCLE_CLASS:
            n_bicycle += 1
        elif c == TRAFFIC_LIGHT_CLASS:
            n_tl += 1
        elif c == STOP_SIGN_CLASS:
            n_stop += 1

    n_corridor = 0
    for d in vehicles:
        cx = (d["x1"] + d["x2"]) / 2.0
        if CORRIDOR_LEFT * w <= cx <= CORRIDOR_RIGHT * w and d["y2"] >= CORRIDOR_MIN_Y2 * h:
            n_corridor += 1

    # Any vehicle centred in the ego-lane corridor is a lead candidate, regardless
    # of how near it is (the old rule dropped distant leads because their box sits
    # high in the frame). Prefer the previous frame's lead when it still overlaps,
    # which keeps the identity stable frame-to-frame.
    corridor = [d for d in vehicles
                if CORRIDOR_LEFT * w <= (d["x1"] + d["x2"]) / 2.0 <= CORRIDOR_RIGHT * w]
    best = None
    if prev_lead is not None:
        overlap = [( _iou(d, prev_lead), d) for d in corridor]
        overlap = [x for x in overlap if x[0] >= LEAD_IOU_CONTINUITY]
        if overlap:
            best = max(overlap, key=lambda x: x[0])[1]
    if best is None and corridor:
        best = max(corridor, key=lambda d: max(0.0, d["x2"] - d["x1"])
                   * max(0.0, d["y2"] - d["y1"]))

    lead_present = 0.0
    lead_conf = lead_area = lead_cx = lead_y2 = NAN
    if best is not None:
        lead_present = 1.0
        lead_conf = best["conf"]
        lead_area = max(0.0, best["x2"] - best["x1"]) * \
            max(0.0, best["y2"] - best["y1"]) / (w * h)
        lead_cx = ((best["x1"] + best["x2"]) / 2.0 - w / 2.0) / w
        lead_y2 = best["y2"] / h

    scalars = {
        "t_rel": t_rel,
        "frame_idx": int(frame_idx),
        "img_w": int(w),
        "img_h": int(h),
        "n_vehicles": len(vehicles),
        "n_corridor_vehicles": n_corridor,
        "n_person": n_person,
        "n_bicycle": n_bicycle,
        "n_traffic_light": n_tl,
        "n_stop_sign": n_stop,
        "lead_present": lead_present,
        "lead_conf": lead_conf,
        "lead_area_frac": lead_area,
        "lead_cx_offset": lead_cx,
        "lead_y2_norm": lead_y2,
        "brightness_mean": brightness,
        "motion_mean": motion,
    }
    return scalars, best


def process_clip(clip, model, *, fps_target, conf, imgsz, device):
    frames, fps = decode_frames(Path(clip.video), fps_target)
    if not frames:
        raise RuntimeError("no frames decoded")
    bgr = [f for _, _, f in frames]
    import cv2
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in bgr]
    dets = detect_batch(model, bgr, conf=conf, imgsz=imgsz, device=device)
    rows = []
    prev_lead = None
    for i, (idx, t_rel, _) in enumerate(frames):
        prev = grays[i - 1] if i > 0 else None
        s, prev_lead = frame_scalars(dets[i], bgr[i], prev, t_rel, idx,
                                     clip.vid_kind, prev_lead)
        s["car_model"] = clip.car_model
        s["driver"] = clip.driver
        s["route"] = clip.route
        s["clip_id_pub"] = clip.clip_id_pub
        s["vid_kind"] = clip.vid_kind
        rows.append(s)
    return rows, fps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process N clips (smoke test)")
    ap.add_argument("--fps", type=float, default=C.VISION_FPS, help="sampling rate (Hz)")
    ap.add_argument("--device", default=None, help="'0' for GPU, 'cpu' for CPU (default auto)")
    ap.add_argument("--weights", default=C.YOLO_MODEL)
    ap.add_argument("--conf", type=float, default=C.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
    ap.add_argument("--resume", action="store_true",
                    help="skip clips already present in vision_frames.parquet")
    args = ap.parse_args()

    forecast = pd.read_parquet(C.FORECAST_TABLE, columns=KEY_COLS)
    wanted = set(map(tuple, forecast.drop_duplicates()[KEY_COLS].to_numpy()))
    idx = pd.read_parquet(C.CLIP_INDEX)
    idx = idx[idx[KEY_COLS].apply(tuple, axis=1).isin(wanted)].reset_index(drop=True)

    existing = None
    if args.resume and C.VISION_FRAMES.exists():
        existing = pd.read_parquet(C.VISION_FRAMES)
        done = set(map(tuple, existing[KEY_COLS].drop_duplicates().to_numpy()))
        idx = idx[~idx[KEY_COLS].apply(tuple, axis=1).isin(done)].reset_index(drop=True)
        print(f"[vision] resume: {len(done)} clips already extracted, "
              f"{len(idx)} remaining")
    if args.limit:
        idx = idx.head(args.limit).reset_index(drop=True)
    if idx.empty:
        print("[vision] nothing to do")
        return

    from ultralytics import YOLO
    weights = resolve_weights(args.weights)
    device = args.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[vision] {len(idx)} clips @ {args.fps} Hz  weights={weights}  device={device}")
    model = YOLO(weights)

    t0 = time.time()
    records, failed = [], []
    for pos, clip in enumerate(idx.itertuples(index=False)):
        try:
            rows, fps = process_clip(clip, model, fps_target=args.fps,
                                     conf=args.conf, imgsz=args.imgsz, device=device)
            records.extend(rows)
        except Exception as e:
            failed.append((clip.car_model, clip.driver, clip.route, clip.clip_id_pub, str(e)))
        if (pos + 1) % 25 == 0 or pos + 1 == len(idx):
            print(f"      {pos + 1}/{len(idx)} clips  ({time.time() - t0:.0f}s)")

    new = pd.DataFrame(records)
    if new.empty:
        raise SystemExit("[vision] no frames extracted; nothing written")
    new = new[FRAME_COLS]
    if existing is not None and not existing.empty:
        out = pd.concat([existing[FRAME_COLS], new], ignore_index=True)
    else:
        out = new
    out = out.sort_values(KEY_COLS + ["frame_idx"]).reset_index(drop=True)

    out.to_parquet(C.VISION_FRAMES, index=False)
    out.to_csv(C.VISION_FRAMES_CSV, index=False)

    lead_rate = float(out["lead_present"].mean())
    print(f"[vision] {len(out):,} frames from {out[KEY_COLS].drop_duplicates().shape[0]} clips "
          f"({len(failed)} failed) -> {C.VISION_FRAMES}")
    print(f"[vision] mean detections/frame={out['n_vehicles'].mean():.2f}  "
          f"lead-present rate={lead_rate:.3f}  "
          f"mean img={out['img_w'].mean():.0f}x{out['img_h'].mean():.0f}")
    for f in failed[:10]:
        print(f"      skip {f[0]}/{f[1]}/{f[2]}/{f[3]}: {f[4]}")


if __name__ == "__main__":
    main()
