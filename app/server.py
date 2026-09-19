"""
ADAS-TO Explorer - FastAPI backend.

Endpoints
---------
GET /api/index                 clip list (1,591) + joined labels where available
GET /api/clip/{model}/{drv}/{route}/{clip}   metadata + labels + 20 Hz telemetry
GET /api/media/{model}/{drv}/{route}/{clip}  takeover.mp4 with HTTP Range support
GET /api/stats                 distributions for the explorer view
GET /api/model                 available models + TabPFN readiness
GET /api/predict?...           model prediction for a clip (feature vector)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adas_to import config as C
from adas_to import model as M
from adas_to.telemetry import build_clip_telemetry

app = FastAPI(title="ADAS-TO Explorer")

# ------------------------------------------------------------------ load once
IDX = pd.read_parquet(C.CLIP_INDEX)
MODEL_TABLE = pd.read_parquet(C.MODEL_TABLE) if C.MODEL_TABLE.exists() else pd.DataFrame()
SCHEMA = json.loads(C.SCHEMA_JSON.read_text(encoding="utf-8")) if C.SCHEMA_JSON.exists() else {}

IDX["key"] = (IDX.car_model + "/" + IDX.driver + "/" + IDX.route + "/"
              + IDX.clip_id_pub.astype(str))

# attach labels where re-identified
LABELS = None
if len(MODEL_TABLE):
    lt = MODEL_TABLE.copy()
    lt["key"] = (lt.car_model + "/" + lt.driver + "/" + lt.route + "/"
                 + lt.clip_id_pub.astype(str))
    keep = ["key", "primary_trigger", "scenario", "brand", "powertrain",
            "post_maneuver_type", "risk_score", "maneuver_score"]
    LABELS = lt[[c for c in keep if c in lt.columns]]

VIEW = IDX.merge(LABELS, on="key", how="left") if LABELS is not None else IDX

_CACHE: dict[str, dict] = {}


def _pyval(v):
    """Convert numpy/pandas scalars to native Python so FastAPI can serialise them."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _row(key: str):
    m = VIEW[VIEW.key == key]
    if not len(m):
        raise HTTPException(404, f"clip not found: {key}")
    return m.iloc[0]


# ------------------------------------------------------------------ endpoints
@app.get("/api/model")
def api_model():
    ok, why = M.tabpfn_available()
    return {"tabpfn_available": ok, "reason": why,
            "available_models": (["tabpfn", "lightgbm", "logistic"] if ok
                                 else ["lightgbm", "logistic"]),
            "headline_model": "tabpfn" if ok else "lightgbm",
            "note": "TabPFN needs a one-time license acceptance; set TABPFN_TOKEN."}


@app.get("/api/index")
def api_index(brand: str | None = None, log_kind: str | None = None,
              trigger: str | None = None, scenario: str | None = None,
              model: str | None = None, labelled_only: bool = False,
              q: str | None = None, limit: int = 300):
    d = VIEW
    if brand:      d = d[d.brand == brand]
    if log_kind:   d = d[d.log_kind == log_kind]
    if trigger:    d = d[d.primary_trigger == trigger]
    if scenario:   d = d[d.scenario == scenario]
    if model:      d = d[d.car_model == model]
    if labelled_only: d = d[d.primary_trigger.notna()]
    if q:
        d = d[d.key.str.contains(q, case=False, na=False)]
    cols = ["key", "car_model", "driver", "route", "clip_id_pub", "log_kind", "log_hz",
            "clip_dur_s", "primary_trigger", "scenario", "brand", "powertrain",
            "post_maneuver_type", "risk_score"]
    cols = [c for c in cols if c in d.columns]
    sub = d[cols].head(limit)
    # NaN is not valid JSON -> round-trip through pandas' serialiser which emits null
    records = json.loads(sub.to_json(orient="records", date_format="iso"))
    return {"total": int(len(d)), "returned": int(len(sub)), "clips": records}


@app.get("/api/facets")
def api_facets():
    def vc(c):
        if c not in VIEW.columns:
            return {}
        return {str(k): int(v) for k, v in VIEW[c].value_counts(dropna=True).items()}
    models = VIEW.car_model.value_counts()
    return {"brand": vc("brand"), "log_kind": vc("log_kind"),
            "primary_trigger": vc("primary_trigger"), "scenario": vc("scenario"),
            "post_maneuver_type": vc("post_maneuver_type"),
            "powertrain": vc("powertrain"),
            "car_model": {str(k): int(v) for k, v in models.head(60).items()},
            "n_clips": int(len(VIEW)),
            "n_labelled": int(VIEW.primary_trigger.notna().sum())}


@app.get("/api/clip/{car_model}/{driver}/{route}/{clip_id}")
def api_clip(car_model: str, driver: str, route: str, clip_id: str):
    key = f"{car_model}/{driver}/{route}/{clip_id}"
    r = _row(key)
    if key not in _CACHE:
        try:
            _CACHE[key] = build_clip_telemetry(r.clip_dir)
        except Exception as e:
            raise HTTPException(500, f"telemetry failed: {e}")
    tel = _CACHE[key]
    meta = {k: _pyval(r.get(k))
            for k in ["car_model", "driver", "route", "clip_id_pub", "log_kind", "log_hz",
                      "vid_kind", "camera_fps", "clip_dur_s", "primary_trigger", "scenario",
                      "brand", "powertrain", "post_maneuver_type", "risk_score",
                      "maneuver_score"]}
    return {"key": key, "meta": meta, "telemetry": tel,
            "labelled": bool(pd.notna(r.get("primary_trigger")))}


@app.get("/api/media/{car_model}/{driver}/{route}/{clip_id}")
def api_media(car_model: str, driver: str, route: str, clip_id: str, request: Request):
    r = _row(f"{car_model}/{driver}/{route}/{clip_id}")
    path = Path(r.video)
    if not path.exists():
        raise HTTPException(404, "video missing")
    size = path.stat().st_size
    rng = request.headers.get("range")
    if not rng:
        return FileResponse(path, media_type="video/mp4",
                            headers={"Accept-Ranges": "bytes"})
    try:
        unit, _, spec = rng.partition("=")
        start_s, _, end_s = spec.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else size - 1
    except Exception:
        raise HTTPException(416, "bad range")
    start = max(0, start)
    end = min(end, size - 1)
    if start > end:
        raise HTTPException(416, "unsatisfiable range")
    length = end - start + 1
    with open(path, "rb") as f:
        f.seek(start)
        body = f.read(length)
    return Response(content=body, status_code=206, media_type="video/mp4",
                    headers={"Content-Range": f"bytes {start}-{end}/{size}",
                             "Accept-Ranges": "bytes",
                             "Content-Length": str(length)})


@app.get("/api/stats")
def api_stats():
    d = VIEW
    out: dict = {"n_clips": int(len(d))}
    if "primary_trigger" in d.columns:
        out["primary_trigger"] = d.primary_trigger.value_counts(dropna=True).to_dict()
    if "scenario" in d.columns:
        out["scenario"] = d.scenario.value_counts(dropna=True).to_dict()
    if "post_maneuver_type" in d.columns:
        out["post_maneuver_type"] = d.post_maneuver_type.value_counts(dropna=True).to_dict()
    if "brand" in d.columns:
        out["brand"] = d.brand.value_counts(dropna=True).head(25).to_dict()
    if len(MODEL_TABLE):
        mt = MODEL_TABLE
        num = [c for c in ["risk_score", "maneuver_score", "speed_mps",
                           "post_max_abs_steer_torque"] if c in mt.columns]
        if num:
            q = mt[num].describe(percentiles=[.05, .25, .5, .75, .95]).round(3)
            out["quantiles"] = json.loads(q.to_json())
    return JSONResponse(out)


@app.get("/api/predict")
def api_predict(car_model: str, driver: str, route: str, clip_id: str,
                target: str = "post_maneuver_type",
                model_kind: str = "auto"):
    """Fit on all other drivers, predict this clip. Leave-driver-out, no leakage."""
    key = f"{car_model}/{driver}/{route}/{clip_id}"
    if not len(MODEL_TABLE):
        raise HTTPException(503, "model table unavailable")
    mt = MODEL_TABLE.copy()
    mt["key"] = (mt.car_model + "/" + mt.driver + "/" + mt.route + "/"
                 + mt.clip_id_pub.astype(str))
    row = mt[mt.key == key]
    if not len(row):
        raise HTTPException(404, "clip not in model table (no labels available)")
    if target not in SCHEMA.get("targets_regression", []) + [SCHEMA["target_classification"]]:
        raise HTTPException(400, f"unsupported target: {target}")

    task = "regression" if target in SCHEMA.get("targets_regression", []) else "classification"
    train = mt[mt.dongle_id != row.iloc[0].dongle_id]
    P = M.prepare(train, SCHEMA, target)
    ytr = P.y if task == "classification" else pd.to_numeric(P.y, errors="coerce")
    if task == "classification":
        ytr = M.collapse_rare(ytr, 10)
    Xtr, Xte = M.encode(P.X, M.prepare(row, SCHEMA, target).X, P.categorical)

    if model_kind == "auto":
        ok, _ = M.tabpfn_available()
        model_kind = "tabpfn" if ok else "lightgbm"
    est = M.make_model(model_kind, task=task)
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    resp = {"key": key, "target": target, "task": task, "model": model_kind,
            "prediction": (pred[0].item() if hasattr(pred[0], "item") else pred[0]),
            "degenerate": bool(len(train) < 100 or row.iloc[0].dongle_id
                               not in mt.dongle_id.values)}
    if task == "classification" and hasattr(est, "predict_proba"):
        pr = est.predict_proba(Xte)[0]
        resp["probabilities"] = {str(c): float(p) for c, p in zip(est.classes_, pr)}
        resp["actual"] = str(row.iloc[0][target])
    return resp


# ------------------------------------------------------------------ static
static_dir = Path(__file__).parent / "static"
if static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
