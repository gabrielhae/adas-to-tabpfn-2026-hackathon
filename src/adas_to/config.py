"""
ADAS-TO configuration and path resolution.
Single source of truth for paths across the project.
"""
from __future__ import annotations
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# --- inputs -----------------------------------------------------------------
SAMPLE_DIR = ROOT / "data" / "ADAS-TO-Sample" / "ADAS-TO-Sample"
if not SAMPLE_DIR.is_dir():
    SAMPLE_DIR = ROOT / "data" / "ADAS-TO-Sample"

REPO_CODE = ROOT / "ADAS-TO" / "Code"
RECON = ROOT / "_recon"

ID_MAP = RECON / "id_map.csv"
ANALYSIS_MASTER = REPO_CODE / "stats_output" / "analysis_master.csv"
PER_CLIP = REPO_CODE / "stats_output" / "per_clip.csv"

# --- outputs ----------------------------------------------------------------
DERIVED = ROOT / "data" / "derived"
RESULTS = ROOT / "results"
for _p in (DERIVED, RESULTS):
    _p.mkdir(parents=True, exist_ok=True)

MODEL_TABLE = DERIVED / "model_table.parquet"
CLIP_INDEX = DERIVED / "clip_index.parquet"
TELEMETRY_DIR = DERIVED / "telemetry"          # per-clip 20 Hz json
SCHEMA_JSON = DERIVED / "schema.json"

# --- experiment constants (documented, verified) ----------------------------
RESAMPLE_HZ = 20            # NEVER above 20: qlog model data is 1.3 Hz
CLIP_SECONDS = 20.0
EVENT_OFFSET_S = 10.0       # event = csv_min + 10.0 s  (98.2% within 0.25 s)

# repo analysis windows (from ADAS-TO/Code/configs/analysis_thresholds.yaml)
PRE_WINDOW = (-5.0, 0.0)
POST_WINDOW = (0.0, 5.0)
TRIGGER_WINDOW = (-3.0, 0.5)   # overlaps PRE -> source of label leakage

# leakage-safe windows we recompute ourselves
SAFE_PRE_WINDOW = (-10.0, -3.0)   # strictly before any takeover action
SAFE_POST_WINDOW = (0.0, 5.0)

# robust TTC/THW guards, from ADAS-TO/Code/configs/analysis_thresholds.yaml
ANALYSIS = {
    "closing_min_mps": 0.5,     # only compute TTC when |vRel| exceeds this
    "drel_min_m": 5.0,          # only compute TTC when dRel exceeds this
    "ttc_cap_s": 100.0,         # cap the TTC tail
    "min_speed_thw_mps": 0.5,   # avoid absurd THW at near-zero speed
}

KEY_COLS = ["car_model", "dongle_id", "route_id", "clip_id"]

# --- TabPFN ------------------------------------------------------------------
TABPFN_TOKEN_ENV = "TABPFN_TOKEN"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def tabpfn_token() -> str | None:
    return os.environ.get(TABPFN_TOKEN_ENV) or None
