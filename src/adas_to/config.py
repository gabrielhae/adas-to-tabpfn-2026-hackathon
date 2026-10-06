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
VISION_DIR = DERIVED / "vision"                # detector weights + caches
FLOW_DIR = DERIVED / "flow"                    # optical-flow caches
for _p in (DERIVED, RESULTS, VISION_DIR, FLOW_DIR):
    _p.mkdir(parents=True, exist_ok=True)

MODEL_TABLE = DERIVED / "model_table.parquet"
CLIP_INDEX = DERIVED / "clip_index.parquet"
TELEMETRY_DIR = DERIVED / "telemetry"          # per-clip 20 Hz json
SCHEMA_JSON = DERIVED / "schema.json"
FORECAST_TABLE = DERIVED / "forecast_table.parquet"
FORECAST_SCHEMA = DERIVED / "forecast_schema.json"

# --- train / validation / test split ----------------------------------------
# Driver-disjoint (the paper's stated protocol). See src/adas_to/splits.py.
SPLIT_ASSIGNMENTS = DERIVED / "splits.parquet"   # dongle_id -> split
SPLIT_META = DERIVED / "split_meta.json"         # summary written by build_splits.py
TEST_SCORES = RESULTS / "test_scores.json"       # batch score over the test split
SPLIT_FRACTIONS = {"train": 0.6, "val": 0.2, "test": 0.2}
SPLIT_SEED = 0

# --- vision (YOLOv8n structured geometry) -----------------------------------
VISION_FRAMES = DERIVED / "vision_frames.parquet"
VISION_FRAMES_CSV = DERIVED / "vision_frames.csv"
VISION_TABLE = DERIVED / "vision_table.parquet"
FORECAST_TABLE_VISION = DERIVED / "forecast_table_vision.parquet"
FORECAST_SCHEMA_VISION = DERIVED / "forecast_schema_vision.json"
VISION_FPS = 5.0                                # decode/detection sampling rate (Hz)
YOLO_MODEL = "yolov8n.pt"
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# --- flow (ego-motion-compensated optical flow / independently moving objects) --
FLOW_FRAMES = DERIVED / "flow_frames.parquet"
FLOW_FRAMES_CSV = DERIVED / "flow_frames.csv"
FLOW_TABLE = DERIVED / "flow_table.parquet"
FORECAST_TABLE_FLOW = DERIVED / "forecast_table_flow.parquet"
FORECAST_SCHEMA_FLOW = DERIVED / "forecast_schema_flow.json"
FLOW_HZ = 10.0                    # decode/flow sampling rate (Hz)
FLOW_METHOD = "raft"              # "dis" (fast, noisy at 526x330) | "raft" (torchvision)
FLOW_RAFT_W, FLOW_RAFT_H = 256, 160   # RAFT working resolution (upscaled to the frame)
FLOW_RANSAC_REPROJ_PX = 3.0       # estimateAffinePartial2D RANSAC threshold
FLOW_HOOD_FRAC = 0.15             # bottom fraction of the frame masked from the global fit

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

# --- forecast: end the feature window at t = -DELTA -------------------------
# The only change that turns M1 into a forecast: features are aggregated over
# [-10, -DELTA] (strictly before the takeover) instead of the repo's [-5, 0].
# Targets stay in the post window [0, +5].
FORECAST_DELTA_S = float(os.environ.get("ADAS_TO_DELTA_S", "3.0"))
FEATURE_WINDOW = (-10.0, -FORECAST_DELTA_S)   # feature window ends at -DELTA

# robust TTC/THW guards, from ADAS-TO/Code/configs/analysis_thresholds.yaml
ANALYSIS = {
    "closing_min_mps": 0.5,     # only compute TTC when |vRel| exceeds this
    "drel_min_m": 5.0,          # only compute TTC when dRel exceeds this
    "ttc_cap_s": 100.0,         # cap the TTC tail
    "min_speed_thw_mps": 0.5,   # avoid absurd THW at near-zero speed
}

KEY_COLS = ["car_model", "dongle_id", "route_id", "clip_id"]
GROUP_COL = "dongle_id"          # the driver key used for all group-disjoint splits

# --- models shown side-by-side in the explorer (order preserved) -------------
COMPARE_MODELS = ["tabpfn", "lightgbm", "xgboost", "catboost"]
COMPARE_MODEL_LABELS = {
    "tabpfn": "TabPFN",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "logistic": "Logistic",
}

# --- TabPFN ------------------------------------------------------------------
TABPFN_TOKEN_ENV = "TABPFN_TOKEN"


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or None


def tabpfn_token() -> str | None:
    return os.environ.get(TABPFN_TOKEN_ENV) or None
