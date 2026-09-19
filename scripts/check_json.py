"""Verify no NaN/Infinity can leak into the JSON payload (browsers reject both)."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adas_to.telemetry import build_clip_telemetry

idx = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "derived" / "clip_index.parquet")
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = idx.sample(min(n, len(idx)), random_state=11)

bad, ok, err = [], 0, 0
for _, r in rows.iterrows():
    key = f"{r.car_model}/{r.driver}/{r.route}/{r.clip_id_pub}"
    try:
        t = build_clip_telemetry(r.clip_dir)
    except Exception as e:
        err += 1
        bad.append((key, f"ERROR {type(e).__name__}: {e}"))
        continue
    # strict: must serialise with allow_nan=False
    try:
        json.dumps(t, allow_nan=False)
        ok += 1
    except ValueError as e:
        bad.append((key, f"JSON {e}"))

print(f"checked {len(rows)} clips: strict-json OK={ok}  bad={len(bad)}  errors={err}")
for k, why in bad[:20]:
    print(f"   {k} -> {why}")

# also confirm the derived series behave sensibly on one clip with a lead vehicle
for _, r in rows.iterrows():
    t = build_clip_telemetry(r.clip_dir)
    s = t["series"]
    if s.get("ttc") and any(x is not None for x in s["ttc"]):
        import numpy as np
        ttc = np.array([x if x is not None else np.nan for x in s["ttc"]], float)
        thw = np.array([x if x is not None else np.nan for x in s["thw"]], float)
        print(f"\nexample with lead: {r.car_model}/{r.driver}/{r.route}/{r.clip_id_pub}")
        print(f"  TTC  n={np.isfinite(ttc).sum():3d}  min={np.nanmin(ttc):.2f}  "
              f"p50={np.nanmedian(ttc):.2f}")
        print(f"  THW  n={np.isfinite(thw).sum():3d}  min={np.nanmin(thw):.2f}  "
              f"p50={np.nanmedian(thw):.2f}")
        print(f"  engaged={t['adas_engaged_pct']:.1f}%  samples={t['n_samples']}  "
              f"warnings={len(t['warnings'])}")
        break
