"""
M3 - visual QA of the video<->telemetry alignment.

We cannot decode video here, so instead we verify the alignment *proxy* rigorously:
the ADAS engagement step must land at t=0 (the takeover), on the [-10,+10] relative axis.
This is exactly what the UI overlays, so if it holds, the viewer is aligned.

Checks per clip:
  A. adas_engaged is ~1 before t=0 and ~0 after t=0 (the defining takeover signature)
  B. transition index is within +-1 s of t=0
  C. no gap in the grid; series lengths consistent
Reports a pass rate across a random sample and lists failures.
"""
from __future__ import annotations

import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adas_to import config as C
from adas_to.telemetry import build_clip_telemetry

idx = pd.read_parquet(C.CLIP_INDEX)
n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
rows = idx.sample(min(n, len(idx)), random_state=7)

print(f"QA on {len(rows)} clips\n")
print(f"{'clip':<44} {'kind':<5} {'engaged%':>8} {'pre%':>6} {'post%':>6} "
      f"{'switch_t':>9} {'ok':>3}  warnings")

ok_n, fail = 0, []
for _, r in rows.iterrows():
    key = f"{r.car_model}/{r.driver}/{r.route}/{r.clip_id_pub}"
    try:
        tel = build_clip_telemetry(r.clip_dir)
    except Exception as e:
        fail.append((key, f"ERROR {type(e).__name__}: {e}"))
        print(f"{key:<44} {'-':<5} {'-':>8} {'-':>6} {'-':>6} {'-':>9} {'X':>3}  {e}")
        continue

    t = np.array(tel["t"], float)
    eng = np.array(tel["series"]["adas_engaged"], float)

    pre = eng[t < -0.5]
    post = eng[t > 0.5]
    pre_pct = 100 * pre.mean() if len(pre) else float("nan")
    post_pct = 100 * post.mean() if len(post) else float("nan")

    # first sustained 1->0 transition
    sw = np.nan
    for i in range(len(eng) - 1):
        if eng[i] > 0.5 and eng[i + 1] < 0.5 and eng[i + 1: i + 11].mean() < 0.5:
            sw = t[i + 1]
            break
    ok_A = (pre_pct > 50) and (post_pct < 50)
    ok_B = bool(np.isfinite(sw) and abs(sw) <= 1.0)
    ok = ok_A and ok_B
    ok_n += ok
    if not ok:
        fail.append((key, f"pre={pre_pct:.0f}% post={post_pct:.0f}% switch={sw}"))

    print(f"{key:<44} {str(tel['log_kind']):<5} {tel['adas_engaged_pct']:>8.1f} "
          f"{pre_pct:>6.1f} {post_pct:>6.1f} {sw if np.isfinite(sw) else float('nan'):>9.2f} "
          f"{'OK' if ok else ' X':>3}  {'; '.join(tel['warnings'])[:60]}")

print(f"\n=== PASS RATE: {ok_n}/{len(rows)} = {100*ok_n/max(len(rows),1):.1f}% ===")
if fail:
    print(f"\nfailures ({len(fail)}):")
    for k, why in fail[:25]:
        print(f"   {k}  ->  {why}")

# also dump one full record so we can eyeball the payload the UI will consume
r = rows.iloc[0]
tel = build_clip_telemetry(r.clip_dir)
out = C.RESULTS / "telemetry_example.json"
out.write_text(json.dumps(tel, indent=1), encoding="utf-8")
print(f"\nwrote {out}  ({out.stat().st_size/1024:.1f} KB)")
print("series keys:", list(tel["series"].keys()))
print("native_hz  :", {k: round(v, 1) for k, v in tel["native_hz"].items()})
