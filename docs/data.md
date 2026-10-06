# The data we actually have

| | |
|---|---|
| Source | `HenryYHW/ADAS-TO-Sample` |
| Clips on disk | **1,591** (the dataset card says 1,570) |
| Size / files | 3.21 GiB, 15,912 files |
| Structure | `<CAR_MODEL>/<driver_XXX>/<route_XXX>/<clip_id>/` × 10 files |
| Per clip | 8 CSVs + `meta.json` + `takeover.mp4` (20.00 s, ~1.4 MiB, 20 fps) |
| Coverage | **163 car models · 283 drivers · 1,238 routes** |
| Log rate | 989 `qlog` (10 Hz) · 602 `rlog` (100 Hz) |
| Labelled | **1,043 clips (65.6%)** — see below |

## Why only 1,043 of 1,591 clips have labels

The public release **re-anonymised the identifiers**. The dataset uses `driver_085` / `route_001`,
while the repository's own pre-computed tables use raw comma.ai hashes
(`3f8ae015ce70365f` / `00000000--5c5085329e`). A grep of every CSV in the repo for `driver_\d+`
returns **zero hits**, so there is no explicit join key — a direct join scores **0 / 1,591**.

There *is* an implicit key. `dataset_statistics.py` builds its per-clip scalars with
`closest_row(cs_df, event_mono)` — the single `carState` row at the takeover instant. `event_mono`
ships in every public `meta.json` and falls inside the `logMonoTime` range 100% of the time. That
gives each clip a **5-float fingerprint** (`vEgo`, `aEgo`, `steeringAngleDeg`, `steeringTorque`,
`cruiseState.speed`), matched against `per_clip.csv` within `(car_model, log_kind, log_hz)` at
tolerances 1e-6 → 1e-2.

| outcome | n | % |
|---|---|---|
| **unique match** | **1,043** | **65.6%** |
| ambiguous | 4 | 0.3% |
| no match | 543 | 34.1% |

**Validation:** of the 1,043 unique matches, the matched row's own `clip_id` equals the public
clip's `clip_id` in **1,043 / 1,043 = 100.0%** of cases. A false match would agree by chance about
1-in-15 (clip ids span 0–115). 100% agreement is not coincidence.

That recovers 832 repository routes and 208 drivers, and unlocks the repository's full
**210-column `analysis_master.csv`** — labels (`primary_trigger`, `scenario`, `risk_score`,
`post_maneuver_type`, …) plus the `pre_*` / `post_*` aggregate families.

The 543 unmatched clips remain fully usable for the **viewer** (video + telemetry); they simply
carry no labels. The 34% shortfall is almost certainly snapshot drift — the sample card describes
16,446 clips while the repository tables hold 15,659.
