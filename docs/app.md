# How the app works

## Architecture

```
browser (app/static/index.html)
   │  vanilla JS + uPlot (no build step)
   ├── GET /api/facets    → filter vocabularies
   ├── GET /api/index     → clip list (filterable)
   ├── GET /api/clip/...  → meta + labels + 20 Hz telemetry JSON   [cached in-process]
   ├── GET /api/media/... → takeover.mp4 with HTTP Range support
   └── GET /api/predict   → leave-driver-out prediction for one clip
                                    │
                        FastAPI (app/server.py)
                                    │
              src/adas_to/telemetry.py   → resample 8 CSVs → 20 Hz grid
              src/adas_to/model.py       → model factory + preprocessing
              data/derived/*.parquet     → clip index + model table
```

The backend loads the clip index and model table **once at import**, keeps per-clip telemetry in an
in-process dict cache, and streams video with byte-range requests (so seeking is instant and does
not re-download the file).

## The UI

- **Left rail** — search plus Brand / Trigger / Log-kind / Powertrain filters, and a
  *labelled-only* toggle. Selecting a clip loads everything.
- **Video** (top-left) with a live `t = ±x.xx s` HUD that flags `← TAKEOVER` near zero.
- **Engagement scrubber** under the video: blue = ADAS engaged, amber = manual, red line = the
  takeover, white line = the playhead. Click to seek.
- **Clip card** (top-right) — brand, model, powertrain, log kind/rate, trigger, scenario, post
  manoeuvre, risk/maneuver score, share of the clip with ADAS engaged, plus any data-quality
  warnings for that clip.
- **Prediction card** — class probabilities with the actual value marked ✔.
- **Chart stack** (below) — eight stacked panels sharing one x-axis in seconds relative to the
  takeover, cursor-synced to the video and to each other:

  | # | Panel | Signals |
  |---|---|---|
  | 1 | Speed & cruise set | `vEgo`, `cruiseState.speed`, `vCruise` |
  | 2 | Acceleration | `aEgo`, `aTarget` (planner), commanded accel |
  | 3 | Steering | `steeringAngleDeg`, `steeringTorque` |
  | 4 | Driver inputs | `brakePressed`, `gasPressed`, `steeringPressed` (stepped) |
  | 5 | ADAS engagement | `adas_engaged` (OR), `controlsState.enabled`, `active`, `latActive`, `longActive` |
  | 6 | Lead vehicle radar | `leadOne.dRel`, `leadTwo.dRel`, `leadOne.vRel` |
  | 7 | Safety margins | `TTC`, `THW` (derived) |
  | 8 | Lane perception & planner | `leftProb`, `rightProb`, `curvature`, `desiredCurvature` |

**Interactions:** charts → video (click a chart to seek), video → charts (the playhead drives a
cursor on every panel), `−1s / +1s / ⤓ takeover` buttons, and click-to-seek on the scrubber.

Every panel is captioned with its **native sample rate**, because several channels are
interpolated on the 20 Hz grid (see [Video ↔ telemetry alignment](alignment.md)).

## Derived signals

`TTC` (time-to-collision) and `THW` (time-headway) are computed on the resampled grid, guarded the
way the upstream config specifies:

```
closing = −vRel                 only when vRel < −0.5 m/s
TTC     = dRel / closing        only when dRel > 5 m   and closing > 0, clipped to [0, 100] s
THW     = dRel / vEgo           only when vEgo > 0.5 m/s, positive only
```

Both are masked to `leadOne.status > 0` — i.e. reported only when a lead vehicle actually exists
(median lead presence across the corpus is ~50%).
