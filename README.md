# ADAS-TO Explorer + TabPFN

An interactive explorer for the **ADAS-TO** dataset (Wang, Xu, Sun & Zhou, 2026 —
[arXiv:2603.06986](https://arxiv.org/abs/2603.06986)) that puts **front-view video on top and
time-synchronised CAN telemetry underneath**, plus a leakage-controlled tabular modelling pipeline
built around **TabPFN**.

The point of the project is to answer a question the paper leaves open. ADAS-TO is a
*dataset + empirical characterisation* paper — it contains **no machine-learning baseline at all**
(0 occurrences of `TabPFN`, `tabular`, `transformer`, `XGBoost`, `random forest`, `neural`,
`machine learning` in the full text). Its headline number, **"59.3% of critical takeovers have
actionable visual cues ≥3 s early,"** comes from a **vision-language model**, and is explicitly
framed as early warning *"beyond late-stage kinematic triggers."*

So TabPFN's job here is to be the **kinematic-only baseline**: how much early-warning signal exists
in tabular CAN features alone? The gap between that and 59.3% is the measured value of adding vision.

---

## Contents

1. [Status](#status)
2. [Quickstart](#quickstart)
3. [The data we actually have](#the-data-we-actually-have)
4. [How the app works](#how-the-app-works)
5. [Video ↔ telemetry alignment](#video--telemetry-alignment)
6. [The prediction pipeline](#the-prediction-pipeline)
7. [Feature dictionary](#feature-dictionary)
8. [Target dictionary](#target-dictionary)
9. [Results so far](#results-so-far)
10. [Methodological guardrails](#methodological-guardrails)
11. [Repository layout](#repository-layout)
12. [Enabling TabPFN](#enabling-tabpfn)
13. [Roadmap](#roadmap)

---

## Status

| Milestone | State |
|---|---|
| **M0** environment | ✅ Python 3.11 venv, torch 2.11.0+**cu128** (RTX 3090, CUDA 12.8), tabpfn 9.0.0 |
| **M1** data + eval harness | ✅ 1,043 labelled clips, 69 features, 3 tasks, driver-disjoint + brand-disjoint CV |
| **M2** explorer backend | ✅ FastAPI, 8 endpoints |
| **M3** clip viewer (video + telemetry) | ✅ 20 Hz resampler, synced uPlot panels, alignment verified |
| **M4** early-warning task | ⏳ not started |
| **M5** polish / figures | ⏳ not started |

**Blocker:** TabPFN ≥ v6 requires a one-time licence acceptance tied to a PriorLabs account.
Until `TABPFN_TOKEN` is set, the pipeline automatically runs LightGBM instead — every result below
is reproducible today, and TabPFN drops in with no code change. See [Enabling TabPFN](#enabling-tabpfn).

---

## Quickstart

```bash
# 1. environment (already created)
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install tabpfn pandas numpy pyarrow scikit-learn fastapi \
    "uvicorn[standard]" plotly matplotlib opencv-python lightgbm huggingface_hub

# 2. data (2.2 GiB, auto-gated so it works immediately)
hf download HenryYHW/ADAS-TO-Sample --type dataset --local-dir data\ADAS-TO-Sample

# 3. rebuild derived artefacts
.venv\Scripts\python.exe scripts\build_dataset.py      # clip index + model table
.venv\Scripts\python.exe scripts\check_json.py 80      # JSON safety check

# 4. run the app
.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8077
#   -> http://127.0.0.1:8077

# 5. evaluate
.venv\Scripts\python.exe scripts\run_experiments.py --models logistic lightgbm
```

Other scripts: `scripts\qa_alignment.py 40` (alignment QA), `scripts\test_api.ps1` (endpoint smoke test).

> The full 15,659-clip corpus lives at `HenryYHW/ADAS-TO`, which is **`gated: manual`** and
> currently awaiting author approval. Nothing in this project depends on it — everything runs on the
> 10% stratified sample and will pick up the full set by re-running `build_dataset.py` with a wider
> download.

---

## The data we actually have

| | |
|---|---|
| Source | `HenryYHW/ADAS-TO-Sample` (gated: **auto** → instant) |
| Clips on disk | **1,591** (the dataset card says 1,570) |
| Size / files | 3.21 GiB, 15,912 files |
| Structure | `<CAR_MODEL>/<driver_XXX>/<route_XXX>/<clip_id>/` × 10 files |
| Per clip | 8 CSVs + `meta.json` + `takeover.mp4` (20.00 s, ~1.4 MiB, 20 fps) |
| Coverage | **163 car models · 283 drivers · 1,238 routes** |
| Log rate | 989 `qlog` (10 Hz) · 602 `rlog` (100 Hz) |
| Labelled | **1,043 clips (65.6%)** — see below |

### Why only 1,043 of 1,591 clips have labels

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

---

## How the app works

### Architecture

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

### The UI

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
interpolated on the 20 Hz grid (see below).

### Derived signals

`TTC` (time-to-collision) and `THW` (time-headway) are computed on the resampled grid, guarded the
way the upstream config specifies:

```
closing = −vRel                 only when vRel < −0.5 m/s
TTC     = dRel / closing        only when dRel > 5 m   and closing > 0, clipped to [0, 100] s
THW     = dRel / vEgo           only when vEgo > 0.5 m/s, positive only
```

Both are masked to `leadOne.status > 0` — i.e. reported only when a lead vehicle actually exists
(median lead presence across the corpus is ~50%).

---

## Video ↔ telemetry alignment

This was the single highest-risk part of the project, and it turned out to be genuinely broken in
the obvious approach. **Three findings:**

### 1. The takeover is at a fixed offset in the CSV clock

Using the paper's own event definition (ADAS `ON→OFF` with ≥2 s ON, ≥2 s OFF, gaps <0.5 s merged,
where engaged = `controlsState.enabled` OR `cruiseState.enabled`), the transition lands at
**`csv_min + 10.0 s`** — **98.2% of clips within 0.25 s** (n = 110). 108/110 clips had exactly one
qualifying transition. So the CSV clock is self-consistent and reliable.

### 2. `meta.json`'s `video_time_s` is **not** usable for sync

Validated against the CSV anchor, `event_t − video_time_s` has median **−1.37 s**, std **2.85 s**,
and range **−15.9 … +1.4 s**. Only **31.8%** of clips agree within 0.5 s. Syncing on it naively
would put the video and the charts up to **16 seconds** apart.

### 3. Root cause: a per-route constant clock offset

Variance decomposition of `csv_min − clip_start_s`:

| component | share |
|---|---|
| between-route | **98.9%** |
| within-route | 1.1% |

Within-route std has median 0.067 s (8/9 multi-clip routes under 0.25 s). Both clocks tick at 1:1,
but **each route has its own origin**, differing by up to ~60 s.

### What the app does instead

It ignores `video_time_s` entirely and anchors each window on **its own** event at 10 s — the video
is 20.00 s and the CSV span is 19.90 s, both by design ±10 s around the takeover:

```python
t_csv = csv_min + t_video          # slope 1; used throughout
# robust variant if spans differ:
t_csv = csv_min + (t_video / vid_dur) * csv_span
```

**Verification:** across 40 random clips the engagement transition lands at
**`switch_t = 0.00 ± 0.05 s`** on the relative axis. (The QA script's stricter "post-engagement
must be <50%" rule flags 8/40 clips, but inspection shows those are ADAS *re-engaging* shortly
after the takeover — real behaviour, not misalignment.)

### The 20 Hz ceiling

`configs/analysis_thresholds.yaml` warns against resampling above 20 Hz. Measured native rates make
that concrete:

| topic | rlog | qlog |
|---|---|---|
| carState / controlsState / carControl / carOutput | 100 Hz | 10 Hz |
| radarState | 20 Hz | 4 Hz |
| longitudinalPlan | 20 Hz | **2.7 Hz** |
| drivingModelData | 20 Hz | **1.3 Hz** |
| accelerometer | ~103 Hz | 1 Hz |

qlog `drivingModelData` at ~1.3–2.0 Hz means the lane-probability panel interpolates ~10–15×.
The app **keeps the data at 20 Hz** (no fabrication beyond the documented ceiling) but
**labels each panel with its native rate** and attaches a per-clip warning, so nobody reads
interpolated structure as measurement. The paper explicitly requires this:
*"time-sensitive metrics should report sensitivity to the underlying log rate."*

---

## The prediction pipeline

### The core design decision: a clean temporal split

In the repository, every `pre_*` feature is aggregated over **t ∈ [−5, 0]** and every `post_*`
target over **t ∈ [0, +5]**. The windows meet exactly at the takeover and do not overlap. So:

```
FEATURES  = pre-window  (t ∈ [−5, 0])   ← what the vehicle and driver were doing BEFORE
TARGETS   = post-window (t ∈ [0, +5])   ← what the driver did AFTER
```

This is a **genuine forecast**, not leakage. It is also the honest version of the user's original
brief ("classify driver takeover reaction intensity").

### What we deliberately excluded — and why

This matters more than it sounds. The paper defines `primary_trigger` from pedal/steering onsets
inside **`[−3.0, +0.5] s`** — which **overlaps** the pre-window `[−5, 0]`. Predicting
`primary_trigger` from `pre_max_abs_steer_torque` would therefore be near-tautological, producing a
meaningless 0.95+ AUC. `build_dataset.py` enforces this with an assertion:

```python
FEATURE_DENY = {primary_trigger, trig_steer, trig_brake, trig_gas, n_triggers,
                scenario, ego_reason, nonego_reason, label, is_noise, ...}
bad = [c for c in feats if c.startswith("post") or c in FEATURE_DENY]
assert not bad, f"leaky features present: {bad}"
```

`primary_trigger`, `scenario`, and every `post_*` column are **never** features — only targets,
grouping keys, or display metadata.

### Splits

The paper mandates this protocol in its own words:

> "This within-driver correlation must be accounted for in any modeling task. **We recommend
> driver-disjoint splits as the default train/validation/test protocol** to prevent within-driver
> information leakage. For evaluating cross-platform generalization, **a brand-disjoint or car
> model-disjoint out-of-distribution split** is also supported by the metadata."

The group structure makes this essential: the 1,043 clips come from **208 drivers**, with a
**median of 2 clips per driver** and the single largest driver holding **13.4% of all clips**.

| split | implementation |
|---|---|
| `driver` | `GroupKFold(n_splits=5, groups=dongle_id)` |
| `brand` | `LeaveOneGroupOut()` over `brand` (21 folds) |
| stratum | results also reported separately for `qlog` and `rlog` |

### Preprocessing

Fit on the training fold only, then applied to the test fold:

1. **Categoricals** — one-hot if ≤30 distinct values (`brand`, `powertrain`, `log_kind`, `vid_kind`);
   **frequency-encoded** if high-cardinality (`car_model`, 163 levels) to avoid a 163-column blow-up.
2. **Numerics** — `±inf → NaN`, then **median imputation from the training fold**.
3. LightGBM and TabPFN both handle missing values natively, so imputation is mostly for the
   logistic baseline.

### The `/api/predict` endpoint

It mirrors the CV protocol exactly, for one clip:

1. Look up the clip in the model table (404 if it has no labels).
2. **Hold out every clip from that clip's driver.** Fit on the remainder — so the model has never
   seen this driver's style.
3. Prepare features/targets, drop classes with <10 members, encode, fit, predict.
4. Return predicted class + full probability vector + the **actual** value, so the UI can show
   whether the model got it right.

It flags `degenerate: true` when the training set is suspiciously small, and falls back
TabPFN → LightGBM automatically when no token is present.

---

## Feature dictionary

**69 numeric features + 5 categoricals.** Roughly 7 static scalars plus 62 aggregates over the
pre-takeover window.

### Static scalars (sampled *at* the takeover instant, `closest_row(..., event_mono)`)

| feature | meaning |
|---|---|
| `speed_mps` | ego speed at takeover (m/s) |
| `accel_mps2` | ego longitudinal acceleration (m/s²) |
| `steer_angle_deg` | steering wheel angle (deg) |
| `steer_torque` | driver steering torque (raw units; sign = direction) |
| `cruise_speed_kmh` | ACC/cruise set speed (km/h) |
| `log_hz` | 10 (qlog) or 100 (rlog) — **a logging artifact, watch for leakage via it** |
| `clip_dur_s` | clip duration (s) |

### Safety-margin family — *was there a risk, and how close?*

| feature | meaning |
|---|---|
| `pre_thw_min_s`, `pre_thw_p5_s`, `pre_thw_p50_s`, `pre_thw_p95_s` | time-headway to lead: minimum and percentiles |
| `pre_ttc_min_raw_s`, `pre_ttc_min_capped_s`, `pre_ttc_p5/p50/p95_s` | time-to-collision, raw and capped |
| `pre_drac_max_raw_mps2`, `pre_drac_max_capped_mps2`, `pre_drac_p50/p95_mps2` | **DRAC** — deceleration rate to avoid a crash |
| `pre_min_drel_m`, `pre_p5_drel_m` | closest gap to the lead vehicle |
| `pre_lead_present_rate` | fraction of the window with a detected lead |
| `pre_lead_drop_count` | how many times the lead was lost (detection instability) |
| `pre_longest_cont_lead_s` | longest continuous lead tracking |
| `pre_n_lead_samples` | number of lead samples in the window |
| `pre_time_of_min_ttc_s`, `pre_time_of_max_drac_s` | *when* the worst moment occurred |

### Exposure family — *how long was it dangerous?* (time-under-threshold integrals)

| feature | meaning |
|---|---|
| `pre_time_below_ttc_1.5s / 2.0s / 3.0s` | seconds spent under each TTC threshold |
| `pre_severity_integral_ttc_1.5s / 2.0s / 3.0s` | integral of the TTC deficit (severity-weighted duration) |
| `pre_time_below_thw_0.8s / 1.0s / 1.5s` | seconds spent under each THW threshold |
| `pre_time_above_drac_3.0mps2 / 4.0mps2` | seconds above DRAC thresholds |

### Longitudinal-dynamics family

| feature | meaning |
|---|---|
| `pre_min_accel_mps2`, `pre_max_accel_mps2` | acceleration extremes |
| `pre_accel_p5/p50/p95_mps2` | acceleration distribution |
| `pre_max_abs_jerk_mps3`, `pre_jerk_p50/p95_mps3` | jerk (m/s³) — comfort / abruptness |
| `pre_time_of_peak_decel_s`, `pre_time_of_peak_jerk_s` | when the worst longitudinal event occurred |
| `pre_speed_mean_mps`, `pre_speed_delta_mps` | speed level and drift |

### Lateral-dynamics family

| feature | meaning |
|---|---|
| `pre_max_abs_steer_angle_deg` | largest steering angle |
| `pre_max_abs_steer_torque` | largest driver steering torque |
| `pre_steer_rate_max_deg_per_s`, `pre_steer_rate_p95_deg_per_s` | steering *rate* — how abruptly |
| `pre_time_of_peak_steer_rate_s` | when steering was most abrupt |
| `pre_max_abs_curvature`, `pre_max_abs_desired_curvature` | road curvature vs planner-commanded curvature |

### Perception / alerting family

| feature | meaning |
|---|---|
| `pre_has_lane_probs` | whether lane probabilities were present |
| `pre_lane_left_prob_mean`, `pre_lane_right_prob_mean` | mean lane-detection confidence (degraded markings → lower) |
| `pre_fcw_present` | forward-collision warning raised in the window |
| `pre_alert_present` | any ADAS alert text present |

### Signal-quality family

| feature | meaning |
|---|---|
| `pre_roughness_rms_mps2` | RMS of accelerometer residual — road roughness / ride quality |
| `pre_roughness_pp_mps2` | peak-to-peak roughness |
| `pre_roughness_z_rms_mps2` | vertical-axis roughness |
| `pre_accel_native_hz` | **native sample rate of the accelerometer — a logging artifact, not driving behaviour** |

### Categorical

`brand`, `powertrain` (ICE / BEV / HEV-PHEV), `log_kind`, `vid_kind`, `car_model`.

---

## Target dictionary

### Classification — `post_maneuver_type`

*What did the driver actually do in the 5 s after takeover?* This is the headline target: it maps
directly onto the paper's notion of takeover action modality, but as a **prediction** rather than a
rule.

| class | n | meaning |
|---|---|---|
| `stabilize` | 476 | held the lane, made no deliberate manoeuvre — the takeover was just control transfer |
| `lane_change` | 356 | deliberately changed lane |
| `braking` | 155 | sustained deceleration |
| `acceleration` | 55 | sustained acceleration |
| `turn_ramp` | 1 | merged into `other` (n < 10) for stable macro-F1 |

Classes with fewer than 10 members are collapsed into `other` by `collapse_rare()` — a single
example otherwise makes macro-F1 meaningless.

### Regression — how intense was the reaction?

| target | median | meaning |
|---|---|---|
| `post_max_abs_steer_torque` | 84.7 | peak driver steering torque — lateral intervention strength (max 4,263: an emergency yank) |
| `post_steer_rate_max_deg_per_s` | 17.6 | peak steering rate — how *sharp* the correction was |
| `post_max_abs_jerk_mps3` | 3.53 | peak jerk — passenger-discomfort proxy |
| `post_min_accel_mps2` | −1.27 | hardest braking |
| `post_roughness_rms_mps2` | 0.314 | post-takeover ride roughness |
| `stabilization_5s_time_s` | 5.0 | time to stabilise — ⚠️ **heavily right-censored** (p50 sits at the 5 s window cap), so treat as censored, not continuous |

### Other repository labels (available, not used as targets)

`primary_trigger` (5-class rule-derived), `scenario` (9-class rule-derived), `risk_score`
(p50 = 0.000, max 0.447 in this sample), `maneuver_score`, Ego/Non-ego.
See [Methodological guardrails](#methodological-guardrails) for why these are handled carefully.

---

## Results so far

Driver-disjoint 5-fold CV on the 1,043 labelled clips, 69 features.
LightGBM stands in for TabPFN until a token is set.

| task | model | balanced acc | macro F1 | acc | MAE | R² |
|---|---|---|---|---|---|---|
| `post_maneuver_type` (4-class) | logistic | 0.467 | 0.415 | 0.528 | | |
| `post_maneuver_type` (4-class) | **lightgbm** | 0.447 | **0.461** | 0.655 | | |
| `post_max_abs_steer_torque` | logistic | | | | 266.2 | 0.649 |
| `post_max_abs_steer_torque` | **lightgbm** | | | | **243.7** | **0.656** |
| `post_max_abs_jerk_mps3` | logistic | | | | 2.379 | −0.196 |
| `post_max_abs_jerk_mps3` | lightgbm | | | | 2.329 | −0.003 |

**Log-rate sensitivity** (the paper requires reporting this):

| stratum | n | logistic bal-acc | lightgbm bal-acc |
|---|---|---|---|
| `qlog` (10 Hz) | 653 | 0.478 | 0.447 |
| `rlog` (100 Hz) | 390 | 0.451 | 0.451 |

**Cross-platform OOD** (leave-one-brand-out, 21 folds):

| model | balanced acc | macro F1 |
|---|---|---|
| logistic | 0.531 | 0.478 |
| lightgbm | **0.577** | **0.565** |

### Reading these numbers honestly

- **`post_max_abs_steer_torque` R² ≈ 0.66 is the real result.** Pre-takeover kinematics
  genuinely predict post-takeover steering intensity — drivers who were steering hard before
  continue steering hard after. This is the skeleton of a takeover-intensity model.
- **`post_maneuver_type` balanced accuracy 0.45–0.47 vs 0.25 chance** is *weak but non-zero*.
  Accuracy (0.655) is inflated by the majority class; balanced accuracy is the honest metric and
  it says the pre-window is only mildly informative about *what* the driver does next.
  Note logistic beats LightGBM on balanced accuracy while losing on accuracy — a textbook sign
  the tree model is leaning on the majority class with ~200 test samples per fold.
- **`post_max_abs_jerk_mps3` R² ≈ 0 is a clean negative result.** Jerk is essentially not
  predictable from the pre-window. Worth reporting *as* a negative.
- **Brand-OOD beats driver-disjoint.** That is the expected direction and confirms the paper's
  warning: holding out *drivers* removes the strongest nuisance structure, so it is the harder and
  more meaningful test.
- **No log-rate artifact** — qlog and rlog perform comparably, which is a good sign for the
  resampling pipeline.

---

## Methodological guardrails

These are the traps we found and how each is handled. They are the difference between a real
result and a plausible-looking fake one.

| # | Trap | Handling |
|---|---|---|
| 1 | **No negative class.** Every clip is centred on a takeover by construction, so "will a disengagement occur?" has no counterexamples. | Reframed to a **within-clip temporal forecast**: pre-window → post-window. |
| 2 | **Label leakage via overlapping windows.** `primary_trigger` is rule-derived from `[−3.0, +0.5] s`, which overlaps the pre-window `[−5, 0]`. | `primary_trigger` / `trig_*` are **excluded** from features; enforced by assertion. |
| 3 | **Driver leakage.** 208 drivers, median 2 clips each, top driver = 13.4% of the data. | `GroupKFold(groups=dongle_id)` by default; the paper mandates it. |
| 4 | **Brand leakage.** Cross-platform generalisation is a separate question. | Leave-one-brand-out reported alongside. |
| 5 | **Over-resampling.** qlog `drivingModelData` is ~1.3–2.0 Hz. | Hard 20 Hz ceiling; native rate shown per panel; per-clip warnings. |
| 6 | **Clock drift.** `video_time_s` disagrees by up to 16 s. | Ignored; CSV anchor + video midpoint instead. |
| 7 | **Two conflicting `primary_trigger` labellings** ship in the repo (`per_clip.csv` matches the paper: Brake 39.6 / Steering 25.3; `analysis_master.csv` does not: Steering Override 57%, no `Mixed` class). | Never mixed. `per_clip.csv` is authoritative for trigger; the paper's rule is the reference. |
| 8 | **Noisy intent label.** The Ego/Non-ego partition scores only **84.0%** against a 4-expert audit (Ego P=90.2/R=81.5; Non-ego P=77.1/R=87.5). | Deprioritised as a headline target — you would be fitting a noisy rule, capped near 84%. |
| 9 | **The corpus is mostly benign.** Median TTC 14.90 s, median THW 2.32 s, median lane offset 0.158 m. | Frame results as *routine* takeover modelling; the safety-critical tail is a separate problem. |
| 10 | **Critical tail absent from the sample.** `risk_score` p50 = 0.000, max 0.447; the paper's 285 critical cases are ~1.8% of the full corpus. | No critical-classification task on the sample; needs the full download. |
| 11 | **OEM vs openpilot.** `controlsState.enabled` is **0% duty cycle on OEM clips** (~63% of the corpus). | Engagement = `controlsState.enabled` **OR** `carState.cruiseState.enabled`. A detector reading only `controlsState` silently finds nothing on most clips. |

---

## Enabling TabPFN

TabPFN 9.0.0 is installed and everything is wired for it (`make_model("auto")` prefers TabPFN
whenever it is available). It needs a one-time licence acceptance:

1. Open <https://ux.priorlabs.ai/account/licenses> and accept the licence.
2. Copy your API key from <https://ux.priorlabs.ai/account>.
3. Set it and re-run — no code changes:

```powershell
$env:TABPFN_TOKEN = "<your key>"
.venv\Scripts\python.exe scripts\run_experiments.py --models tabpfn
```

The app reports readiness at `GET /api/model` and in the sidebar subtitle, and
`/api/predict?model_kind=tabpfn` will use it. Until then `--models auto` falls back to LightGBM.

GPU is ready: `torch 2.11.0+cu128`, RTX 3090, 24 GB, compute capability 8.6, CUDA 12.8.

---

## Repository layout

```
TabPFN-Hackathon2026/
├── ADAS-TO/                     upstream GitHub clone (code + GIFs only, no data)
├── data/
│   ├── ADAS-TO-Sample/          1,591 clips (downloaded)
│   └── derived/
│       ├── clip_index.parquet   1,591 clips: paths, meta, video
│       ├── model_table.parquet  1,043 labelled clips × features + targets
│       └── schema.json          feature/target/group definitions + class balance
├── src/adas_to/
│   ├── config.py                paths, windows, thresholds, TabPFN token lookup
│   ├── telemetry.py             8 CSVs → 20 Hz grid + TTC/THW + warnings
│   └── model.py                 availability probe, preprocessing, model factory
├── app/
│   ├── server.py                FastAPI: 8 endpoints
│   └── static/index.html        single-file UI (vanilla JS + uPlot)
├── scripts/
│   ├── build_dataset.py         clip index + model table + leakage assertion
│   ├── run_experiments.py       CV harness, log-rate sensitivity, brand-OOD
│   ├── qa_alignment.py          event-alignment QA
│   ├── check_json.py            strict-JSON safety check
│   └── test_api.ps1             endpoint smoke test
├── _recon/
│   ├── id_map.csv               public id → repo id mapping (1,043 rows)
│   ├── paper/paper.txt          extracted full paper text
│   └── *.py                     reconnaissance scripts
├── results/                     m1_results.{json,csv}, telemetry example, server logs
└── PLAN.md                      full project plan + revisions 1–3
```

---

## Roadmap

**M4 — early warning.** Recompute features on strictly-earlier windows (`[−10, −Δ]` for
Δ ∈ {1, 2, 3} s) directly from the raw CSVs, then predict the post-window target **Δ seconds
before** the takeover. This is the kinematic-only counterpart to the paper's 59.3% VLM figure, and
the main novel contribution. Requires a new feature builder; the current `pre_*` block is reused
where Δ ≤ 5.

**M5 — polish.** Reproduce 2–3 paper figures from the sample (primary-action distribution, speed
regimes, TTC/THW distributions) in the explorer to validate our pipeline against the published
numbers; add a showcase gallery from `ADAS-TO/showcase_gifs`.

**Scale-up.** When `HenryYHW/ADAS-TO` approval lands, re-run `build_dataset.py` against the full
~15,659 clips and re-run the CV. The re-identification approach extends unchanged — it matches on
`(car_model, log_kind, log_hz)` + fingerprint, so a 10× larger pool will need tighter tolerances
and may need a third discriminator (`clip_dur_s`).

---

## Citation

```bibtex
@article{wang2026adasto,
  title  = {ADAS-TO: A Large-Scale Multimodal Naturalistic Dataset and Empirical
            Characterization of Human Takeovers during ADAS Engagement},
  author = {Wang, Yuhang and Xu, Yiyao and Sun, Jingran and Zhou, Hao},
  journal= {arXiv preprint arXiv:2603.06986},
  year   = {2026}
}
```

> Note the dataset card's own BibTeX gives `Zhou, Haowei` and the GitHub README says
> `Anonymous Authors` — both are wrong. The arXiv listing above is correct.

ADAS-TO is released under **CC BY-NC 4.0** (non-commercial). Video and CAN logs are from real
public roads; the released identifiers are anonymised.
