# ADAS-TO Explorer + TabPFN's 2026 Hackathon

My profession has given me the opportunity in the past to touch some aspects of synthetic data
generation for ADAS systems training. The **ADAS-TO** dataset (Wang, Xu, Sun & Zhou, 2026 —
[arXiv:2603.06986](https://arxiv.org/abs/2603.06986)) caught my eye, with the goal of seeing whether
non-tabular data could be used in a tabular foundation-model context.

A summary of results can be found in the presentation [ADAS-TO & TabPFN3.5 Preliminary Investigations.pdf](results/ADAS-TO%20%26%20TabPFN3.5%20Preliminary%20Investigations.pdf).

This repo is an interactive explorer for the dataset plus a leakage-controlled tabular modelling
pipeline built around **TabPFN 3.5**. It answers a question the paper leaves open: ADAS-TO is a
*dataset + empirical-characterisation* paper with **no machine-learning baseline**, and its headline
figure — **"59.3% of critical takeovers have actionable visual cues ≥3 s early"** — comes from a
**vision-language model**. TabPFN's job here is the **kinematic-only baseline**: how much
early-warning signal exists in tabular CAN features alone? The gap to 59.3% is the measured value of
adding vision.

---

## Contents

[Status](#status) · [Quickstart](#quickstart) · [The data](#the-data) · [How the app works](#how-the-app-works) · [Prediction pipeline](#prediction-pipeline) · [Results](#results) · [Vision & flow ablations](#vision--flow-ablations) · [Methodological guardrails](#methodological-guardrails) · [Enabling TabPFN](#enabling-tabpfn) · [Repository layout](#repository-layout) · [Roadmap](#roadmap) · [Citation](#citation)

**Deep dives (`docs/`):** [data](docs/data.md) · [app](docs/app.md) · [video↔telemetry alignment](docs/alignment.md) · [prediction pipeline](docs/pipeline.md) · [feature dictionary](docs/features.md) · [target dictionary](docs/targets.md) · [vision features](docs/vision.md) · [flow features](docs/flow.md)

---

## Status

| Milestone | State |
|---|---|
| **M0** environment | ✅ Python 3.11 venv, torch 2.11.0+**cu128** (RTX 3090, CUDA 12.8), tabpfn 9.0.0 |
| **M1** data + eval harness | ✅ 1,043 labelled clips, 69 features, 3 tasks, driver-disjoint + brand-disjoint CV |
| **M2** explorer backend | ✅ FastAPI, 8 endpoints |
| **M3** clip viewer (video + telemetry) | ✅ 20 Hz resampler, synced uPlot panels, alignment verified |
| **M4** early-warning task | ✅ sliding forecast table + YOLOv8n vision ablation + ego-motion-compensated flow ablation (null results) |
| **M5** polish / figures | ⏳ not started |

TabPFN ≥ v6 requires a one-time licence acceptance tied to a PriorLabs account.
Until `TABPFN_TOKEN` is set, the pipeline automatically runs LightGBM instead — every result below
is reproducible today, and TabPFN drops in with no code change. See [Enabling TabPFN](#enabling-tabpfn).

---

## Quickstart

```bash
# 1. environment (already created)
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install tabpfn pandas numpy pyarrow scikit-learn fastapi \
    "uvicorn[standard]" plotly matplotlib opencv-python lightgbm xgboost catboost \
    huggingface_hub ultralytics

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

Alignment QA: `scripts\qa_alignment.py 40`. Optional ablations (order matters):
vision → [`docs/vision.md`](docs/vision.md), flow → [`docs/flow.md`](docs/flow.md).

> The full 15,659-clip corpus lives at `HenryYHW/ADAS-TO`, which is **`gated: manual`** and
> currently awaiting author approval. Nothing in this project depends on it — everything runs on the
> 10% stratified sample and the plan is to pick up the full set by re-running `build_dataset.py`.

---

## The data

`HenryYHW/ADAS-TO-Sample` — **1,591 clips** (3.21 GiB) across **163 car models · 283 drivers ·
1,238 routes**, 8 CSVs + `meta.json` + `takeover.mp4` per clip. **1,043 clips (65.6%) are
labelled**, recovered by matching a 5-float fingerprint against the repository's own
re-identification tables after the public release re-anonymised the identifiers. The 543
unmatched clips remain fully usable in the viewer, just without labels.

Full breakdown and the re-identification validation: [docs/data.md](docs/data.md).

---

## How the app works

A FastAPI backend (`app/server.py`) plus a single-file vanilla-JS + uPlot UI
(`app/static/index.html`): **video on top, time-synchronised CAN telemetry underneath**, and an
8-panel chart stack sharing one axis (speed, acceleration, steering, driver inputs, ADAS
engagement, lead-vehicle radar, derived `TTC`/`THW`, lane perception). The backend loads the index
once, caches per-clip telemetry in-process, and streams video over HTTP Range so seeking is instant.

Architecture, the full panel table, and the derived-signal guards: [docs/app.md](docs/app.md).
Video↔telemetry sync (a genuinely broken "obvious" approach) is documented in
[docs/alignment.md](docs/alignment.md).

---

## Prediction pipeline

The core design decision is a **clean temporal split**: every `pre_*` feature is aggregated over
`t ∈ [−5, 0]` and every `post_*` target over `t ∈ [0, +5]`. The windows meet at the takeover and do
not overlap, so this is a **genuine forecast**, not leakage:

```
FEATURES  = pre-window  (t ∈ [−5, 0])   ← what the vehicle and driver were doing BEFORE
TARGETS   = post-window (t ∈ [0, +5])   ← what the driver did AFTER
```

`primary_trigger` is rule-derived from `[−3.0, +0.5] s`, which *overlaps* the pre-window — so it,
`scenario`, and every `post_*` column are excluded from features by an assertion in
`build_dataset.py`. Splits are **driver-disjoint** (the paper's stated protocol) and there is also a
persisted 60/20/20 train/val/test split (seed-stable, balanced by clip count):

| split | drivers | rows | share |
|---|---|---|---|
| train | 126 | 625 | 0.599 |
| val | 41 | 209 | 0.200 |
| test | 41 | 209 | 0.200 |

The explorer is scoped to the **test** split; `/api/predict` holds out the target clip's driver,
fits on the rest, and returns four models' probabilities (TabPFN, LightGBM, XGBoost, CatBoost)
alongside the actual value. Full details — preprocessing, the endpoint, TabPFN KV-cache internals —
are in [docs/pipeline.md](docs/pipeline.md).

---

## Results

A summary of results can be found in the presentation
[ADAS-TO & TabPFN3.5 Preliminary Investigations.pdf](results/ADAS-TO%20%26%20TabPFN3.5%20Preliminary%20Investigations.pdf).

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

- **`post_max_abs_steer_torque` R² ≈ 0.66 is the real result.** Pre-takeover kinematics genuinely
  predict post-takeover steering intensity — drivers who were steering hard before continue steering
  hard after. This is the skeleton of a takeover-intensity model.
- **`post_maneuver_type` balanced accuracy 0.45–0.47 vs 0.25 chance** is *weak but non-zero*.
  Accuracy (0.655) is inflated by the majority class; balanced accuracy is the honest metric and it
  says the pre-window is only mildly informative about *what* the driver does next. Note logistic
  beats LightGBM on balanced accuracy while losing on accuracy — a textbook sign the tree model is
  leaning on the majority class with ~200 test samples per fold.
- **`post_max_abs_jerk_mps3` R² ≈ 0 is a clean negative result.** Jerk is essentially not
  predictable from the pre-window. Worth reporting *as* a negative.
- **Brand-OOD beats driver-disjoint.** That is the expected direction and confirms the paper's
  warning: holding out *drivers* removes the strongest nuisance structure, so it is the harder and
  more meaningful test.
- **No log-rate artifact** — qlog and rlog perform comparably, which is a good sign for the
  resampling pipeline.

---

## Vision & flow ablations

Two extra feature families were built and scored against the frozen CAN baseline. Each reproduces
the same shape — **a small pooled gain that reverses out-of-distribution, i.e. a null**.

- **Vision (YOLOv8n geometry)** — lead boxes, a 3×3 occupancy grid, and traffic-light state. The
  detector aligns with radar (median Spearman 0.75) and finds a lead in 73% of windows, but the
  block adds nothing net and the pooled bumps vanish under brand-OOD.
  → [docs/vision.md](docs/vision.md)
- **Flow (ego-motion-compensated optical flow)** — RAFT residual field, IMO components, and onset
  detection. The radar cross-check does not pass (median Spearman −0.026), so the residual is
  dominated by non-modelable ego motion rather than compact agents. → [docs/flow.md](docs/flow.md)

**Verdict:** a third feature family, a third null. The kinematic-only story rests on CAN.

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

1. Accept the licence at <https://ux.priorlabs.ai/account/licenses>.
2. Copy your API key from <https://ux.priorlabs.ai/account>.
3. Set it and re-run — no code changes:

```powershell
$env:TABPFN_TOKEN = "<your key>"
.venv\Scripts\python.exe scripts\run_experiments.py --models tabpfn
```

The app reports readiness at `GET /api/model` and in the sidebar subtitle, and
`/api/predict?model_kind=tabpfn` will use it. Until then `--models auto` falls back to LightGBM.
KV-cache internals and timings: [docs/pipeline.md](docs/pipeline.md).

---

## Repository layout

```
TabPFN-Hackathon2026/
├── app/            FastAPI server + single-file UI (vanilla JS + uPlot)
├── src/adas_to/    config.py · telemetry.py · model.py · splits.py
├── scripts/        build_* · extract_* · qa_* · run_* · build_splits.py · score_test.py
├── docs/           deep dives: data · app · alignment · pipeline · features · targets · vision · flow
├── index/          clip_manifest.csv
├── results/        metrics, figures, ablation tables
├── data/           (local, git-ignored) sample clips + derived parquet
├── _recon/         (local, git-ignored) re-identification scripts
└── ADAS-TO/        (local, git-ignored) upstream clone — code + GIFs only, no data
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

---

ADAS-TO is released under **CC BY-NC 4.0** (non-commercial). Video and CAN logs are from real
public roads; the released identifiers are anonymised.
