# The prediction pipeline

## The core design decision: a clean temporal split

In the repository, every `pre_*` feature is aggregated over **t ∈ [−5, 0]** and every `post_*`
target over **t ∈ [0, +5]**. The windows meet exactly at the takeover and do not overlap. So:

```
FEATURES  = pre-window  (t ∈ [−5, 0])   ← what the vehicle and driver were doing BEFORE
TARGETS   = post-window (t ∈ [0, +5])   ← what the driver did AFTER
```

This is a **genuine forecast**, not leakage. It is also the honest version of the user's original
brief ("classify driver takeover reaction intensity").

## What we deliberately excluded — and why

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

## Splits

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

### Fixed train / validation / test split

`scripts/build_splits.py` (see `src/adas_to/splits.py`) persists one deterministic
**driver-disjoint** three-way split to `data/derived/splits.parquet`: every
`dongle_id` is assigned to exactly one of `train` / `val` / `test`, balanced by
**clip count** (not driver count) because driver sizes run 1–140 clips and the largest
driver is 13.4% of the data. Fractions are 60 / 20 / 20; assignment is seed-stable
(`SPLIT_SEED=0`).

| split | drivers | rows | share |
|---|---|---|---|
| train | 126 | 625 | 0.599 |
| val | 41 | 209 | 0.200 |
| test | 41 | 209 | 0.200 |

Protocol: fit on **train**, tune on **val**, report once on **test**. The explorer
(`app/server.py`) is scoped to the **test** split only — `/api/index`, `/api/facets`,
`/api/stats`, `/api/clip` and `/api/media` all return test clips, and `/api/predict`
fits on train+val before predicting the held-out test clip.

## Preprocessing

Fit on the training fold only, then applied to the test fold:

1. **Categoricals** — one-hot if ≤30 distinct values (`brand`, `powertrain`, `log_kind`, `vid_kind`);
   **frequency-encoded** if high-cardinality (`car_model`, 163 levels) to avoid a 163-column blow-up.
2. **Numerics** — `±inf → NaN`, then **median imputation from the training fold**.
3. LightGBM and TabPFN both handle missing values natively, so imputation is mostly for the
   logistic baseline.

## The `/api/predict` endpoint

It mirrors the CV protocol exactly, for one clip:

1. Look up the clip in the model table (404 if it has no labels).
2. **Hold out every clip from that clip's driver.** Fit on the remainder — so the model has never
   seen this driver's style.
3. Prepare features/targets, drop classes with <10 members, encode, fit, predict.
4. Return predicted class + full probability vector + the **actual** value, so the UI can show
   whether the model got it right.

It flags `degenerate: true` when the training set is suspiciously small, and falls back
TabPFN → LightGBM automatically when no token is present.

## Model comparison

Four models are trained and scored on the same fixed split: **TabPFN, LightGBM,
XGBoost, CatBoost** (`COMPARE_MODELS` in `config.py`). `scripts/score_test.py` fits each
on train+val and scores all 209 test clips, writing `results/test_scores.json`
(served at `/api/score`, shown top-left in the explorer). `/api/predict` returns all
four models' probabilities for the selected clip, which the Prediction card renders as
four side-by-side columns with alphabetically ordered class bars. On this 10% sample the
boosted trees edge out TabPFN on balanced accuracy (~0.49 vs ~0.45) and steer-torque R²
is comparable across all four.

## TabPFN runtime internals

**KV cache is enabled.** `make_model(..., kv_cache=True)` passes
`fit_mode="fit_with_cache"` (TabPFN-3+; guarded by signature inspection, so older
installs fall back silently). The explorer's train+val pool is fixed, so
`/api/predict` fits once and caches the estimator per `(target, model_kind)`, and
TabPFN reuses its training-side KV cache on every later call. Measured on the RTX
3090: first call ~5.7 s (fit + cache build), subsequent calls ~0.7 s.
