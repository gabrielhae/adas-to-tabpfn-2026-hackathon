# Flow features (ego-motion-compensated optical flow)

Box geometry describes objects; it cannot express **"something started moving."** This family
estimates dense optical flow, fits and subtracts the global (ego) motion, and turns the
**residual** field — what moves relative to the world — into IMO and onset features. It is
scored **standalone** against the frozen CAN baseline; it is deliberately *not* stacked onto the
`vis_*` block, which already showed no gain.

## Pipeline (run in order)

| step | script | output |
|---|---|---|
| 1 | `scripts/extract_flow_frames.py` | `data/derived/flow_frames.{parquet,csv}` |
| 2 | `scripts/build_flow_table.py --method raft` | `data/derived/flow_table.parquet`, `forecast_table_flow.parquet`, `forecast_schema_flow.json` |
| 3 | `scripts/qa_flow_alignment.py 80 --method raft` | `results/flow_alignment.json`, `results/flow_qa_grid.png` |
| 4 | `scripts/run_flow_experiments.py --models lightgbm tabpfn --feature-sets can flow both --brand` | `results/flow_results.{json,csv}`, `flow_ablation.csv`, `flow_by_lead.csv`, `flow_by_lead_wide.csv`, `flow_vs_can.csv` |

1. **Per-frame extraction.** Decode each clip's `takeover.mp4` once at **10 Hz** and keep
   `t_rel ∈ [−10, 0]` (no feature or target uses post-takeover frames, so decoding stops at
   `t = 0`). Dense flow between consecutive grey frames uses **RAFT-Large** (torchvision), run at
   256×160 and upscaled: **1,041 clips / 105,141 frames**, flow valid on **99.0%**, in ~28 min on
   the RTX 3090. The alternative `--method dis` is ~3× cheaper but its residual is ~3.3 px median
   vs **~1 px with RAFT** at 526×330 and it fails the visual gate, so RAFT is the configured
   default (`FLOW_METHOD`).

2. **Global (ego-motion) fit.** `cv2.estimateAffinePartial2D` (RANSAC similarity, 3 px) on an
   every-8th-pixel grid, masking the ego hood (bottom 15%) and a 4-px border; median inlier
   fraction **0.857**. The residual is `observed − global-model` flow. Divergence
   (`∂u/∂x + ∂v/∂y`) is the looming signal; IMO components are
   `|residual| > max(1.5 px, 3 × median|residual|)` with area ≥ 0.1% of the frame.

3. **Onsets.** Activity = corridor residual magnitude; a **rising edge** is flagged when it
   clears `max(1.8 × rolling-median over 2 s, 1.0 px)` — a documented heuristic tuned on the
   smoke run. Onsets fire on **6.1%** of frames (6,372 global / 6,473 corridor).

4. **Window aggregation** onto the same closed windows as CAN (`[s − 3.5, s]`, `W` from
   `forecast_schema.json`), giving 14,574 rows × **22 `flow_*` columns** in four groups
   (`flow_res` 10, `flow_imo` 4, `flow_onset` 4, `flow_global` 2, plus `flow_available` /
   `flow_n_samples`; mean 36 samples per window). Same **zero-inflated** encoding as `vis_*`:
   a window with no flow-valid frame or no IMO is a real observation — features fall back to
   `0.0` sentinels and `flow_available` / `flow_imo_available` record presence, so
   `model.encode()` never median-imputes a non-existent event. The merge onto `forecast_table`
   is asserted `one_to_one` on the five keys.

## QA gate — the visual check is marginal and the radar cross-check does not pass

The blocking gate **fails on its radar leg**: across 80 sampled (clip, window) rows the per-clip
Spearman between corridor residual divergence and radar `leadOne.vRel` is a median of **−0.026**
with **55% negative** (the expected sign is negative, i.e. closing traffic → expanding residual).
A wider sweep of four residual features over 60 clips found no feature with a meaningful
relationship (all medians ≈ 0, ~50% positive). The **jitter check passes**: onsets are not driven
by camera rotation (onset rate 0.098 in the top-decile `|global rot|` frames vs 0.057 elsewhere).

The `results/flow_qa_grid.png` contact sheet shows the residual arrows concentrating on actual
vehicles (the corridor car, oncoming traffic on the left) **and** on the near road plane, whose
depth-dependent expansion a similarity model cannot remove. The detected IMO components are
therefore a mix of real agents and broad ego-residual regions: the largest component has a median
area of **20,168 px ≈ 11.6%** of a 526×330 frame, far larger than a vehicle. At this resolution a
distant pedestrian is a few pixels, below the 0.1% (≈13×13 px) minimum. The gate is reported as
**not passed**; the ablation below is run anyway and read with that caveat.

## Ablation (driver-disjoint `GroupKFold(5)`, full table)

CAN-only reproduces `results/forecast_results.json` fold-for-fold (checked), so the arms are
comparable. `results/flow_vs_can.csv` holds the paired CAN vs CAN+flow numbers for every task,
model, split and lead.

| task / metric | model | CAN | flow only | CAN + flow |
|---|---|---|---|---|
| `post_maneuver_type` bal-acc | lightgbm | **0.335** | 0.290 | 0.334 |
| `post_maneuver_type` macro-F1 | lightgbm | 0.306 | 0.263 | **0.318** |
| `post_maneuver_type` bal-acc | tabpfn | **0.300** | 0.299 | 0.291 |
| `post_max_abs_steer_torque` R² | lightgbm | 0.514 | −0.409 | **0.547** |
| `post_max_abs_steer_torque` R² | tabpfn | 0.639 | −0.275 | **0.647** |
| `post_max_abs_jerk_mps3` R² | lightgbm | −0.282 | −0.230 | **−0.242** |
| `post_max_abs_jerk_mps3` R² | tabpfn | 0.000 | −0.047 | **0.010** |
| `post_maneuver_type` bal-acc (brand-OOD) | lightgbm | **0.474** | 0.350 | 0.460 |

## Reading it honestly

- **Same shape as the vision block: a pooled bump that reverses out-of-distribution.** Adding flow
  moves pooled LightGBM steer-torque R² 0.514 → 0.547 (+0.033, MAE −10.9), macro-F1 0.306 → 0.318
  and jerk R² −0.282 → −0.242; TabPFN is essentially flat (0.639 → 0.647). But under the
  brand-OOD split the CAN+flow arm *drops* (0.474 → 0.460) and flow-only collapses to 0.350, so the
  pooled gain does not survive the non-radar / cross-platform test.
- **Flow-only is far behind CAN**, with a *negative* steer-torque R². The block is not a standalone
  kinematic signal; whatever small positive it carries is redundant with CAN.
- **The radar cross-check did not validate the residual field.** The one external ground truth we
  have — the lead vehicle's relative speed — does not correlate with residual divergence, which is
  the honest reason to distrust the pooled bump rather than celebrate it. Combined with the IMO
  size distribution (largest blob ~12% of the frame), the residual is dominated by non-modelable
  ego motion, not by compact agents.
- **Confounding is expected and guarded.** Residual magnitude tracks ego speed, road type and
  route scenery — all already in CAN — which is exactly what the brand-OOD reversal exposes.
- The per-lead breakdowns (`flow_by_lead.csv` in the `forecast_by_lead.csv` schema,
  `flow_by_lead_wide.csv` in the `vision_by_lead.csv` style) stay flat across all 14 leads.

**Verdict:** a third feature family, a third null. Cut-in/pedestrian onsets remain a plausible
non-radar cue, but at 526×330 and 10 Hz the optical-flow residual is not clean enough to isolate
them, and the radar cross-check confirms it. Raising the ceiling needs higher-resolution frames
(or a detector), not a better global-motion model.
