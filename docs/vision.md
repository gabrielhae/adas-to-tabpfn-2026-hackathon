# Vision features (YOLOv8n)

The paper's 59.3% early-cue figure comes from a vision-language model. The cheap,
interpretable counterpart is to add **structured YOLOv8n geometry** as extra columns
beside the CAN features and ask whether the sliding forecast improves. It does not —
this is reported as a null result.

## Pipeline (run in order)

| step | script | output |
|---|---|---|
| 1 | `scripts/extract_vision_frames.py` | `data/derived/vision_frames.{parquet,csv}` |
| 2 | `scripts/build_vision_table.py` | `data/derived/vision_table.parquet`, `forecast_table_vision.parquet`, `forecast_schema_vision.json` |
| 3 | `scripts/qa_vision_alignment.py 80` | `results/vision_alignment.json`, `results/vision_qa_grid.png` |
| 4 | `scripts/run_vision_experiments.py --brand` | `results/vision_results.{json,csv}`, `vision_ablation.csv`, `vision_by_lead.csv` |

1. **Per-frame extraction.** Decode each clip's `takeover.mp4` once, sequentially at
   **5 Hz** (100 frames/clip), and run **YOLOv8n** (`conf 0.35`, `imgsz 640`, GPU) in
   batches. 1,041 clips / **104,059 frames** in ~6 min. Per-frame scalars: vehicle /
   corridor-vehicle / person / bicycle / traffic-light / stop-sign counts, lead-box
   geometry (`area_frac`, signed `cx_offset`, `y2_norm`, `conf`), frame brightness and
   frame-to-frame motion. The lead is the largest-area vehicle centred in the ego-lane
   corridor, with the previous frame's lead retained when IoU ≥ 0.1; there is **no
   near-field gate**, so distant leads are included.

2. **Window aggregation.** The same windows as the CAN forecast table — closed
   `[s − 3.5, s]`, `W` read from `forecast_schema.json` — giving 14,574 rows × 58
   `vis_*` columns (18 lead/scene + an 18-column **3×3 occupancy grid** + 7 **traffic-light
   state** features). The grid counts vehicles and sums box area per cell (rows far→near by
   box centre, columns left→right); the light features read the lamp state (red/amber/green)
   from the colour of each detected light box, with a conservative `unknown`. Both are
   zero-inflated and defined on every row. `vis_n_frames` averages 18 per window
   (expected `W × 5 = 17.5`).

3. **Alignment.** `t_rel = frame_idx / CAP_PROP_FPS − 10.0`, i.e. clip-relative video
   time with the takeover at 0 — the same axis as `build_clip_telemetry`'s `t`, using the
   by-construction fact that `video_time_s − clip_start_s == 10.0`. The QA gate samples
   (clip, window) rows, correlates `1/√lead_area_frac` against radar `leadOne.dRel`, and
   writes a contact sheet. It passes: **median Spearman 0.75, 83% of usable clips
   positive**, and the contact sheet shows the box tracking the car ahead (near and
   distant). Vision now finds a lead in **73% of windows** (was 21% with the old
   near-field gate) and misses radar's lead in only **5.9%** of radar-lead windows
   (was 63.6%); `corr(vision lead-present, radar lead-present)` is **0.60** (was 0.35).

4. **Encoding — no imputation of a non-existent lead.** A window with no detected lead is
   a real observation, not missing data. So the lead block uses a **zero-inflated**
   encoding: geometry is `0.0` when absent and `vis_lead_available` (0/1) records whether
   a lead was there at all. This stops `model.encode()` from median-imputing a "typical
   lead" into an empty lane; all 58 `vis_*` columns are defined on every row.

## Ablation (driver-disjoint `GroupKFold(5)`, full table)

CAN-only reproduces `results/forecast_by_lead.csv` / `forecast_results.json` exactly,
which is the check that the arms are comparable. `results/vision_vs_can.csv` holds the
paired CAN vs CAN+vision numbers for every task, model, split and lead.

| task / metric | model | CAN | CAN + vision | CAN + light only |
|---|---|---|---|---|
| `post_maneuver_type` bal-acc | tabpfn | 0.300 | 0.305 | 0.306 |
| `post_maneuver_type` macro-F1 | lightgbm | **0.306** | 0.266 | 0.306 |
| `post_max_abs_steer_torque` R² | lightgbm | 0.514 | 0.533 | **0.546** |
| `post_max_abs_steer_torque` R² | tabpfn | **0.639** | 0.637 | 0.638 |
| `post_max_abs_jerk_mps3` R² | lightgbm | −0.282 | **−0.204** | −0.291 |
| `post_maneuver_type` bal-acc (brand-OOD) | lightgbm | **0.474** | 0.433 | 0.467 |

(Logistic was dropped from these runs; vision-only was measured in an earlier run at
0.275 bal-acc / negative steer R² and was not rescored — it remains far behind CAN.)

## Reading it honestly

- **Two more feature families, same null.** The occupancy grid (3a) and traffic-light
  state (3b) were each added and each rescored against the frozen CAN baseline.
  Isolated light state (`CAN + light only`, 69 features) moves pooled LightGBM
  steer-torque 0.514 → 0.546 and leaves everything else flat; the full block moves
  TabPFN classification 0.300 → 0.305. Neither survives the brand-OOD split
  (LightGBM 0.474 → 0.467 light-only, 0.474 → 0.433 with the full block), and the
  per-lead table stays flat throughout. The small pooled bumps are noise.
- **The detector is not the excuse.** Vision matches radar's lead in 94% of radar-lead
  windows, the alignment QA passes cleanly, and the grid tells the model where traffic
  sits. It still adds nothing net.
- **Why `traffic-light state` didn't capitalise on being non-radar information:** lights
  appear in only ~10% of frames (~8% of windows have a red, ~5% a green), so the feature
  is sparse, and the light-heavy clips are concentrated (stoplight approaches), which
  makes any apparent signal a candidate for scenario/driver confounding — exactly what the
  brand-OOD reversal shows.
- Beating the kinematic baseline needs cues that are both non-radar **and** dense across
  clips; the remaining candidates are lane geometry (3c) and the dense representations
  (segmentation / depth / embeddings, option 4), all of which carry higher cost and higher
  leakage risk.
- The per-lead table (`results/vision_by_lead.csv`) shares `forecast_by_lead.csv`'s schema
  so the CAN baselines can be diffed line-for-line.
