# Feature dictionary

**69 numeric features + 5 categoricals.** Roughly 7 static scalars plus 62 aggregates over the
pre-takeover window.

## Static scalars (sampled *at* the takeover instant, `closest_row(..., event_mono)`)

| feature | meaning |
|---|---|
| `speed_mps` | ego speed at takeover (m/s) |
| `accel_mps2` | ego longitudinal acceleration (m/s²) |
| `steer_angle_deg` | steering wheel angle (deg) |
| `steer_torque` | driver steering torque (raw units; sign = direction) |
| `cruise_speed_kmh` | ACC/cruise set speed (km/h) |
| `log_hz` | 10 (qlog) or 100 (rlog) — **a logging artifact, watch for leakage via it** |
| `clip_dur_s` | clip duration (s) |

## Safety-margin family — *was there a risk, and how close?*

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

## Exposure family — *how long was it dangerous?* (time-under-threshold integrals)

| feature | meaning |
|---|---|
| `pre_time_below_ttc_1.5s / 2.0s / 3.0s` | seconds spent under each TTC threshold |
| `pre_severity_integral_ttc_1.5s / 2.0s / 3.0s` | integral of the TTC deficit (severity-weighted duration) |
| `pre_time_below_thw_0.8s / 1.0s / 1.5s` | seconds spent under each THW threshold |
| `pre_time_above_drac_3.0mps2 / 4.0mps2` | seconds above DRAC thresholds |

## Longitudinal-dynamics family

| feature | meaning |
|---|---|
| `pre_min_accel_mps2`, `pre_max_accel_mps2` | acceleration extremes |
| `pre_accel_p5/p50/p95_mps2` | acceleration distribution |
| `pre_max_abs_jerk_mps3`, `pre_jerk_p50/p95_mps3` | jerk (m/s³) — comfort / abruptness |
| `pre_time_of_peak_decel_s`, `pre_time_of_peak_jerk_s` | when the worst longitudinal event occurred |
| `pre_speed_mean_mps`, `pre_speed_delta_mps` | speed level and drift |

## Lateral-dynamics family

| feature | meaning |
|---|---|
| `pre_max_abs_steer_angle_deg` | largest steering angle |
| `pre_max_abs_steer_torque` | largest driver steering torque |
| `pre_steer_rate_max_deg_per_s`, `pre_steer_rate_p95_deg_per_s` | steering *rate* — how abruptly |
| `pre_time_of_peak_steer_rate_s` | when steering was most abrupt |
| `pre_max_abs_curvature`, `pre_max_abs_desired_curvature` | road curvature vs planner-commanded curvature |

## Perception / alerting family

| feature | meaning |
|---|---|
| `pre_has_lane_probs` | whether lane probabilities were present |
| `pre_lane_left_prob_mean`, `pre_lane_right_prob_mean` | mean lane-detection confidence (degraded markings → lower) |
| `pre_fcw_present` | forward-collision warning raised in the window |
| `pre_alert_present` | any ADAS alert text present |

## Signal-quality family

| feature | meaning |
|---|---|
| `pre_roughness_rms_mps2` | RMS of accelerometer residual — road roughness / ride quality |
| `pre_roughness_pp_mps2` | peak-to-peak roughness |
| `pre_roughness_z_rms_mps2` | vertical-axis roughness |
| `pre_accel_native_hz` | **native sample rate of the accelerometer — a logging artifact, not driving behaviour** |

## Categorical

`brand`, `powertrain` (ICE / BEV / HEV-PHEV), `log_kind`, `vid_kind`, `car_model`.
