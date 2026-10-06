# Target dictionary

## Classification — `post_maneuver_type`

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

## Regression — how intense was the reaction?

| target | median | meaning |
|---|---|---|
| `post_max_abs_steer_torque` | 84.7 | peak driver steering torque — lateral intervention strength (max 4,263: an emergency yank) |
| `post_steer_rate_max_deg_per_s` | 17.6 | peak steering rate — how *sharp* the correction was |
| `post_max_abs_jerk_mps3` | 3.53 | peak jerk — passenger-discomfort proxy |
| `post_min_accel_mps2` | −1.27 | hardest braking |
| `post_roughness_rms_mps2` | 0.314 | post-takeover ride roughness |
| `stabilization_5s_time_s` | 5.0 | time to stabilise — ⚠️ **heavily right-censored** (p50 sits at the 5 s window cap), so treat as censored, not continuous |

## Other repository labels (available, not used as targets)

`primary_trigger` (5-class rule-derived), `scenario` (9-class rule-derived), `risk_score`
(p50 = 0.000, max 0.447 in this sample), `maneuver_score`, Ego/Non-ego.
See [Methodological guardrails](../README.md#methodological-guardrails) for why these are handled
carefully.
