# Video ↔ telemetry alignment

This was the single highest-risk part of the project, and it turned out to be genuinely broken in
the obvious approach. **Three findings:**

## 1. The takeover is at a fixed offset in the CSV clock

Using the paper's own event definition (ADAS `ON→OFF` with ≥2 s ON, ≥2 s OFF, gaps <0.5 s merged,
where engaged = `controlsState.enabled` OR `cruiseState.enabled`), the transition lands at
**`csv_min + 10.0 s`** — **98.2% of clips within 0.25 s** (n = 110). 108/110 clips had exactly one
qualifying transition. So the CSV clock is self-consistent and reliable.

## 2. `meta.json`'s `video_time_s` is **not** usable for sync

Validated against the CSV anchor, `event_t − video_time_s` has median **−1.37 s**, std **2.85 s**,
and range **−15.9 … +1.4 s**. Only **31.8%** of clips agree within 0.5 s. Syncing on it naively
would put the video and the charts up to **16 seconds** apart.

## 3. Root cause: a per-route constant clock offset

Variance decomposition of `csv_min − clip_start_s`:

| component | share |
|---|---|
| between-route | **98.9%** |
| within-route | 1.1% |

Within-route std has median 0.067 s (8/9 multi-clip routes under 0.25 s). Both clocks tick at 1:1,
but **each route has its own origin**, differing by up to ~60 s.

## What the app does instead

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

## The 20 Hz ceiling

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
