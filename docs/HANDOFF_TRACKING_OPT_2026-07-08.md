# Handoff: Tracking Optimization Next Step

Date: 2026-07-08
Repo on Jetson: `/home/jetson/VINGS-Mono`
Windows workspace mirror: `J:/VINGS-Mono`
SSH alias used from Windows: `jetson-codex`

## Goal For The Next Conversation

Continue the VINGS-Mono undergraduate optimization project. The next target is **tracking optimization**. Do not start with INT8. Do not immediately tune frontend image size. First add **tracking internal profiling** so we know which part of tracking is actually expensive.

Suggested opening prompt for the new thread:

```text
Continue tracking optimization in /home/jetson/VINGS-Mono. First read docs/HANDOFF_TRACKING_OPT_2026-07-08.md, then implement internal tracking profiling: split MotionFilter timing into feature encoder, corr/update, context encoder, and append. Run Hotel-200 with the current Jetson defaults and identify the real tracking bottleneck. Do not start with INT8 and do not directly tune frontend image size before collecting profiling evidence.
```

## Current Git State

Latest relevant commits:

```text
b7c68bb chore: default jetson hotel depth scale to 0.75
9da0913 feat: add scalable metric depth inference
6412615 feat: add motion-aware metric depth budget
457bd96 feat: add keyframe-aware metric depth budget
24d43ac chore: add jetson powershell command helper
6ee6fde feat: add scheduled metric depth inference
13663f7 feat: add module runtime profiling
914daf1 feat: add dynamic pixel budget for mapper training
```

Known dirty/untracked state that should not be touched unless explicitly requested:

```text
 m submodules/metric_modules
??   2
?? " \\"
?? "\\"
?? data/
?? docs/CODEX_JETSON_RULES.md
?? docs/HANDOFF_2026-07-07.md
?? reports/
```

This new handoff file is intentionally in `docs/` and may be untracked unless the user asks to commit it.

## Jetson/PowerShell Workflow

Read `docs/CODEX_JETSON_RULES.md` if needed. Windows `J:` access is flaky. Complex commands should run on Jetson using the helper.

Reliable pattern from Windows/Codex:

```powershell
# 1. Write a temporary .sh script under $env:TEMP.
# 2. Run it with the copied helper:
powershell -NoProfile -ExecutionPolicy Bypass -File $env:TEMP/Invoke-JetsonScript-local.ps1 $scriptPath -UseConda
```

The helper was added in `scripts/codex/Invoke-JetsonScript.ps1`, but if `J:` is unavailable, reuse the copied temp helper at `$env:TEMP/Invoke-JetsonScript-local.ps1`.

## What Has Already Been Optimized

### Gaussian Mapping

Implemented earlier:

- `scripts/gaussian/mapping_budget.py`
- `scripts/gaussian/pruning_budget.py`
- `scripts/gaussian/pixel_budget.py`
- runtime budget and profiler support

Main Hotel-200 results:

```text
baseline:
  time around 554s
  final gaussians around 909,806
  PLY around 207.7 MB

mapping + pruning + pixel:
  time around 532s
  final gaussians around 316,329
  PLY around 72.7 MB
```

Mapping optimization greatly reduced Gaussian count and memory/PLY size, but wall-time improvement was limited because the bottleneck moved elsewhere.

### Module-Level Runtime Profiling

Implemented in `scripts/profiling/runtime_profiler.py` with `--profile-runtime` / `VINGS_PROFILE_RUNTIME=1`.

Important baseline with mapping/pruning/pixel but before metric-depth scheduling:

```text
output/07-08-11-02-rtgslam-hote-hotel/runtime_profile_summary.md
frame_total 537.316s
metric_depth 206.252s, 200 calls, 38.4%
tracking     159.945s, 200 calls, 29.8%
mapping      121.435s, 91 calls, 22.6%
loop          38.312s, 5 calls, 7.1%
```

### Metric Depth Scheduling

Implemented:

- `scripts/metric_depth_schedule.py`
- lazy metric depth provider realized inside `MotionFilter.track()` only when a frame is appended as a keyframe
- keyframe budget with `keyframe_min_interval`
- motion-aware optional extension with `high_motion_ratio` and `keyframe_force_interval`

Key result:

```text
keyframe budget, Hotel-200:
frame_total 374.796s
metric_depth 56 calls
metric_depth 67.726s
```

Motion-aware high-motion trigger was tested. Default is conservative `high_motion_ratio=3.0` because `2.0` over-triggered on Hotel and hurt quick RGB metrics. Keep it as a tunable branch, not the main optimization.

### Scalable Metric Depth Inference

Implemented in `9da0913`:

- `metric_depth_scale` in config/CLI
- scales Metric3D internal `crop_size`/`vit_size`, not just the wrapper input RGB
- output depth still returns original image size

Jetson Hotel script default was changed in `b7c68bb`:

```bash
VINGS_METRIC_DEPTH_SCALE:-0.75
```

Generic YAML configs still keep:

```yaml
metric_depth_scale: 1.0
```

This preserves generic algorithm behavior, while Jetson Hotel default enables the edge optimization.

Metric-depth scale Hotel-200 results:

```text
scale=1.0:
  output/07-08-15-11-rtgslam-hote-hotel
  frame_total 379.656s
  tracking     232.884s
  mapping      118.378s
  metric_depth 67.701s, 56 calls, 1208.95ms/call

scale=0.75:
  output/07-08-15-34-rtgslam-hote-hotel
  frame_total 341.451s
  tracking     192.664s
  mapping      119.680s
  metric_depth 25.740s, 56 calls, 459.65ms/call

scale=0.5:
  output/07-08-15-40-rtgslam-hote-hotel
  frame_total 324.890s
  tracking     178.289s
  mapping      117.042s
  metric_depth 11.098s, 56 calls, 198.17ms/call
```

Quick RGB sanity metrics from `reports/jetson_metric_depth_scale_compare_20260708/summary.md`:

```text
scale=1.0:  PSNR 24.380, Simple SSIM 0.9533
scale=0.75: PSNR 23.109, Simple SSIM 0.9192
scale=0.5:  PSNR 23.166, Simple SSIM 0.9057
```

Interpretation: `0.75` is a reasonable default edge trade-off. `0.5` is faster but more aggressive.

## Why Tracking Is Now Next

After `metric_depth_scale=0.75`, metric depth is no longer the dominant module:

```text
scale=0.75 Hotel-200:
frame_total 341.451s
tracking     192.664s
mapping      119.680s
metric_depth 25.740s
loop          18.833s
```

Important caveat: because metric depth is lazily realized inside `MotionFilter.track()`, profiler nesting means `tracking` includes metric-depth time. Rough pure tracking estimate:

```text
tracking - metric_depth ~= 192.664 - 25.740 = 166.924s
```

That is close to the older tracking number `159.945s`; still, tracking is now the biggest remaining target.

Loop timing varies mainly because loop closure calls change with keyframe trajectory and runtime state:

```text
no metric schedule: loop 5 calls, 38.312s
scale=0.75:        loop 4 calls, 18.833s
scale=0.5:         loop 2 calls, 18.506s
```

Do not interpret loop reduction as a direct loop optimization.

## Recommended Next Task

Do **tracking internal profiling first**, before proposing a tracking optimization.

Split `tracking` into at least these stages:

```text
motion_filter_feature_encoder
motion_filter_corr_update
motion_filter_context_encoder
motion_filter_append
frontend_dba_update / frontend call outside MotionFilter, if easy
lazy_metric_depth_inside_tracking, if nested visibility can be improved
```

Likely files:

```text
scripts/frontend/motion_filter.py
scripts/frontend/dbaf_frontend.py
scripts/frontend/dbaf.py or related frontend DBA files
scripts/run.py
scripts/profiling/runtime_profiler.py
```

Start by reading:

```bash
sed -n '1,180p' scripts/frontend/motion_filter.py
sed -n '1,260p' scripts/frontend/dbaf_frontend.py
grep -R "with self.profiler.time(\"tracking\"\|MotionFilter\|track(" -n scripts tests
```

## Suggested Implementation Shape

Preferred minimal design:

1. Pass the existing runtime profiler into the frontend or `MotionFilter` without changing default behavior.
2. Add optional fine-grained `profiler.time(...)` blocks around major tracking stages.
3. Keep all profiling disabled unless `VINGS_PROFILE_RUNTIME=1` / `--profile-runtime` is on.
4. Run Hotel-200 with current default Jetson settings and inspect which tracking substage dominates.
5. Only after that choose the actual optimization.

Avoid jumping straight to INT8. After metric depth scale, INT8 has smaller upside because metric depth is only about `25.7s` at scale `0.75`. Tracking/mapping dominate more.

## Possible Tracking Optimizations After Profiling

Pick based on evidence.

### A. Frontend Image Size Ablation

If feature extraction and correlation/update dominate, try reducing frontend image size:

```bash
VINGS_FRONTEND_IMAGE_SIZE=224,384
```

Compare against current default:

```bash
VINGS_FRONTEND_IMAGE_SIZE=256,448
```

This is simple and likely effective, but quality/tracking stability must be checked.

### B. Keyframe Threshold / MotionFilter Budget

If context encoder and keyframe append dominate, tune or schedule the motion threshold/keyframe density. Need to inspect where `MotionFilter(thresh=...)` is constructed and how `frontend.thresh` is configured.

Risk: too aggressive threshold can reduce keyframes and hurt mapping/loop stability.

### C. Lightweight Pre-Gate Before CorrBlock

If `CorrBlock + update` dominates every frame, consider a cheap pre-gate before full correlation/update. This is more research-like but riskier because it may change tracking behavior.

Do not implement this first without profiling evidence.

## Useful Hotel-200 Run Environment

Use current default Jetson script and override dataset:

```bash
cd /home/jetson/VINGS-Mono
export VINGS_DATASET_ROOT=/home/jetson/VINGS-Mono/data/hotel_compare_200
export VINGS_MAPPING_BUDGET=1
export VINGS_PRUNING_BUDGET=1
export VINGS_PIXEL_BUDGET=1
export VINGS_METRIC_DEPTH_SCHEDULE=1
export VINGS_METRIC_DEPTH_MODE=keyframe
export VINGS_METRIC_DEPTH_KEYFRAME_MIN_INTERVAL=3
export VINGS_METRIC_DEPTH_KEYFRAME_FORCE_INTERVAL=10
export VINGS_METRIC_DEPTH_HIGH_MOTION_RATIO=3.0
export VINGS_METRIC_DEPTH_SCALE=0.75
export VINGS_PROFILE_RUNTIME=1
export VINGS_ADAPTIVE_RUNTIME=1
export VINGS_TRAINING_ITERS=30
export VINGS_FRONTEND_IMAGE_SIZE=256,448
bash scripts/run_jetson_hotel.sh
```

Expected current default-ish result is close to:

```text
frame_total about 341s
metric_depth about 25.7s
tracking about 192.7s including nested metric depth
mapping about 119.7s
```

## Quality Check Notes

Existing quick quality reports use `rgbdnua` crops:

- top-left cell: GT RGB
- bottom-left cell: rendered RGB

This is only a quick sanity check, not full SLAM evaluation. For tracking optimization, also visually inspect whether output trajectory/map looks stable, because PSNR alone may miss tracking failures.

## 2026-07-08 Adaptive DBA Update Result

Implemented:

- `scripts/profiling/compare_tracking_runs.py` compares runtime totals, keyframe count, and quick `rgbdnua` crop PSNR/SSIM across runs.
- `MotionFilter` records `video.last_motion_score`, `video.last_motion_threshold`, and `video.last_motion_added_keyframe`.
- `DBAFusionFrontend` supports motion-aware adaptive update iterations:
  - fixed low values: `frontend.iters1` / `frontend.iters2`
  - high-motion values: `frontend.iters1_high` / `frontend.iters2_high`
  - trigger: `last_motion_score >= last_motion_threshold * iters_high_motion_ratio`
  - optional force refresh: `iters_force_interval`
- CLI / Hotel launcher flags:
  - `--frontend-adaptive-iters`
  - `--frontend-iters1-high`
  - `--frontend-iters2-high`
  - `--frontend-iters-high-motion-ratio`
  - `--frontend-iters-force-interval`
  - `VINGS_FRONTEND_ADAPTIVE_ITERS`, `VINGS_FRONTEND_ITERS1_HIGH`, `VINGS_FRONTEND_ITERS2_HIGH`, `VINGS_FRONTEND_ITERS_HIGH_MOTION_RATIO`, `VINGS_FRONTEND_ITERS_FORCE_INTERVAL`

Local verification:

```text
python -m pytest --import-mode=importlib tests/test_tracking_compare.py tests/test_dbaf_frontend_iteration_config.py tests/test_dbaf_frontend_runtime_profiling.py tests/test_dbaf_runtime_profiling.py tests/test_motion_filter_lazy_depth.py tests/test_runtime_profiler.py tests/test_jetson_runtime_overrides.py tests/test_run_jetson_hotel_script.py -q
33 passed, 5 warnings
```

Hotel-200 comparison:

```bash
python scripts/profiling/compare_tracking_runs.py \
  --run fixed_3_1=output/07-08-17-02-rtgslam-hote-hotel \
  --run adaptive_2_1_high3_force5=output/07-08-18-39-rtgslam-hote-hotel \
  --output-dir reports/tracking_adaptive_2_1_high3_force5_vs_fixed_3_1_20260708 \
  --max-contact-frames 12
```

Runtime:

| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s |
|---|---:|---:|---:|---:|---:|---:|
| fixed_3_1 | 298.390 | 151.349 | 56 | 117.233 | 82.993 | 12.282 |
| adaptive_2_1_high3_force5 | 280.702 | 133.351 | 56 | 99.080 | 64.106 | 12.247 |

Quick RGB sanity:

| Run | Common Frames | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|---:|
| fixed_3_1 | 23 | 18.635 | 12.351 | 0.9236 | 0.8512 |
| adaptive_2_1_high3_force5 | 23 | 19.269 | 14.399 | 0.9255 | 0.7877 |

Takeaway: adaptive low `2/1`, high `3/1`, force interval `5` keeps the same keyframe count and cuts tracking by about `18.0s` vs fixed `3/1`. The lower min SSIM still needs visual/trajectory sanity before making it the default.

## 2026-07-08 Adaptive Force Interval Follow-Up

Added detailed comparison outputs:

- `quality_by_frame.csv`: per-common-frame PSNR/SSIM and delta vs the first run.
- `pose_deltas.csv`: per-common-pose translation and rotation deltas vs the first run.

Validation:

```text
python -m pytest --import-mode=importlib tests/test_tracking_compare.py tests/test_dbaf_frontend_iteration_config.py tests/test_dbaf_frontend_runtime_profiling.py tests/test_dbaf_runtime_profiling.py tests/test_motion_filter_lazy_depth.py tests/test_runtime_profiler.py tests/test_jetson_runtime_overrides.py tests/test_run_jetson_hotel_script.py -q
33 passed, 5 warnings
```

Force interval `3` run:

```text
output/07-08-19-03-rtgslam-hote-hotel
```

Three-way report:

```text
reports/tracking_adaptive_force3_force5_vs_fixed_20260708
```

Runtime:

| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s |
|---|---:|---:|---:|---:|---:|---:|
| fixed_3_1 | 298.390 | 151.349 | 56 | 117.233 | 82.993 | 12.282 |
| adaptive_force5 | 280.702 | 133.351 | 56 | 99.080 | 64.106 | 12.247 |
| adaptive_force3 | 286.464 | 137.532 | 56 | 103.371 | 68.151 | 12.462 |

Quick RGB sanity on the 16 frames common to all three:

| Run | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|
| fixed_3_1 | 18.965 | 14.933 | 0.9274 | 0.8512 |
| adaptive_force5 | 18.770 | 14.399 | 0.9217 | 0.7877 |
| adaptive_force3 | 18.980 | 14.036 | 0.9225 | 0.8240 |

Pose sanity vs fixed `3/1`:

- `adaptive_force5`: previous common-pose max translation about `0.0293`, max rotation about `0.65 deg`.
- `adaptive_force3`: max translation `0.0196`, max rotation `0.3481 deg` on the three-way common poses.

Recommendation: prefer force interval `3` over `5` as the conservative adaptive candidate. It keeps about `13.8s` tracking speedup vs fixed `3/1`, with smaller pose drift and a less severe min-SSIM drop than force interval `5`.

## 2026-07-08 Adaptive Force Interval 2 Check

Force interval `2` run:

```text
output/07-08-19-18-rtgslam-hote-hotel
```

Four-way report:

```text
reports/tracking_adaptive_force2_force3_force5_vs_fixed_20260708
```

Pairwise force2 report:

```text
reports/tracking_adaptive_force2_vs_fixed_20260708
```

Pairwise force2 vs fixed on 23 common frames:

| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_3_1 | 298.390 | 151.349 | 56 | 117.233 | 82.993 | 12.282 | 0.9307 | 0.8512 |
| adaptive_force2 | 290.183 | 141.752 | 56 | 107.736 | 72.500 | 12.519 | 0.9345 | 0.8477 |

Pairwise force3 vs fixed on 26 common frames:

| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_3_1 | 298.390 | 151.349 | 56 | 117.233 | 82.993 | 12.282 | 0.9342 | 0.8512 |
| adaptive_force3 | 286.464 | 137.532 | 56 | 103.371 | 68.151 | 12.462 | 0.9305 | 0.8240 |

Problem-frame visual sheet:

```text
reports/tracking_adaptive_force2_force3_force5_vs_fixed_20260708/problem_frame_contact_sheet.png
```

Visual check: no obvious tracking collapse in the selected low-score frames; differences look like local render/color/depth differences rather than a bad trajectory break.

Recommendation update:

- Use `adaptive_force3` (`low=2/1`, `high=3/1`, `force_interval=3`) as the speed/quality candidate: about `13.8s` tracking speedup vs fixed `3/1`.
- Use `adaptive_force2` as the quality-conservative candidate: about `9.6s` tracking speedup, mean SSIM slightly above fixed on pairwise common frames, min SSIM nearly equal to fixed.
- Do not promote either to default until trajectory/ATE-style evaluation is available or at least a broader visual pass across more sequences is done.

## 2026-07-08 Full Hotel Dataset Check

Full Hotel input:

```text
data/hotel/nosky_color: 405 frames
```

The full dataset was run after the Hotel-200 adaptive DBA results, using the same Jetson edge defaults:

```text
VINGS_MAPPING_BUDGET=1
VINGS_PRUNING_BUDGET=1
VINGS_PIXEL_BUDGET=1
VINGS_METRIC_DEPTH_SCHEDULE=1
VINGS_METRIC_DEPTH_MODE=keyframe
VINGS_METRIC_DEPTH_KEYFRAME_MIN_INTERVAL=3
VINGS_METRIC_DEPTH_KEYFRAME_FORCE_INTERVAL=10
VINGS_METRIC_DEPTH_HIGH_MOTION_RATIO=3.0
VINGS_METRIC_DEPTH_SCALE=0.75
VINGS_PROFILE_RUNTIME=1
VINGS_ADAPTIVE_RUNTIME=1
VINGS_TRAINING_ITERS=30
VINGS_FRONTEND_IMAGE_SIZE=256,448
```

Runs:

```text
fixed_3_1:        output/07-08-19-29-rtgslam-hote-hotel
adaptive_force3:  output/07-08-19-43-rtgslam-hote-hotel
adaptive_force2:  output/07-08-19-58-rtgslam-hote-hotel
```

Reports:

```text
reports/tracking_full_hotel_force3_vs_fixed_20260708
reports/tracking_full_hotel_force2_vs_fixed_20260708
reports/tracking_full_hotel_force2_force3_vs_fixed_20260708
```

Runtime:

| Run | Frame Total s | Tracking s | Keyframes | DBA Update s | Non-KF Update s | KF Update s |
|---|---:|---:|---:|---:|---:|---:|
| fixed_3_1 | 771.275 | 361.861 | 136 | 292.510 | 209.528 | 32.169 |
| adaptive_force3 | 787.070 | 331.308 | 138 | 262.267 | 177.733 | 32.706 |
| adaptive_force2 | 770.459 | 338.358 | 138 | 268.901 | 185.405 | 32.505 |

Pairwise force3 vs fixed on 56 common `rgbdnua` frames:

| Run | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|
| fixed_3_1 | 16.537 | 9.170 | 0.8705 | 0.4101 |
| adaptive_force3 | 17.529 | 9.130 | 0.8792 | 0.3207 |

Pairwise force2 vs fixed on 58 common `rgbdnua` frames:

| Run | Mean PSNR | Min PSNR | Mean SSIM | Min SSIM |
|---|---:|---:|---:|---:|
| fixed_3_1 | 16.837 | 9.170 | 0.8793 | 0.4101 |
| adaptive_force2 | 16.308 | 8.494 | 0.8587 | 0.3608 |

Trajectory/pose sanity vs fixed:

- `adaptive_force3` shows late divergence around frames `315-351`, with max pose deltas about `1.32m` and `13.2 deg`.
- `adaptive_force2` also diverges late, with max pose delta `0.9209m` at frame `382` and max rotation delta `11.5841 deg` at frame `343`.
- Worst force2 RGB sanity frame was frame `321`: SSIM `0.3608` vs fixed `0.9558`, delta `-0.5950`.

Full Hotel conclusion:

- `adaptive_force3` reduces tracking by about `30.6s`, but total frame time is worse (`787.070s` vs `771.275s`) and late-pose divergence is too large.
- `adaptive_force2` reduces tracking by about `23.5s`, but total frame time is basically flat (`770.459s` vs `771.275s`) and still has large late-pose divergence.
- Keep `VINGS_FRONTEND_ADAPTIVE_ITERS=0` by default. The current adaptive DBA iteration scheduler is useful as an experimental branch, but is not safe to promote based on full Hotel.

Recommended next optimization direction:

- Do not spend more time tuning only `iters_force_interval` in the current policy.
- Investigate why frames around `300+` are sensitive before trying another adaptive scheduler.
- If continuing DBA iteration optimization, add a stability guard instead of a pure periodic/motion trigger. Candidate signals: recent pose delta, keyframe age, loop/mapping pressure, or temporarily falling back to fixed `3/1` after long runs or high residual.
- Otherwise move to the next profiling-backed target, such as frontend image-size ablation or a safer non-keyframe DBA skip/early-exit rule, with full-dataset validation from the start.
