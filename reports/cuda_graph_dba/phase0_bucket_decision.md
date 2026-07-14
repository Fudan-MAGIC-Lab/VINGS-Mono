# Bucketed DBA CUDA Graph Phase 0 Decision

## Environment

- Device: NVIDIA Jetson Orin
- Python: 3.8.20
- PyTorch: `2.1.0a0+41361538.nv23.06`
- CUDA: 11.4 (`torch.cuda.is_available() == True`)
- Original profiling commit: `eaa30d998b68893aa5eb5a3f181fd3014a054158`
- Hotel retry commit: `c813c8f` (Hotel CPU save buffer corrected from 64 to 512)
- Evidence directory: `reports/cuda_graph_dba/phase0_20260714`

## Input runs and exact commands

The guarded driver `reports/cuda_graph_dba/run_phase0_dba_profiles.sh` ran:

1. SmallCity-50 eager control, `344x616`, Torch update backend, no runtime profiler: exit 0, 127 s.
2. SmallCity-50 instrumented, same runtime configuration plus `--profile-runtime`: exit 0, 126 s.
3. SmallCity-200 instrumented: exit 0, 327 s.
4. Hotel `344x616` instrumented: the first attempt exposed an undersized experimental `--frontend-save-buffer 64` and stopped at frame 173 with an index-64 bounds error. A regression-tested driver fix changed only Hotel to `--frontend-save-buffer 512`; the protected retry completed 405/405 frames with exit 0 in 1432 s.

The full shared options and output guards are preserved in the driver. The Hotel retry command summary, Git state, run log, status, and `tegrastats` log are under `phase0_20260714/hotel344_retry1`.

## Observed signature distribution

All 1696 recorded DBA calls match the Phase 0 support contract: VO mode, Torch backend, FP16 update state, frontend `344x616`, and feature grid `43x77`. There are no unsupported calls.

| Run | Calls | Unique shapes | Active edges | BA edges | Source poses | Pose window |
|---|---:|---:|---:|---:|---:|---:|
| SmallCity-50 | 140 | 10 | 25-48 | 32-72 | 5-11 | 4-11 |
| SmallCity-200 | 376 | 10 | 25-48 | 32-72 | 5-11 | 4-11 |
| Hotel retry | 1180 | 16 | 24-48 | 31-92 | 5-12 | 4-14 |

The dominant exact signatures are `(48,66,11,10)` with 832 calls, `(48,72,12,11)` with 445 calls, and `(48,72,11,11)` with 176 calls. Together they account for 85.7% of all calls.

## Proposed buckets and call coverage

With the approved limits (90% target, at most 6 buckets, 35% maximum padding, and 1024 MiB maximum workspace), the planner can admit only one bucket:

| Bucket | Capacity `(active, BA, source, window)` | Estimated workspace | Covered calls |
|---|---|---:|---:|
| `e31_ba38_s6_p5_float16_inactive_up` | `(31,38,6,5)` | 986.69 MiB | 44 |

Official planned coverage is **44/1696 = 2.59%**, below the required 90%.

A budget sensitivity check is informative but is not the approved manifest: raising the per-bucket limit to 1536 MiB lets one `(48,72,12,11)` bucket cover 91.45% of calls, with an estimated 1531.73 MiB workspace. This identifies memory, rather than excessive bucket count, as the immediate boundary.

## Padding distribution

The admitted bucket covers four small signatures. Its observed padding ranges are 0-22.6% for active edges, 0-18.4% for BA edges, 0-16.7% for source poses, and 0-20.0% for the pose window. These calls satisfy the 35% padding limit, but represent only 2.59% of traffic.

## Estimated workspace and Jetson headroom

- Observed PyTorch CUDA allocated memory: 1311.81-2198.83 MiB.
- Observed PyTorch CUDA reserved memory: 1646-5504 MiB.
- Hotel retry peak unified RAM: 12,395/15,503 MB; minimum observed free unified RAM was 3108 MB.
- Twenty-five percent of that optimistic free-memory upper bound is about 777 MB.

The 1531.73 MiB bucket required for 90% coverage exceeds both the approved 1024 MiB cap and 25% of observed free unified memory. Even the official 986.69 MiB bucket exceeds the latter conservative headroom bound. Capture-time `cudaMemGetInfo` was not separately recorded, so no stronger CUDA-only headroom claim is made.

## Unsupported or uncovered calls

- Unsupported calls: 0.
- Uncovered calls under the approved manifest: 1652 (97.41%).
- Most uncovered calls have 36-48 active edges and estimated fixed workspaces above 1 GiB; the dominant 48-edge signatures estimate to approximately 1518-1532 MiB.

## Profiling overhead check

The paired SmallCity-50 wall times were 127 s control and 126 s instrumented, so no positive wall-time overhead was resolved in this single pair. Both runs produced 16 keyframes.

The complete evaluator outputs were not identical:

| Metric | Control | Profiled | Change |
|---|---:|---:|---:|
| ATE Sim3 RMSE | 0.07644 m | 0.05712 m | -25.3% |
| Mean PSNR | 14.565 dB | 13.948 dB | -0.617 dB |
| Mean SSIM | 0.7025 | 0.6598 | -0.0427 |
| Exported samples | 12 | 14 | +16.7% |

On the ten common comparison frames, PSNR changed by -0.388 dB and SSIM by -0.0060, but the complete exported sample count and full-run quality metrics exceed the acceptance deltas. This is consistent with runtime-sensitive scheduling and is not evidence of numerical equivalence.

## Phase 1 decision

**NO-GO. Stop before implementing CUDA Graph replay.**

Successful hardware profiling established the real `43x77` signatures and found no unsupported call contract, but three required conditions fail:

1. Planned coverage is 2.59%, not at least 90%, under the 1024 MiB limit.
2. A bucket large enough for 90% coverage is about 1531.73 MiB and fails both the absolute and observed-headroom memory gates.
3. The instrumented/control complete evaluator outputs are not equivalent and exceed the PSNR, SSIM, and exported-sample deltas.

The next investigation should reduce the capture workspace boundary (especially correlation storage) or reduce active-edge capacity before reconsidering Phase 1. The current evidence does not authorize CUDA Graph runtime code or GPU BA work.
