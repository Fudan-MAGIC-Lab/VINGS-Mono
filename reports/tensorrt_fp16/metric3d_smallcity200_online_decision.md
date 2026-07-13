# Metric3D TensorRT Online SmallCity-200 Decision

Date: 2026-07-11

## Original strict decision

**NO-GO for enabling Metric3D TensorRT as the recommended/default online backend.**

The strict SmallCity-200 SSIM gate failed: mean SSIM dropped by `0.022343`, exceeding the allowed `0.02` drop. PyTorch remains the default. Do not begin the DROID encoder branch on the basis of this run.

The TensorRT implementation remains available only as an explicit opt-in experimental backend. It ran strictly with no fallback and delivered a real end-to-end speed improvement, but the approved decision requires every quality and runtime gate to pass.

## User-approved exception

After reviewing the complete result, the user explicitly accepted the `0.022343` SSIM drop. This is recorded as a **conditional GO to continue the DROID fnet/cnet acceleration work**.

The original strict gate result above remains unchanged and auditable. The exception does not make Metric3D TensorRT the default: PyTorch remains the default, and Metric3D TensorRT remains explicit opt-in. The justification is that the limit was exceeded by only `0.002343`, while the 26 common exported frames showed an SSIM drop of `0.0142`, and every other quality, speed, memory, tracking, and shutdown gate passed.

## Runs

| Variant | Run directory | Process status | Evaluator status | Wall time |
|---|---|---:|---:|---:|
| PyTorch | `output/smallcity_metric3d_online_200_20260711/pytorch/07-11-20-14-hierarchical-smallcit-jetson_smallcity_gt200_pytorch_metric3d_online_t10` | 0 | 0 | 325 s |
| TensorRT | `output/smallcity_metric3d_online_200_20260711/tensorrt/07-11-20-20-hierarchical-smallcit-jetson_smallcity_gt200_tensorrt_metric3d_online_t10` | 0 | 0 | 317 s |

Both variants used the same worktree, 200-frame dataset, `training_iters=10`, `metric_depth_scale=0.75`, `frontend_save_buffer=64`, mapping/pruning/pixel budgets, metric-depth schedule, and VPI C++ motion gate. TensorRT added only its explicit backend, accepted engine, and strict flags.

Engine SHA-256:

```text
fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9
```

## Actual backend evidence

The PyTorch run recorded `requested_backend=torch`, `actual_backend=torch`, and no fallback. The TensorRT run recorded `requested_backend=tensorrt`, `actual_backend=tensorrt`, `strict=true`, and `fallback_reason=null`. Each backend line appeared exactly once in its run log.

## Acceptance results

| Gate | PyTorch | TensorRT | Change | Limit | Result |
|---|---:|---:|---:|---:|---|
| Tracking/run failure | none | none | none | none | PASS |
| ATE Sim3 RMSE | 0.535978 m | 0.568415 m | +0.032437 m / +6.052% | at most max(0.10 m, 15%) | PASS |
| Mean PSNR | 14.133403 dB | 13.796926 dB | -0.336477 dB | no worse than -0.5 dB | PASS |
| Mean SSIM | 0.685839 | 0.663496 | **-0.022343** | no worse than -0.02 | **FAIL** |
| Keyframe count | 58 | 58 | 0%; lists byte-identical | at most 10% | PASS |
| Exported samples | 39 | 38 | -2.564% | at most 10% | PASS |
| Metric3D model mean | 438.087 ms | 173.890 ms | -264.197 ms / 60.307% faster | at least 20% faster | PASS |
| `frame_total` | 313.649 s | 304.151 s | -9.498 s / 3.028% faster | positive measurable gain | PASS |
| Peak system RAM | 10,943 MB | 10,592 MB | -351 MB | at most +500 MB | PASS |
| Shutdown | clean | clean | no residual process | no hang | PASS |

Both variants made 44 Metric3D predictions. Total Metric3D time fell from 20.386 seconds to 8.784 seconds. Preprocessing remained stable at 22.657 versus 22.918 ms mean; the gain came from the neural model boundary.

The quick comparison tool found 26 common exported frames. On those common frames, mean quick SSIM changed from 0.8069 to 0.7927, a drop of 0.0142. The formal evaluator covers each run's full exported sample set and is the acceptance source; its 0.022343 drop controls the decision.

## Evidence

```text
reports/tensorrt_fp16/metric3d_smallcity200_online_20260711/
reports/tensorrt_fp16/metric3d_smallcity200_online_20260711/tracking_comparison/summary.md
reports/tensorrt_fp16/metric3d_smallcity200_pair_controller_status.txt
```

## Revised next action

Keep PyTorch as the default, retain Metric3D TensorRT as explicit opt-in, and proceed to the DROID fnet/cnet export-boundary audit. The DROID encoder work must pass its own module and paired-run gates. The user exception does not relax the existing DROID update-core `1%` parity threshold.
