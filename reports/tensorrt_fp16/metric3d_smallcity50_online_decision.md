# Metric3D TensorRT Online SmallCity-50 Decision

Date: 2026-07-11

## Decision

**GO for a paired SmallCity-200 Metric3D TensorRT decision run.**

The opt-in online backend passed every approved SmallCity-50 gate. PyTorch remains the default. This result does not authorize DROID encoder conversion and does not make TensorRT the production default.

## Runs

| Variant | Run directory | Process status | Evaluator status | Wall time |
|---|---|---:|---:|---:|
| PyTorch | `output/smallcity_metric3d_online_20260711_retry1/pytorch/07-11-20-00-hierarchical-smallcit-jetson_smallcity_gt50_pytorch_metric3d_online_t10_retry1` | 0 | 0 | 126 s |
| TensorRT | `output/smallcity_metric3d_online_20260711_retry1/tensorrt/07-11-20-02-hierarchical-smallcit-jetson_smallcity_gt50_tensorrt_metric3d_online_t10_retry1` | 0 | 0 | 120 s |

Both variants used the same dirty worktree at `b7c68bb`, the 50-frame SmallCity subset, `training_iters=10`, `metric_depth_scale=0.75`, `frontend_save_buffer=64`, and the same balanced-fast mapping, pruning, pixel-budget, metric-depth scheduling, and VPI C++ motion-gate options. The TensorRT variant added only:

```text
--metric-depth-backend tensorrt
--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan
--tensorrt-strict
```

Engine SHA-256:

```text
fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9
```

## Actual backend evidence

PyTorch metadata:

```json
{
  "requested_backend": "torch",
  "actual_backend": "torch",
  "strict": false,
  "engine_path": null,
  "fallback_reason": null
}
```

TensorRT metadata:

```json
{
  "requested_backend": "tensorrt",
  "actual_backend": "tensorrt",
  "strict": true,
  "engine_path": "engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan",
  "fallback_reason": null
}
```

Each run logged its backend record exactly once. The TensorRT run did not fall back.

## Acceptance results

| Gate | PyTorch | TensorRT | Change | Limit | Result |
|---|---:|---:|---:|---:|---|
| Tracking/run failure | none | none | none | none | PASS |
| ATE Sim3 RMSE | 0.056609 m | 0.059765 m | +0.003156 m / +5.575% | at most max(0.10 m, 15%) | PASS |
| Mean PSNR | 13.949339 dB | 13.885726 dB | -0.063613 dB | no worse than -0.5 dB | PASS |
| Mean SSIM | 0.653102 | 0.653980 | +0.000878 | no worse than -0.02 | PASS |
| Keyframe count | 16 | 16 | 0%; lists byte-identical | at most 10% | PASS |
| Exported samples | 12 | 11 | -8.333% | at most 10% | PASS |
| Metric3D model mean | 474.322 ms | 174.613 ms | -299.709 ms / 63.187% faster | at least 20% faster | PASS |
| `frame_total` | 113.626 s | 107.729 s | -5.897 s / 5.190% faster | positive measurable gain | PASS |
| Peak system RAM | 10,007 MB | 10,026 MB | +19 MB | at most +500 MB | PASS |
| Shutdown | clean | clean | no residual process | no hang | PASS |

Both runs made 19 Metric3D calls. Preprocessing stayed effectively unchanged at 22.746 versus 22.985 ms mean, while the neural model boundary delivered the speedup. TensorRT total Metric3D time was 3.810 seconds versus 9.493 seconds for PyTorch.

The quick comparison tool found 10 common exported frames. Full evaluator sample counts differed by one, within the gate. Existing mapping/BA nondeterminism can change which export frames survive even when the tracking keyframe list is identical; this was already observed in Phase 0 controls.

## Preserved failed attempt

The first control attempt is preserved under:

```text
reports/tensorrt_fp16/metric3d_smallcity50_online_20260711/
output/smallcity_metric3d_online_20260711/pytorch/
```

It failed before TensorRT started because the paired driver omitted the established balanced-fast `--frontend-save-buffer 64` override and reverted to the 2500-frame default. RAM reached 13,622 MB, the largest free block fell to 512 KB, `NvMapMemAlloc` returned error 12, and cuDNN initialization then failed. A script regression test was added before retry1; no failed evidence was removed or overwritten.

## Evidence

```text
reports/tensorrt_fp16/metric3d_smallcity50_online_20260711_retry1/
reports/tensorrt_fp16/metric3d_smallcity50_online_20260711_retry1/tracking_comparison/summary.md
reports/tensorrt_fp16/metric3d_smallcity50_pair_controller_status_retry1.txt
```

Next authorized step: run a fresh, paired SmallCity-200 PyTorch/TensorRT comparison with unique prefixes and the same strict actual-backend reporting. Do not reuse the SmallCity-50 control as the SmallCity-200 baseline.
