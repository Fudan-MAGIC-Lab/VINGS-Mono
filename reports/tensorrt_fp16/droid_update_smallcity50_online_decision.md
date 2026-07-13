# DROID Update-Core Delta-FP32 TensorRT SmallCity-50 Decision

Date: 2026-07-12

## Scope

This speed-first paired gate changed only the DROID update-core backend. Metric3D, fnet, cnet, GraphAgg, correlation construction, BA, upsample, resolution, motion gate, mapping parameters, and checkpoint remained identical and on their existing backends.

- Engine: `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan`
- Engine SHA-256: `2659d4f195eb64ddbaad634259810fb187f63b11572636a0e20c06ab8e49c516`
- Precision: `fp16_delta_fp32`
- Accepted run: `reports/tensorrt_fp16/droid_update_smallcity50_online_20260712_retry2/`

The original attempt preserved a strict failure on online AMP FP16 inputs. Retry1 preserved a strict failure on valid strided CovisibleGraph flow. Regression tests were added before the adapter was changed to normalize FP16/FP32 and strided inputs to contiguous FP32 TensorRT bindings while restoring the recurrent input dtype on output. Retry2 is the first completed pair.

Both retry2 runs, backend checks, evaluators, and tracking comparison exited with status 0. The candidate recorded strict actual backend `tensorrt`, precision `fp16_delta_fp32`, and no fallback.

## Relaxed acceptance results

| Gate | PyTorch | TensorRT | Change | Approved limit | Result |
|---|---:|---:|---:|---:|:---:|
| Tracking failure | none | none | — | none | PASS |
| ATE Sim3 RMSE | 0.074169 m | 0.064707 m | -12.76% | <=20% increase | PASS |
| Mean PSNR | 14.400798 dB | 13.290965 dB | -1.109833 dB | <=0.5 dB drop | **FAIL** |
| Mean SSIM | 0.692210 | 0.620412 | -0.071798 | <=0.03 drop | **FAIL** |
| Exported samples | 12 | 10 | -16.67% | <=10% change | **FAIL** |
| `frame_total` | 119.1256 s | 104.5651 s | -12.22% | >=5% improvement | PASS |

The 16-entry keyframe lists were byte-identical, but exported IDs changed from `7,8,9,10,11,13,15,17,19,20,27,32` to `7,8,9,10,12,14,15,16,19,27`.

## Performance and resources

| Metric | PyTorch | TensorRT | Change |
|---|---:|---:|---:|
| `covisible_graph_update_op` total | 32.5726 s | 18.1823 s | -44.18% |
| `covisible_graph_update_op` mean | 195.0457 ms | 108.8762 ms | -44.18% |
| `frontend_dba_update` total | 70.7587 s | 56.0967 s | -20.72% |
| `frame_total` | 119.1256 s | 104.5651 s | -12.22% |
| Driver elapsed | 131 s | 118 s | -9.92% |
| Peak RAM | 11,379 MB | 11,604 MB | +225 MB |
| Peak swap | 313 MB | 313 MB | unchanged |

The component and end-to-end speedups are material. However, three approved relaxed correctness gates failed.

## Decision

**NO-GO under the approved relaxed gate.** Keep the update backend default on PyTorch and stop before the Metric3D+update combined pair and SmallCity-200. The opt-in implementation and engine may be retained for further speed/quality policy experiments, but they are not authorized as the recommended runtime configuration by this decision.

### Subsequent user-authorized experiment

The user later explicitly authorized running the combined Metric3D+update pair despite this stop condition. That experiment does not change the update-only verdict. Its evidence and decision are recorded in `reports/tensorrt_fp16/metric3d_update_smallcity50_online_decision.md`.
