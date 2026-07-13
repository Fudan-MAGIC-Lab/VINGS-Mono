# Metric3D + DROID Update-Core TensorRT SmallCity-50 Trial

Date: 2026-07-12

## Scope

The update-only run failed its previously approved relaxed gate, but the user explicitly authorized this combined trial. This is an experimental override to collect evidence, not a retroactive pass for the update-only decision.

The paired candidate enabled exactly two strict TensorRT backends:

- Metric3D: `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan`, SHA-256 `fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9`.
- DROID update core: `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan`, SHA-256 `2659d4f195eb64ddbaad634259810fb187f63b11572636a0e20c06ab8e49c516`, precision `fp16_delta_fp32`.

Both variants, backend checks, evaluators, and tracking comparison exited with status 0. Runtime metadata confirmed both candidate backends were actual strict TensorRT with no fallback. fnet and cnet remained PyTorch, and all balanced-fast parameters were unchanged.

Evidence: `reports/tensorrt_fp16/metric3d_update_smallcity50_online_20260712_try1/`.

## Paired results

| Gate / metric | PyTorch | Combined TensorRT | Change | Previous relaxed limit | Result |
|---|---:|---:|---:|---:|:---:|
| Tracking failure | none | none | — | none | PASS |
| ATE Sim3 RMSE | 0.057683 m | 0.072073 m | +24.947% | <=20% increase | **FAIL** |
| Mean PSNR | 14.540591 dB | 14.392889 dB | -0.147702 dB | <=0.5 dB drop | PASS |
| Mean SSIM | 0.691400 | 0.683186 | -0.008214 | <=0.03 drop | PASS |
| Exported samples | 10 | 11 | +10.0% | <=10% change | PASS |
| `frame_total` | 115.677 s | 98.574 s | -14.785% | >=5% improvement | PASS |

The 16-entry keyframe lists were byte-identical. Exported IDs differed: PyTorch exported `7,8,9,10,11,13,15,19,20,27`; TensorRT exported `7,8,9,10,11,12,13,16,17,20,32`.

## Performance and resources

| Metric | PyTorch | Combined TensorRT | Improvement |
|---|---:|---:|---:|
| `metric_depth_model` | 9.011 s | 3.357 s | 62.746% |
| `covisible_graph_update_op` | 31.972 s | 18.174 s | 43.157% |
| `frontend_dba_update` | 68.478 s | 56.667 s | 17.248% |
| Tracking | 82.538 s | 64.176 s | 22.247% |
| `frame_total` | 115.677 s | 98.574 s | 14.785% |
| Driver elapsed | 127 s | 112 s | 11.811% |

Peak RAM increased from 10,491 MB to 11,576 MB (+1,085 MB). Peak swap remained 313 MB.

## Decision

The combination is technically viable and produces a material end-to-end speedup, but it is still a **NO-GO under the previously approved relaxed gate** because ATE increased by 24.947%, above the 20% limit. Unlike the update-only pair, PSNR, SSIM, sample count, tracking stability, and speed all passed their relaxed limits.

For a speed-first policy that explicitly accepts about 25% SmallCity-50 ATE regression and roughly 1.1 GB additional peak RAM, this combination is the strongest online TensorRT candidate tested so far. Keep it explicit opt-in and keep all defaults on PyTorch until that policy is explicitly approved or repeated-run evidence is collected.
