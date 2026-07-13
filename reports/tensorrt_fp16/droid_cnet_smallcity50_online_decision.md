# DROID cnet TensorRT SmallCity-50 Online Decision

Date: 2026-07-12

## Scope

This paired run changed only the DROID cnet backend. Both variants used the same PyTorch Metric3D backend, SmallCity-50 dataset, balanced-fast arguments, input resolution, motion gate, mapping budgets, OFA path, and checkpoint.

- PyTorch control: `jetson_smallcity_gt50_pytorch_droid_cnet_online_t10`
- TensorRT candidate: `jetson_smallcity_gt50_tensorrt_droid_cnet_online_t10`
- Engine SHA-256: `6b90bb5c1d7ccd9d730f587df69adf269a8e1e10d71576cbe93c57008f90218e`

Both runs, backend checks, evaluators, and tracking comparison exited with status 0. The control recorded actual cnet backend `torch`. The candidate recorded requested/actual backend `tensorrt`, strict mode enabled, and no fallback.

## Acceptance results

| Gate | PyTorch | TensorRT | Delta | Limit | Result |
|---|---:|---:|---:|---:|:---:|
| Tracking failure | none | none | — | none | PASS |
| Keyframe list | 16 entries | 16 entries | byte-identical | identical | PASS |
| Export frame IDs | 7,8,9,10,11,12,15,16,19,24 | 7,8,9,11,13,15,16,19,32 | different | identical | **FAIL** |
| Exported samples | 10 | 9 | -1 (-10.0%) | identical | **FAIL** |
| ATE Sim3 RMSE | 0.050925 m | 0.080243 m | +0.029318 m | <= 0.01 m | **FAIL** |
| Mean PSNR | 14.396384 dB | 14.001549 dB | -0.394835 dB | <= 0.05 dB drop | **FAIL** |
| Mean SSIM | 0.674795 | 0.660947 | -0.013848 | <= 0.005 drop | **FAIL** |

The common-frame quick comparison used seven exported frames and changed mean PSNR from `17.065` to `16.135` and mean SSIM from `0.8336` to `0.8077`. Pose differences on the seven common frames remained small, but that does not override the full evaluator and export-identity failures.

## Runtime and resources

| Metric | PyTorch | TensorRT | Change |
|---|---:|---:|---:|
| `motion_filter_context_encoder` mean | 11.4876 ms | 6.1374 ms | -46.57% |
| `motion_filter_context_encoder` total | 0.3217 s | 0.1718 s | -0.1498 s |
| `frame_total` | 118.9955 s | 118.9100 s | -0.0855 s (-0.072%) |
| Driver elapsed | 131 s | 132 s | +1 s |
| Peak RAM | 11,482 MB | 11,630 MB | +148 MB |
| Peak swap | 312 MB | 313 MB | +1 MB |

The component speedup is real, but its end-to-end contribution is within run noise and does not compensate for the behavior and quality failures.

## Decision

**NO-GO for DROID cnet TensorRT online integration.** Keep cnet on PyTorch and do not run the conditional SmallCity-200 branch. PyTorch remains the default. The TensorRT cnet module artifact may be retained for offline investigation, but it is not authorized for production or paired continuation under the current parity thresholds.

Primary evidence is under `reports/tensorrt_fp16/droid_cnet_smallcity50_online_20260712/` and `output/smallcity_droid_cnet_online_20260712/`.
