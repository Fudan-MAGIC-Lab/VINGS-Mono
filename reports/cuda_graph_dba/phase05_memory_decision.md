# DBA Memory Calibration Phase 0.5 Decision

## Environment and source commits

- Decision: **NO-GO for Phase 1 CUDA Graph runtime work**.
- Evidence root: `/home/jetson/VINGS-Mono/reports/cuda_graph_dba/phase05_20260714`.
- Evidence commit: `ce04e24781283109c5aef98442763271b9bf027a` on `codex/dba-memory-phase05`.
- Platform: Jetson Orin, Linux `5.10.216-tegra`, Python 3.8.20, Torch `2.1.0a0+41361538.nv23.06`, CUDA 11.4.
- Runtime shape/backend: `344x616 -> 43x77`, VO, Torch update, float16.
- The worktree was clean when the driver started. The evidence directory contains the Git HEAD, shared-root recursive submodule snapshot, Python packages, Torch/CUDA snapshot, run logs, and `tegrastats` logs.
- SmallCity details: `/home/jetson/VINGS-Mono/output/cuda_graph_dba_phase05_20260714/smallcity200/07-14-18-56-hierarchical-smallcit-cuda_graph_dba_phase05_smallcity200/runtime_profile_details.jsonl`, SHA-256 `9408f0e1db1fa48667627a537073a02910db86f9038c752790b32da5447f835b`.
- Hotel details: `/home/jetson/VINGS-Mono/output/cuda_graph_dba_phase05_20260714/hotel344/07-14-18-58-rtgslam-hote-cuda_graph_dba_phase05_hotel344/runtime_profile_details.jsonl`, SHA-256 `7561852b196130164373dbce9a6cbe90cc87a9f397317002c2d3cdb709f73811`.

## Evidence runs and sample coverage

| Run | Frames | Exit | Elapsed | DBA calls | Sampled signatures | Weighted sample coverage |
|---|---:|---:|---:|---:|---:|---:|
| SmallCity-200 | 200/200 | 0 | 127 s | 132 | 8 | 100% |
| Hotel | 405/405 | 0 | 1428 s | 1180 | 16 | 100% |
| Combined | 605/605 | 0 | 1555 s | 1312 | 16 unique | 100% |

The profiler produced 48 memory samples: two per signature per run when that signature appeared in both runs. Both required detail kinds were present, and no `run.py` or `tegrastats` process remained afterward.

## Resident eager state

Exact unique CUDA storage at capture entry was:

| Resident category | Minimum MiB | Maximum MiB |
|---|---:|---:|
| Full correlation pyramid | 658.40 | 1316.80 |
| Recurrent `net`/`inp` | 38.80 | 77.60 |
| Graph/update state | 2.26 | 3.48 |

No tensors were excluded by the CUDA-device storage filter. The full correlation pyramid is resident eager state; it is reported but excluded from incremental workspace for `update_aggregation_only`.

## Measured transient peaks

- Per-signature eager transient peaks ranged from **364.07 MiB to 723.70 MiB**.
- The largest directly measured stage-local peak was 723.70 MiB.
- Repeated-measurement peak spread was at most **0.2273%**, well below the 15% stability gate.
- The fitted update-only bucket uses a conservative calibrated transient of 735.06 MiB for its capacity signature.

## Calibration validation

| Boundary | Valid | Weighted MAPE | Held-out MAPE | Max repeat spread | Underpredicted | Unstable |
|---|---|---:|---:|---:|---:|---:|
| `corr_update_aggregation` | yes | 0.5363% | 0.5067% | 0.2273% | 0 | 0 |
| `update_aggregation_only` | yes | 1.3084% | 1.2405% | 0.2273% | 0 | 0 |

Both models pass the 95% sampled-call coverage, 10% weighted-MAPE, no-underprediction, and 15% repeat-spread gates. Each final estimate includes `max(64 MiB, 10% of modeled workspace)` after the pre-margin accuracy calculation.

## Conservative versus calibrated boundaries

The Phase 0 conservative planner covered only 44/1696 calls (2.59%) under 1024 MiB. Its sensitivity result needed a 1531.73 MiB bucket to cover 91.45%.

The corrected boundary model separates persistent fixed-address shadows from calibrated transient peaks:

| Model/boundary | Persistent shadow range | Result under 1024 MiB |
|---|---:|---|
| Phase 0 conservative | Included full eager/capture state and BA terms | 2.59% coverage |
| `corr_update_aggregation` | 751.59–1512.09 MiB | No admissible bucket; 0% coverage |
| `update_aggregation_only` | 93.19–195.29 MiB | One admissible bucket; 91.84% coverage |

Excluding the resident full correlation pyramid materially fixes the model, but the measured update/aggregation transient remains large enough that the resulting bucket sits almost exactly on the absolute limit.

## Revised bucket coverage and padding

The update-only manifest selects one bucket:

| Bucket | Covered calls | Coverage | Padding cap | Shadow | Transient | Margin | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `e48_ba74_s12_p11_float16_inactive_up` | 1205/1312 | 91.84% | 35% | 195.29 MiB | 735.06 MiB | 93.03 MiB | **1023.38 MiB** |

The bucket is only 0.62 MiB below the 1024 MiB cap. All 1312 calls are schema-supported; 107 are uncovered by the selected capacity/padding envelope. The full-correlation manifest contains no bucket because every candidate exceeds the workspace cap.

## Capture-time CUDA headroom

The sampled capture-entry `cudaMemGetInfo` minimum was **2,774,688,000 bytes (2646.15 MiB / 2.584 GiB)** during Hotel. The 25% headroom allowance is therefore **693,672,000 bytes (661.54 MiB)**.

The selected update-only bucket is 1,073,094,249 bytes (1023.38 MiB):

- 38.67% of the minimum measured capture-entry free CUDA memory;
- 379,422,249 bytes (361.85 MiB) above the 25% allowance;
- only 647,575 bytes (0.62 MiB) below the absolute 1024 MiB cap.

It passes the absolute cap but fails the independently required 25%-of-free-memory gate.

## Unsupported, unstable, or underpredicted signatures

- Schema-unsupported calls: 0.
- Unstable signatures: 0.
- Underpredicted validation signatures after margin: 0.
- Sampled weighted coverage: 100%.
- The update-only plan leaves 107 supported calls uncovered to remain within the absolute workspace and padding limits.

## Phase 1 decision

**NO-GO.** The recommended `update_aggregation_only` model is valid and reaches the 90% coverage target with one bucket, but that bucket consumes 38.67% of the minimum measured capture-entry free CUDA memory, exceeding the approved 25% limit. The full-correlation boundary cannot rescue the result and has no admissible bucket under 1024 MiB.

No CUDA Graph runtime, `max_factors` change, scheduling change, or GPU BA implementation is authorized. The next design investigation should first reduce update/aggregation transient or fixed-address shadow memory by at least about 362 MiB at the representative bucket. Reducing active-edge capacity remains a separate behavior-changing proposal that requires its own review and evidence plan.
