# DROID Update-Core TensorRT FP16 + Delta FP32 Module Report

- Engine: `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan`
- Checkpoint: `ckpts/droid.pth`
- Timing: 10 warmup, 50 measured iterations

## Build artifact and precision proof

- TensorRT 8.5.2.2, CUDA 11.4, compute capability 8.7.
- Precision policy: FP16 enabled globally; `delta.0/Conv`, `delta.1/Relu`, and
  `delta.2/Conv` use strict FP32 compute and FP32 outputs via
  `precisionConstraints=obey`.
- The final TensorRT layer summary shows separate Float delta layers and Half
  weight layers. The earlier delta/weight cross-branch fusion is absent.
- Engine size: 5,137,580 bytes (4.9 MiB).
- Engine SHA256: `2659d4f195eb64ddbaad634259810fb187f63b11572636a0e20c06ab8e49c516`.
- Source ONNX SHA256: `63498e2b6e72bc9733a5ab65deb70951070c52e6b0273caff58d089c5d89c08a`.
- Checkpoint SHA256: `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`.
- Build status: PASS; elapsed 1,010 seconds.
- Build resource log: peak system RAM 6,617 MB and peak system swap 311 MB
  across 1,007 samples.
- Benchmark resource log: peak system RAM 9,701 MB and peak system swap 311 MB
  across 63 samples.
- Strict sidecar validation and real `TensorRTEngine` deserialization passed.

| E | PyTorch (ms) | TensorRT (ms) | Speedup | Improvement | Parity |
|---:|---:|---:|---:|---:|:---:|
| 1 | 10.2051 | 3.2635 | 3.127x | 68.02% | PASS |
| 4 | 24.9007 | 9.4348 | 2.639x | 62.11% | FAIL |
| 16 | 107.0562 | 38.5016 | 2.781x | 64.04% | FAIL |
| 32 | 194.8745 | 78.2311 | 2.491x | 59.86% | FAIL |
| 48 | 296.6215 | 120.5136 | 2.461x | 59.37% | FAIL |

## Decision

do not integrate: module parity thresholds were not met

The performance gate passes, but the correctness gate does not. At E=16,
forcing the delta head to FP32 reduces delta mean relative error from 1.21887%
to 1.06598% (a 12.54% reduction), while latency rises from 34.9673 ms for the
pure-FP16 engine to 38.5016 ms (10.11% slower). The remaining delta error is
stable on an independent E=16 reproduction and is consistent with propagation
of the approximately 0.796% error already present in the FP16 `updated_net`.
The 1% acceptance threshold is not relaxed.

## Output parity

### E=1

- `updated_net`: shape=[1, 128, 43, 77], finite=True, cosine=0.999997947, mean_relative_error=0.008095
- `delta`: shape=[1, 2, 43, 77], finite=True, cosine=0.999998371, mean_relative_error=0.008196
- `weight`: shape=[1, 2, 43, 77], finite=True, cosine=0.999989776, mean_relative_error=0.003250

### E=4

- `updated_net`: shape=[4, 128, 43, 77], finite=True, cosine=0.999997976, mean_relative_error=0.007950
- `delta`: shape=[4, 2, 43, 77], finite=True, cosine=0.999998117, mean_relative_error=0.010466
- `weight`: shape=[4, 2, 43, 77], finite=True, cosine=0.999994173, mean_relative_error=0.003377

### E=16

- `updated_net`: shape=[16, 128, 43, 77], finite=True, cosine=0.999997966, mean_relative_error=0.007962
- `delta`: shape=[16, 2, 43, 77], finite=True, cosine=0.999998238, mean_relative_error=0.010660
- `weight`: shape=[16, 2, 43, 77], finite=True, cosine=0.999992093, mean_relative_error=0.003471

### E=32

- `updated_net`: shape=[32, 128, 43, 77], finite=True, cosine=0.999997965, mean_relative_error=0.007929
- `delta`: shape=[32, 2, 43, 77], finite=True, cosine=0.999998296, mean_relative_error=0.010759
- `weight`: shape=[32, 2, 43, 77], finite=True, cosine=0.999992358, mean_relative_error=0.003449

### E=48

- `updated_net`: shape=[48, 128, 43, 77], finite=True, cosine=0.999997964, mean_relative_error=0.007927
- `delta`: shape=[48, 2, 43, 77], finite=True, cosine=0.999998274, mean_relative_error=0.010379
- `weight`: shape=[48, 2, 43, 77], finite=True, cosine=0.999992482, mean_relative_error=0.003451
