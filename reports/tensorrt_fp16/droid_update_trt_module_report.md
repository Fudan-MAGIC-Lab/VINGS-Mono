# DROID Update-Core TensorRT FP16 Module Report

- Engine: `engines/tensorrt/droid/droid_update_core_43x77_fp16.plan`
- Checkpoint: `ckpts/droid.pth`
- Timing: 10 warmup, 50 measured iterations

## Build artifact

- TensorRT: 8.5.2.2; CUDA: 11.4; compute capability: 8.7; precision: FP16.
- Dynamic edge profile: E=1/16/48 (min/opt/max), fixed spatial shape 43x77.
- Engine size: 4,804,018 bytes (4.6 MiB).
- Engine SHA256: `d01104b384bc3883c6e42ece614af12f6c4739ac1b7c4f4ab715a63c34c38a72`.
- Source ONNX SHA256: `63498e2b6e72bc9733a5ab65deb70951070c52e6b0273caff58d089c5d89c08a`.
- Checkpoint SHA256: `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`.
- `trtexec` build status: PASS; elapsed 1042.35 s.
- Build resource log: peak system RAM 6,589 MB; peak system swap 312 MB across 1,040 samples.
- Module benchmark resource log: peak system RAM 8,958 MB; peak system swap 312 MB across 61 samples.
- Strict sidecar validation and real `TensorRTEngine` deserialization passed before benchmarking.

| E | PyTorch (ms) | TensorRT (ms) | Speedup | Improvement | Parity |
|---:|---:|---:|---:|---:|:---:|
| 1 | 10.2113 | 3.0863 | 3.309x | 69.78% | PASS |
| 4 | 27.7181 | 8.6533 | 3.203x | 68.78% | FAIL |
| 16 | 106.7408 | 34.9673 | 3.053x | 67.24% | FAIL |
| 32 | 193.7097 | 72.1060 | 2.686x | 62.78% | FAIL |
| 48 | 294.7791 | 110.9679 | 2.656x | 62.36% | FAIL |

## Decision

do not integrate: module parity thresholds were not met

## Output parity

### E=1

- `updated_net`: shape=[1, 128, 43, 77], finite=True, cosine=0.999997947, mean_relative_error=0.008095
- `delta`: shape=[1, 2, 43, 77], finite=True, cosine=0.999997025, mean_relative_error=0.009417
- `weight`: shape=[1, 2, 43, 77], finite=True, cosine=0.999989776, mean_relative_error=0.003250

### E=4

- `updated_net`: shape=[4, 128, 43, 77], finite=True, cosine=0.999997976, mean_relative_error=0.007950
- `delta`: shape=[4, 2, 43, 77], finite=True, cosine=0.999996618, mean_relative_error=0.011949
- `weight`: shape=[4, 2, 43, 77], finite=True, cosine=0.999994173, mean_relative_error=0.003377

### E=16

- `updated_net`: shape=[16, 128, 43, 77], finite=True, cosine=0.999997966, mean_relative_error=0.007962
- `delta`: shape=[16, 2, 43, 77], finite=True, cosine=0.999996779, mean_relative_error=0.012189
- `weight`: shape=[16, 2, 43, 77], finite=True, cosine=0.999992093, mean_relative_error=0.003471

### E=32

- `updated_net`: shape=[32, 128, 43, 77], finite=True, cosine=0.999997965, mean_relative_error=0.007929
- `delta`: shape=[32, 2, 43, 77], finite=True, cosine=0.999996806, mean_relative_error=0.012515
- `weight`: shape=[32, 2, 43, 77], finite=True, cosine=0.999992358, mean_relative_error=0.003449

### E=48

- `updated_net`: shape=[48, 128, 43, 77], finite=True, cosine=0.999997964, mean_relative_error=0.007927
- `delta`: shape=[48, 2, 43, 77], finite=True, cosine=0.999996796, mean_relative_error=0.011813
- `weight`: shape=[48, 2, 43, 77], finite=True, cosine=0.999992482, mean_relative_error=0.003451
