# Metric3D TensorRT FP16 Module Report

- Engine: `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan`
- Checkpoint: `ckpts/metric_depth_vit_small_800k.pth`
- Timing: 5 warmup, 20 measured iterations per frame
- Backend separation: PyTorch model released before TensorRT engine load

| Frame | PyTorch model (ms) | TensorRT model (ms) | Raw cosine | Raw MRE | Final cosine | Final MRE |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 408.9749 | 164.6574 | 0.999996516 | 0.006726 | 0.999997363 | 0.005922 |
| 5 | 408.4756 | 164.6332 | 0.999975135 | 0.008241 | 0.999979933 | 0.007366 |
| 10 | 409.4611 | 164.6291 | 0.999992048 | 0.008364 | 0.999993596 | 0.007423 |
| 15 | 408.9730 | 164.5471 | 0.999993244 | 0.006420 | 0.999995303 | 0.005816 |
| 20 | 409.2084 | 164.7034 | 0.999993244 | 0.004758 | 0.999994445 | 0.004234 |
| 25 | 408.9494 | 165.0023 | 0.999992069 | 0.005227 | 0.999993780 | 0.004678 |
| 30 | 409.6483 | 164.5825 | 0.999994524 | 0.006049 | 0.999995637 | 0.005476 |
| 35 | 409.6996 | 164.7828 | 0.999981452 | 0.006618 | 0.999983496 | 0.005902 |
| 40 | 407.8087 | 164.4836 | 0.999994569 | 0.006536 | 0.999995681 | 0.005789 |
| 45 | 408.8137 | 164.5618 | 0.999992623 | 0.006713 | 0.999993810 | 0.005972 |

## Latency aggregates

- PyTorch model mean/median/p90: 409.0013 / 408.9739 / 409.6534 ms
- TensorRT model mean/median/p90: 164.6583 / 164.6311 / 164.8047 ms
- Model speedup: 2.484x
- Model improvement: 59.74%
- Estimated PyTorch/TensorRT pipeline: 432.0261 / 187.2378 ms

## Decision

integration candidate: parity passed and mean model latency improved by at least 20%

## Reproducibility and resource evidence

- Engine SHA-256: `fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9`
- Engine size: 80,266,391 bytes
- Source ONNX: `engines/tensorrt/metric3d/metric3d_v2s_448x784_opset16.onnx`
- Source ONNX SHA-256: `601152c515e80fc4d1a92ee7d9f13c6305ede6d4d88d2f15d1f560a4aa574d2b`
- TensorRT build: `status=0`, elapsed 1,254 s; peak system RAM 12,816 MB, peak GPU temperature 64.187 C, peak GR3D utilization 99%
- Module benchmark retry: `status=0`, elapsed 164 s; peak system RAM 5,043 MB, peak GPU temperature 72.531 C, peak GR3D utilization 99%
- CUDA memory allocated after releasing the PyTorch model: 8,519,680 bytes
- Build resource log: `reports/tensorrt_fp16/metric3d_opset16_trtexec_tegrastats.log`
- Benchmark resource log: `reports/tensorrt_fp16/metric3d_trt_benchmark_tegrastats_retry1.log`
- The original benchmark failure is preserved in `metric3d_trt_benchmark.log`; retry1 fixes only the TensorRT input contiguity contract.
