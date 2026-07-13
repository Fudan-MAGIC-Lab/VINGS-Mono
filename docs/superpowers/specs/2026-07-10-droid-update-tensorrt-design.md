# DROID Update-Core TensorRT FP16 Design

## Goal

Build and validate a reusable TensorRT 8.5 runtime around the existing DROID update-core ONNX model. This phase ends at module-level correctness and latency evidence; it does not route live SLAM inference through TensorRT.

## Architecture

`TensorRTEngine` owns one deserialized plan, one execution context, and cached CUDA output tensors. Callers pass a complete mapping of contiguous CUDA PyTorch tensors. The runner validates names, dtypes, devices, and binding shapes, binds `data_ptr()` addresses, and enqueues with `execute_async_v2` on the current PyTorch CUDA stream without normal-path synchronization.

Every plan has a JSON sidecar containing hashes, runtime versions, compute capability, precision, build command, and binding/profile descriptions. Metadata mismatches fail before inference.

The DROID benchmark loads the same checkpoint and `forward_core` used for ONNX export, compares PyTorch and TensorRT at `E=1,4,16,32,48`, and measures synchronized CUDA-event latency only inside the benchmark. The plan is retained only if correctness passes and the representative `E=16` latency improves by at least 20%.

## Boundaries

- Included: shared runner, metadata generator, FP16 update-core plan, randomized module parity, latency benchmark, build/resource logs.
- Deferred: SLAM backend flags, live real-factor capture, Metric3D, fnet/cnet, GraphAgg, correlation, BA, and end-to-end runs.
- PyTorch remains the only runtime backend in `scripts/run.py` during this phase.

## Failure Policy

- TensorRT import adds only `/usr/lib/python3.8/dist-packages` after a normal import failure.
- Missing plans, metadata mismatches, unsupported shapes/dtypes, incomplete dynamic shapes, or enqueue failure raise explicit exceptions.
- No NumPy, PyCUDA, or host staging is allowed in the runner.
- Engine build/parser errors are recorded verbatim; no plugin or precision workaround is attempted without a separate decision.

## Acceptance

- CPU-safe metadata and validation tests pass without importing TensorRT eagerly.
- Jetson runner reuses output tensor objects for repeated equal shapes and executes on the current PyTorch stream.
- `E=1,4,16,32,48` outputs have exact shapes, finite values, cosine similarity at least 0.999, and mean relative error at most 1%.
- Representative component latency improves by at least 20%; otherwise the engine remains an experiment and is not integrated.
