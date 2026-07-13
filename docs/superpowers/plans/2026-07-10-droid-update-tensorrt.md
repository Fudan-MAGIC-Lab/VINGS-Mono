# DROID Update-Core TensorRT FP16 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable direct-binding TensorRT runner and validate a DROID update-core FP16 plan at module level.

**Architecture:** A lazy TensorRT wrapper validates JSON metadata and directly binds PyTorch CUDA tensor addresses on the current stream. A separate metadata CLI introspects built plans, while a benchmark compares the existing PyTorch `forward_core` against TensorRT over the supported dynamic edge range.

**Tech Stack:** Python 3.8, PyTorch 2.1 Jetson build, TensorRT 8.5.2, CUDA 11.4, unittest, trtexec.

---

### Task 1: Metadata model and CPU-safe validation

**Files:**
- Create: `scripts/acceleration/engine_metadata.py`
- Create: `tests/test_trt_engine.py`

- [ ] Write failing tests for SHA256 calculation, required metadata keys, engine hash mismatch, and binding metadata mismatch without importing TensorRT.
- [ ] Run `python -m unittest tests.test_trt_engine` and confirm the missing metadata API causes failure.
- [ ] Implement JSON load/write, file hashing, required-key validation, and binding comparison.
- [ ] Re-run the test and require PASS.

### Task 2: Direct-binding TensorRT runner

**Files:**
- Create: `scripts/acceleration/trt_engine.py`
- Modify: `tests/test_trt_engine.py`

- [ ] Add failing fake-engine tests for missing/unexpected inputs, CPU/non-contiguous tensors, dtype mismatch, dynamic binding shapes, cached outputs, current-stream enqueue, and enqueue failure.
- [ ] Implement lazy TensorRT import, plan deserialization, binding discovery, input validation, dynamic shape assignment, cached `torch.empty` CUDA outputs, pointer binding, and `execute_async_v2`.
- [ ] Run CPU-safe tests, then a Jetson integration test after a real plan exists.

### Task 3: Build update-core FP16 plan and metadata

**Files:**
- Generate: `engines/tensorrt/droid/droid_update_core_43x77_fp16.plan`
- Generate: `engines/tensorrt/droid/droid_update_core_43x77_fp16.json`

- [ ] Verify no SLAM/tegrastats process is running and start a dedicated resource log.
- [ ] Build with FP16, workspace 1024 MiB, and profiles `E=1/16/48` for all four inputs.
- [ ] Record the full trtexec command and parser/build output.
- [ ] Generate metadata containing engine/ONNX/checkpoint hashes, TensorRT/CUDA/device data, build command, bindings, and profiles.
- [ ] Load the plan through `TensorRTEngine` in strict mode.

### Task 4: Module parity and latency benchmark

**Files:**
- Create: `scripts/profiling/benchmark_droid_update_trt.py`
- Create: `tests/test_droid_update_trt.py`

- [ ] Write failing tests for parity metric calculation and benchmark result schema.
- [ ] Implement checkpoint loading, seeded inputs for `E=1,4,16,32,48`, PyTorch/TensorRT warmup, CUDA-event timing, and JSON/Markdown reports.
- [ ] Require exact shapes, finite values, cosine at least 0.999, and mean relative error at most 1% for every output.
- [ ] Require at least 20% TensorRT latency improvement at `E=16` before recommending integration.

### Task 5: Verification and report

**Files:**
- Create: `reports/tensorrt_fp16/droid_update_trt_module_report.md`

- [ ] Run all new tests plus the existing DROID/profiling regression modules in isolated Python processes.
- [ ] Re-run metadata validation, plan load, parity, and benchmark from clean processes.
- [ ] Verify no background benchmark process remains and inspect the dirty worktree without staging, committing, or removing unrelated files.
- [ ] Report correctness, PyTorch latency, TensorRT latency, speedup, engine memory/build cost, and the explicit integrate/do-not-integrate decision.
