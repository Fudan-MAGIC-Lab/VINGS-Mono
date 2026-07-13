# DROID Update-Core TensorRT Online Relaxed-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict opt-in online backend for the existing delta-FP32 DROID update-core engine and evaluate its speed-first SmallCity behavior.

**Architecture:** Preserve `UpdateModule.forward()` and PyTorch GraphAgg while dispatching only flattened `forward_core` to `TensorRTEngine`. Strict mode releases the neural PyTorch core before loading the plan; non-strict mode retains one observable fallback. A guarded pair changes only the update backend and applies the approved relaxed quality gates plus a mandatory 5% `frame_total` improvement.

**Tech Stack:** Python 3.8, PyTorch 2.1, CUDA 11.4, TensorRT 8.5.2, unittest, Bash, SmallCity evaluator, runtime profiler.

**Worktree constraint:** Execute inline in the existing dirty Jetson tree. Do not create a branch/worktree or stage, commit, reset, clean, or delete user content.

---

### Task 1: TDD the TensorRT core adapter

**Files:**
- Create: `scripts/frontend/droid_update_backends.py`
- Create: `tests/test_droid_update_trt_backend.py`

- [ ] Write failing tests for exact CUDA float32 contiguous inputs, shapes `net/inp [E,128,43,77]`, `corr [E,196,43,77]`, `flow [E,4,43,77]`, E=1..48, and exact float32 outputs.
- [ ] Verify RED because `DroidUpdateTensorRTAdapter` is missing.
- [ ] Implement the adapter with lazy `TensorRTEngine` import and bindings `net/inp/corr/flow -> updated_net/delta/weight`.
- [ ] Verify GREEN and ensure module import does not eagerly import TensorRT.

### Task 2: TDD dispatch, lifecycle, and fallback

**Files:**
- Modify: `scripts/frontend/droid_net.py`
- Modify: `scripts/frontend/droid_update_backends.py`
- Modify: `tests/test_droid_update_trt_backend.py`

- [ ] Write failing tests requiring `forward_core_torch`, default PyTorch equivalence, strict module release while preserving `agg`, strict failure propagation, one non-strict fallback, and backend metadata updates.
- [ ] Verify RED for the missing dispatch/install APIs.
- [ ] Move the existing core body unchanged to `forward_core_torch`; make `forward_core` dispatch to an optional backend or the PyTorch method.
- [ ] Implement strict/non-strict installation without changing `forward()` tuple/GraphAgg behavior.
- [ ] Verify focused update-core, profiling, and GraphAgg regressions.

### Task 3: Config, CLI, profiler, and real-engine smoke

**Files:**
- Modify: `configs/hierarchical/smallcity.yaml`
- Modify: `scripts/run.py`
- Modify: `scripts/frontend/dbaf.py`
- Modify: `tests/test_jetson_runtime_overrides.py`
- Modify: `tests/test_dbaf_runtime_profiling.py`
- Modify: `tests/test_droid_update_trt_backend.py`

- [ ] Write RED tests for defaults, CLI overrides, actual backend metadata, and strict real-engine execution at E=1,4,16,32,48.
- [ ] Add `droid_update_backend`/`droid_update_engine`, install after checkpoint loading, and propagate status to `RuntimeProfiler`.
- [ ] Run the real delta-FP32 plan smoke and require finite exact outputs with no fallback.
- [ ] Run the full isolated regression set, syntax, `pip check`, hash/sidecar, artifact, and process checks.

### Task 4: Guarded update-only SmallCity-50 pair

**Files:**
- Create: `reports/tensorrt_fp16/run_droid_update_smallcity50_pair.sh`
- Create: `tests/test_droid_update_smallcity50_pair_script.py`

- [ ] Write RED content tests requiring unique 20260712 prefixes, overwrite/process guards, frozen balanced-fast arguments, environment snapshots, evaluator/comparison, and backend verification.
- [ ] Implement a driver whose candidate adds only `--droid-update-backend tensorrt`, the delta-FP32 engine path, and `--tensorrt-strict`.
- [ ] Verify tests and `bash -n`, then confirm dataset/engine/hash/process preconditions.
- [ ] Run control then candidate once with separate logs, status, output, and `tegrastats`.

### Task 5: Apply relaxed decision and conditional combined pair

**Files:**
- Create: `reports/tensorrt_fp16/droid_update_smallcity50_online_decision.md`
- Conditional create: `reports/tensorrt_fp16/run_metric3d_update_smallcity50_pair.sh`
- Conditional create: `tests/test_metric3d_update_smallcity50_pair_script.py`

- [ ] Compare backend truth, failures, ATE, PSNR, SSIM, sample count, update-op latency, `frame_total`, RAM/swap, and shutdown state.
- [ ] Require ATE <=20% increase, PSNR drop <=0.5 dB, SSIM drop <=0.03, sample-count change <=10%, and `frame_total` improvement >=5%.
- [ ] If any gate fails, record NO-GO and stop before combined/SmallCity-200 runs.
- [ ] If all gates pass, TDD a combined Metric3D+update SmallCity-50 driver and evaluate it before authorizing SmallCity-200.

### Task 6: Handoff and final verification

**Files:**
- Modify: `docs/HANDOFF_TENSORRT_FP16_DROID_METRIC3D_2026-07-10.md`

- [ ] Record implementation, exact backend, module evidence, paired results, decision, and whether the combined branch ran.
- [ ] Re-run focused tests and every preserved module in isolated processes.
- [ ] Run real-engine smoke, `pip check`, Python/Bash syntax, evidence existence, scoped patch checks, and residual-process checks.
- [ ] Preserve the current dirty worktree without git integration actions.
