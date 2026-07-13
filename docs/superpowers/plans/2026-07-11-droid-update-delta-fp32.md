# DROID Update-Core Delta-Head FP32 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict mixed-precision TensorRT update-core engine with only the delta neural path constrained to FP32, then repeat module parity and latency validation.

**Architecture:** A distinct reproducible `trtexec` script applies `obey` constraints to the three original ONNX delta nodes and writes isolated artifacts. A small CPU-safe verifier checks the final TensorRT layer summary before the existing sidecar generator, direct-binding runner, and benchmark are reused.

**Tech Stack:** Bash, Python 3.8, unittest, ONNX 1.14.1, TensorRT 8.5.2.2, CUDA 11.4, PyTorch 2.1 Jetson build.

**Worktree constraint:** Work in the existing dirty Jetson tree. Do not create a branch or worktree and do not stage or commit any file.

---

### Task 1: Lock the strict mixed-precision build specification

**Files:**
- Create: `tests/test_droid_update_deltafp32_build.py`
- Create: `reports/tensorrt_fp16/build_droid_update_deltafp32.sh`

- [ ] **Step 1: Write the failing build-script test**

Create a unittest that reads the proposed shell script and requires:

```python
DELTA_LAYERS = (
    "/delta/delta.0/Conv",
    "/delta/delta.1/Relu",
    "/delta/delta.2/Conv",
)

def test_build_uses_strict_delta_fp32_constraints(self):
    text = BUILD_SCRIPT.read_text()
    self.assertIn("--precisionConstraints=obey", text)
    for name in DELTA_LAYERS:
        self.assertIn(f"{name}:fp32", text)
    self.assertIn("--layerPrecisions=", text)
    self.assertIn("--layerOutputTypes=", text)

def test_build_is_isolated_from_pure_fp16_artifact(self):
    text = BUILD_SCRIPT.read_text()
    self.assertIn("droid_update_core_43x77_deltafp32.plan", text)
    self.assertNotIn("droid_update_core_43x77_fp16.plan", text)
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_droid_update_deltafp32_build
```

Expected: FAIL because `build_droid_update_deltafp32.sh` does not exist.

- [ ] **Step 3: Create the minimal build script**

Use the existing pure-FP16 script as the operational pattern, but target the new plan and dedicated logs. The `trtexec` command must contain:

```bash
--fp16
--precisionConstraints=obey
--layerPrecisions=/delta/delta.0/Conv:fp32,/delta/delta.1/Relu:fp32,/delta/delta.2/Conv:fp32
--layerOutputTypes=/delta/delta.0/Conv:fp32,/delta/delta.1/Relu:fp32,/delta/delta.2/Conv:fp32
--minShapes=net:1x128x43x77,inp:1x128x43x77,corr:1x196x43x77,flow:1x4x43x77
--optShapes=net:16x128x43x77,inp:16x128x43x77,corr:16x196x43x77,flow:16x4x43x77
--maxShapes=net:48x128x43x77,inp:48x128x43x77,corr:48x196x43x77,flow:48x4x43x77
--memPoolSize=workspace:1024MiB
--buildOnly
--verbose
--dumpLayerInfo
```

The script must refuse to overwrite an existing target, reject active `scripts/run.py`, `trtexec`, or `tegrastats`, use a trap to stop its own `tegrastats`, and write `status=<code> elapsed_s=<seconds>`.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run:

```bash
python -m unittest tests.test_droid_update_deltafp32_build
bash -n reports/tensorrt_fp16/build_droid_update_deltafp32.sh
```

Expected: tests PASS and `bash -n` exits zero.

### Task 2: Add CPU-safe precision-boundary verification

**Files:**
- Create: `scripts/acceleration/verify_droid_delta_precision.py`
- Modify: `tests/test_droid_update_deltafp32_build.py`

- [ ] **Step 1: Write failing verifier tests**

Use synthetic final layer-summary lines to require separate Float delta layers, a Half weight layer, and rejection of fused delta/weight or Half delta output:

```python
def test_accepts_separate_fp32_delta_and_fp16_weight(self):
    verify_delta_precision_log(VALID_LOG)

def test_rejects_delta_weight_fusion(self):
    with self.assertRaisesRegex(ValueError, "fused"):
        verify_delta_precision_log(FUSED_LOG)

def test_rejects_half_delta_layer(self):
    with self.assertRaisesRegex(ValueError, "FP32"):
        verify_delta_precision_log(HALF_DELTA_LOG)
```

- [ ] **Step 2: Verify RED**

Run the focused test and require an import failure for the missing verifier.

- [ ] **Step 3: Implement the minimal verifier and CLI**

Implement:

```python
DELTA_LAYER_NAMES = (
    "/delta/delta.0/Conv",
    "/delta/delta.2/Conv",
)

def verify_delta_precision_log(text):
    summary = [line for line in text.splitlines() if line.startswith("Layer(")]
    fused = [line for line in summary if "/delta/delta.0/Conv" in line and "/weight/weight.0/Conv" in line]
    if fused:
        raise ValueError("delta and weight layers remain fused")
    for name in DELTA_LAYER_NAMES:
        matches = [line for line in summary if name in line]
        if not matches or not any("Float[" in line for line in matches):
            raise ValueError(f"{name} is not confirmed FP32")
    weight = [line for line in summary if "/weight/weight.2/Conv" in line]
    if not weight or not any("Half[" in line for line in weight):
        raise ValueError("weight output layer is not confirmed FP16")
    return True
```

The CLI accepts one log path, prints `delta_fp32_precision=PASS`, and exits nonzero with the raw validation error otherwise.

- [ ] **Step 4: Verify GREEN**

Run the focused tests and `python -m py_compile` for the verifier.

### Task 3: Build and inspect the mixed engine

**Files:**
- Generate: `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan`
- Generate: `reports/tensorrt_fp16/droid_update_deltafp32_trtexec_build.log`
- Generate: `reports/tensorrt_fp16/droid_update_deltafp32_trtexec_tegrastats.log`
- Generate: `reports/tensorrt_fp16/droid_update_deltafp32_trtexec_status.txt`

- [ ] **Step 1: Confirm preconditions**

Verify the ONNX SHA remains `63498e2b6e72bc9733a5ab65deb70951070c52e6b0273caff58d089c5d89c08a`, the new plan does not exist, and no target process is active.

- [ ] **Step 2: Launch the build once**

Run the dedicated script detached on Jetson so local SSH timeouts do not terminate monitoring. Record the controller PID and poll status/log timestamps without restarting while it is alive.

- [ ] **Step 3: Require build success**

Require status zero, `&&&& PASSED TensorRT.trtexec`, a nonempty plan, and no remaining `tegrastats` process.

- [ ] **Step 4: Verify the actual precision boundary**

Run:

```bash
python scripts/acceleration/verify_droid_delta_precision.py \
  reports/tensorrt_fp16/droid_update_deltafp32_trtexec_build.log
```

Expected: `delta_fp32_precision=PASS`. If it fails, stop this branch without using `prefer`.

### Task 4: Generate and validate strict metadata

**Files:**
- Generate: `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.json`

- [ ] **Step 1: Generate the sidecar**

Run `scripts/acceleration/engine_metadata.py` with the new engine, original ONNX/checkpoint, full build command, unchanged profiles, and `--precision fp16_delta_fp32`.

- [ ] **Step 2: Verify hashes and bindings**

Require current engine, ONNX, and checkpoint SHA256 values to match the sidecar. Require the same seven float32 I/O bindings and dynamic shapes as the pure-FP16 engine.

- [ ] **Step 3: Strict-load the plan**

Instantiate `TensorRTEngine` with the new plan and sidecar in a fresh Jetson process. Expected: seven bindings and no validation exception.

### Task 5: Repeat module parity and latency benchmark

**Files:**
- Generate: `reports/tensorrt_fp16/droid_update_deltafp32_module_report.json`
- Generate: `reports/tensorrt_fp16/droid_update_deltafp32_module_report.md`

- [ ] **Step 1: Run the complete seeded benchmark**

Use `benchmark_droid_update_trt.py` with E=1,4,16,32,48, 10 warmups, and 50 measured iterations. Do not change parity formulas or thresholds.

- [ ] **Step 2: Check correctness gate**

Require every shape and finite check, cosine at least 0.999, and mean relative error at most 0.01 for `updated_net`, `delta`, and `weight` at every E.

- [ ] **Step 3: Check performance gate**

Require at least 20% improvement over PyTorch at E=16. Compare E=16 mixed latency with the existing pure-FP16 result of 34.9673 ms, but use only the PyTorch comparison for the integration decision.

- [ ] **Step 4: Record the decision**

Recommend later online integration only if both gates pass. Otherwise record the exact failed output/edge/metric and leave runtime behavior unchanged.

### Task 6: Regression and final report

**Files:**
- Modify: `reports/tensorrt_fp16/droid_update_deltafp32_module_report.md`

- [ ] **Step 1: Run focused and preserved tests in isolated processes**

Run the new build/verifier tests, `tests.test_trt_engine`, `tests.test_droid_update_trt`, and the 16 preserved DROID/profiling regression modules. Require all to pass.

- [ ] **Step 2: Run final environment checks**

Run Python compilation, `bash -n`, `pip check`, strict sidecar validation, all three hashes, and a standalone no-background-process check.

- [ ] **Step 3: Complete the report**

Add build elapsed time, plan size/hash, peak RAM/swap, proof of the precision boundary, all parity metrics, PyTorch/mixed/pure-FP16 E=16 latency, and the explicit go/no-go decision.

- [ ] **Step 4: Inspect worktree without mutation**

Record HEAD and relevant modified/untracked files. Do not stage, commit, reset, clean, or delete any existing user content.
