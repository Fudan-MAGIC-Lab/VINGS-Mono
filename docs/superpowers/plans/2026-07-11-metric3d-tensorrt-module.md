# Metric3D TensorRT FP16 Engine and Module Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strict static Metric3D FP16 TensorRT plan and decide whether it qualifies for online integration using ten-frame real-input parity and latency evidence.

**Architecture:** A dedicated guarded shell script builds the plan without touching the online runtime. The existing metadata/runner infrastructure validates the artifact, while a new benchmark runs PyTorch and TensorRT sequentially so the two complete backends do not remain resident together.

**Tech Stack:** Bash, Python 3.8, unittest, TensorRT 8.5.2.2, CUDA 11.4, PyTorch 2.1 Jetson build, OpenCV, NumPy.

**Worktree constraint:** Execute in the existing dirty Jetson worktree. Do not create a branch/worktree and do not stage, commit, reset, clean, or remove unrelated files.

---

### Task 1: TDD the guarded static build command

**Files:**
- Create: `tests/test_metric3d_trt_build.py`
- Create: `reports/tensorrt_fp16/build_metric3d_fp16.sh`

- [ ] **Step 1: Write the failing build-script tests**

Require the script to contain the accepted ONNX path, distinct plan/log paths,
`--fp16`, `--memPoolSize=workspace:1024MiB`, `--buildOnly`, `--verbose`, and
`--dumpLayerInfo`. Require it not to overwrite the ONNX or any DROID plan and
to guard `[s]cripts/run.py|[t]rtexec|[t]egrastats`.

```python
def test_build_uses_static_metric3d_fp16_artifacts(self):
    text = BUILD_SCRIPT.read_text()
    self.assertIn("metric3d_v2s_448x784.onnx", text)
    self.assertIn("metric3d_v2s_448x784_fp16.plan", text)
    self.assertIn("--fp16", text)
    self.assertIn("--memPoolSize=workspace:1024MiB", text)

def test_build_is_guarded_and_reproducible(self):
    text = BUILD_SCRIPT.read_text()
    self.assertIn("[s]cripts/run.py|[t]rtexec|[t]egrastats", text)
    self.assertIn("trap cleanup EXIT", text)
    self.assertIn("--dumpLayerInfo", text)
```

- [ ] **Step 2: Verify RED**

Run `python -m unittest tests.test_metric3d_trt_build` and require failure
because the shell script does not exist.

- [ ] **Step 3: Create the minimal build script**

Follow the existing DROID build-script lifecycle: verify ONNX, reject an
existing plan, reject active target processes, launch dedicated `tegrastats`,
trap cleanup, run:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx \
  --saveEngine=engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan \
  --fp16 \
  --memPoolSize=workspace:1024MiB \
  --buildOnly \
  --verbose \
  --dumpLayerInfo
```

Write `status=<code> elapsed_s=<seconds>` and use dedicated build/resource logs.

- [ ] **Step 4: Verify GREEN and Bash syntax**

Run the focused unittest and `bash -n`; both must pass.

### Task 2: Build the plan once and preserve first-failure evidence

**Files:**
- Generate: `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan`
- Generate: `reports/tensorrt_fp16/metric3d_trtexec_build.log`
- Generate: `reports/tensorrt_fp16/metric3d_trtexec_tegrastats.log`
- Generate: `reports/tensorrt_fp16/metric3d_trtexec_status.txt`

- [ ] **Step 1: Export and validate an independent opset-16 source**

Add a tested `--opset-version` exporter argument, write
`metric3d_v2s_448x784_opset16.onnx`, and repeat the same ten-frame raw/final
ORT CUDA validation. Preserve the accepted opset-17 graph. Require opset 16,
zero `LayerNormalization` nodes, and the unchanged parity thresholds before
changing the build source.

- [ ] **Step 2: Check build preconditions**

Verify the recorded opset-16 source SHA256, absent target plan, no active target
process, and valid shell syntax.

- [ ] **Step 3: Launch one detached build**

Use `nohup` with a dedicated controller log and poll the same PID, build log,
status file, plan, and `tegrastats`. Do not restart while the controller or
`trtexec` is alive.

- [ ] **Step 4: Apply the build gate**

Require status zero, `&&&& PASSED TensorRT.trtexec`, nonempty plan, and no
remaining builder/`tegrastats`. If the parser/build fails, retain the raw first
error and stop before metadata or benchmark work.

- [ ] **Step 5: Inspect final layer information**

Record final input/output formats and confirm no fatal parser/build messages.
Do not infer parity from the layer log.

### Task 3: Generate strict metadata and real-load the engine

**Files:**
- Generate: `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.json`

- [ ] **Step 1: Generate sidecar metadata**

Run `scripts/acceleration/engine_metadata.py` with the engine, ONNX, Metric3D
checkpoint, full `trtexec` command, precision `fp16`, and profiles `{}`.

- [ ] **Step 2: Verify metadata**

Require matching engine/ONNX/checkpoint hashes, TensorRT 8.5.2.2, CUDA 11.4,
compute capability 8.7, and exactly two float32 static bindings:

```text
rgb   input  [1,3,448,784]
depth output [1,1,448,784]
```

- [ ] **Step 3: Strict-load through TensorRTEngine**

Instantiate the real plan in a fresh Jetson process. Require two bindings and
no metadata/runtime validation exception.

### Task 4: TDD benchmark statistics and decision logic

**Files:**
- Create: `tests/test_metric3d_trt.py`
- Create: `scripts/profiling/benchmark_metric3d_trt.py`

- [ ] **Step 1: Write failing CPU-safe tests**

Test `latency_statistics([10, 12, 14])` for count/mean/median/p90. Build ten
synthetic frame cases with raw/final parity, PyTorch/TensorRT model latency,
preprocess, and postprocess timing. Require `build_report()` to accept only
when all ten frame IDs and both parity boundaries pass and mean model
improvement is at least 0.20.

```python
def test_report_accepts_complete_fast_parity_result(self):
    report = build_report(cases, engine="metric.plan", checkpoint="metric.pth")
    self.assertTrue(report["recommendation"]["integrate"])
    self.assertGreaterEqual(report["latency"]["model_improvement"], 0.20)

def test_report_rejects_parity_or_speed_failure(self):
    # One case has MRE 0.0201, then a separate case has <20% improvement.
```

- [ ] **Step 2: Verify RED**

Run `python -m unittest tests.test_metric3d_trt` and require import failure for
the missing benchmark module.

- [ ] **Step 3: Implement minimal statistics and report builder**

Use NumPy percentiles. Require exact frame IDs `(0,5,...,45)`, exact
`raw`/`final` boundaries, finite values, cosine at least 0.999, MRE at most
0.02, and model improvement `(torch_mean-trt_mean)/torch_mean >= 0.20`.
Report model mean/median/p90, speedup, improvement, mean preprocess/postprocess,
and estimated PyTorch/TensorRT pipeline totals.

- [ ] **Step 4: Verify GREEN**

Run the focused unittest and Python compilation; both must pass.

### Task 5: Implement and run the two-phase real benchmark

**Files:**
- Modify: `scripts/profiling/benchmark_metric3d_trt.py`
- Generate: `reports/tensorrt_fp16/metric3d_trt_module_report.json`
- Generate: `reports/tensorrt_fp16/metric3d_trt_module_report.md`
- Generate: `reports/tensorrt_fp16/metric3d_trt_benchmark_tegrastats.log`

- [ ] **Step 1: Implement Phase A PyTorch collection**

Reuse the accepted exporter utilities to load config/predictor, apply 0.75
scale, and recreate the ten runtime inputs. For each frame, measure preprocess
wall time with synchronization at its boundaries, run/store PyTorch raw/final
references, and time the wrapper with five warmups and twenty CUDA-event
iterations. Save only CPU NumPy inputs/references and scalar timings.

- [ ] **Step 2: Enforce backend separation**

Delete the wrapper and `predictor.model_`, run `gc.collect()` and
`torch.cuda.empty_cache()`, then load `TensorRTEngine`. The runner must not be
constructed before PyTorch model release.

- [ ] **Step 3: Implement Phase B TensorRT measurement**

Upload each saved input, run one synchronized correctness inference, compute
raw/final parity using the accepted Metric3D metric, and time TensorRT with the
same five warmups and twenty CUDA-event iterations. Measure the unchanged
postprocess separately.

- [ ] **Step 4: Write reports and run once**

Write JSON/Markdown with every frame, aggregates, engine/checkpoint paths,
warmup/iteration counts, memory architecture, and decision. Run once with a
dedicated `tegrastats` log and preserve any first error before changing code.

### Task 6: Regression, resource evidence, and decision

**Files:**
- Modify: `reports/tensorrt_fp16/metric3d_trt_module_report.md`

- [ ] **Step 1: Run focused and preserved tests in isolated processes**

Run the two new Metric3D TRT test modules, Metric3D ONNX/depth-scale tests,
shared runner/metadata tests, DROID acceleration tests, and the 11 preserved
handoff regression modules. Require all to pass.

- [ ] **Step 2: Run final artifact/environment checks**

Verify Bash/Python syntax, plan/ONNX/checkpoint hashes, strict deserialization,
binding metadata, report coverage/decision, `pip check`, and no residual SLAM,
builder, benchmark, or `tegrastats` process.

- [ ] **Step 3: Complete build and benchmark evidence**

Add build/benchmark elapsed time, plan size/hash, build and benchmark peak
RAM/swap, layer/binding summary, all parity and latency statistics, and the
explicit online-integration candidate/no-go result.

- [ ] **Step 4: Preserve the dirty worktree**

Record HEAD and relevant files without staging, committing, resetting,
cleaning, or deleting generated diagnostics.
