# DROID Encoder TensorRT Module-Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `BasicEncoder` around a stable 4D core, export and validate static SmallCity fnet/cnet ONNX graphs, build separate FP16 engines, and issue independent module-level go/no-go decisions.

**Architecture:** Preserve the existing 5D `BasicEncoder.forward()` contract and move its convolutional body into `forward_core(x4d)`. A single encoder export/benchmark utility loads the DROID checkpoint exactly like `DBAFusion`, prepares real normalized inputs from admitted SmallCity frames, and treats fnet/cnet as independent static `[1,3,344,616]` modules. Online adapters are explicitly deferred until an encoder passes this plan's parity and performance gates.

**Tech Stack:** Python 3.8, PyTorch 2.1, CUDA 11.4, ONNX 1.14.1, ONNX Runtime CUDA, TensorRT 8.5.2, OpenCV, unittest, Jetson `tegrastats`.

---

## Scope Boundary

This plan produces offline module decisions only. It does not modify `MotionFilter`, `DBAFusion` backend selection, CLI flags, or online execution. A passing encoder receives a separate online-integration spec/plan; a failing encoder remains PyTorch.

No existing files are staged, committed, reset, cleaned, checked out, or deleted. Generated ONNX/plan artifacts remain uncommitted.

## File Map

- Modify `scripts/frontend/modules/extractor.py`: add stable `BasicEncoder.forward_core` and keep the 5D wrapper.
- Create `scripts/acceleration/export_droid_encoders_onnx.py`: checkpoint loading, admitted-frame input preparation, export, ORT CUDA parity, and validation reports.
- Create `tests/test_droid_encoder_core.py`: 4D/5D equivalence and finite-output tests.
- Create `tests/test_droid_encoder_export.py`: exporter helpers, checkpoint loading, input normalization, frame selection, report gates, and lazy dependency tests.
- Create `reports/tensorrt_fp16/build_droid_encoders_fp16.sh`: guarded independent fnet/cnet engine builds.
- Create `tests/test_droid_encoder_trt_build.py`: build-script safety and exact command tests.
- Create `scripts/profiling/benchmark_droid_encoders_trt.py`: independent real-input parity, latency, resource, and decision reports.
- Create `tests/test_droid_encoder_trt.py`: benchmark statistics and gate tests.
- Modify `docs/HANDOFF_TENSORRT_FP16_DROID_METRIC3D_2026-07-10.md`: record module results and authorize only passing branches.

### Task 1: Refactor BasicEncoder around a stable 4D core

**Files:**
- Modify: `scripts/frontend/modules/extractor.py`
- Create: `tests/test_droid_encoder_core.py`

- [ ] **Step 1: Write the original-order reference test**

Create a test-only reference function that executes the current body directly:

```python
def legacy_forward(encoder, x):
    batch, count, channels, height, width = x.shape
    flat = x.view(batch * count, channels, height, width)
    flat = encoder.relu1(encoder.norm1(encoder.conv1(flat)))
    flat = encoder.layer1(flat)
    flat = encoder.layer2(flat)
    flat = encoder.layer3(flat)
    flat = encoder.conv2(flat)
    _, output_channels, output_height, output_width = flat.shape
    return flat.view(
        batch, count, output_channels, output_height, output_width
    )
```

For fnet (`output_dim=128`, instance norm) and cnet (`output_dim=256`, no norm), compare the wished-for `encoder.forward_core(flat)` and `encoder(x5d)` against the reference for `(B,N)=(1,1),(1,4),(2,3)`. Require identical shapes, finite outputs, and `torch.testing.assert_close` with zero tolerance in eval mode.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_droid_encoder_core
```

Expected: fail because `BasicEncoder` has no `forward_core`.

- [ ] **Step 3: Implement the minimal core**

Move only the convolutional body:

```python
def forward_core(self, x):
    x = self.conv1(x)
    x = self.norm1(x)
    x = self.relu1(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    return self.conv2(x)

def forward(self, x):
    batch, count, channels, height, width = x.shape
    x = self.forward_core(x.view(batch * count, channels, height, width))
    _, output_channels, output_height, output_width = x.shape
    return x.view(batch, count, output_channels, output_height, output_width)
```

Do not alter initialization, normalization, layer ordering, dropout behavior, or multidim code outside the currently used path.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_droid_encoder_core
python -m unittest tests.test_motion_gate
python -m unittest tests.test_dbaf_motion_gate_packet
```

Expected: all tests pass.

### Task 2: Exporter utilities and exact checkpoint loading

**Files:**
- Create: `scripts/acceleration/export_droid_encoders_onnx.py`
- Create: `tests/test_droid_encoder_export.py`

- [ ] **Step 1: Write RED tests for encoder specification and state loading**

Define wished-for APIs:

```python
encoder_spec("fnet")
# {"output_channels": 128, "norm_fn": "instance", ...}

encoder_spec("cnet")
# {"output_channels": 256, "norm_fn": "none", ...}

load_droid_state_dict("ckpts/droid.pth", torch_module=torch)
load_encoder("fnet", checkpoint, device="cuda")
```

Tests use a fake checkpoint with `module.` prefixes and four-channel update heads. Require prefix removal and the same four head crops as `DBAFusion.load_weights()`. Strict `DroidNet.load_state_dict` must succeed before selecting an encoder.

- [ ] **Step 2: Verify RED**

Run the new test module. Expected: import failure because the exporter does not exist.

- [ ] **Step 3: Implement exact loading and specs**

Use one immutable spec table:

```python
ENCODER_SPECS = {
    "fnet": {"output_channels": 128, "norm_fn": "instance"},
    "cnet": {"output_channels": 256, "norm_fn": "none"},
}
```

Load checkpoint tensors on CPU, strip `module.`, crop the four update tensors, instantiate `DroidNet`, and call strict `load_state_dict`. Return only the selected encoder after validation. Keep PyTorch/ONNX Runtime imports inside functions where possible.

- [ ] **Step 4: Write RED tests for admitted-frame selection and normalization**

Require:

```python
admitted_frame_ids(runtime_events_csv, minimum=20)
select_validation_frame_ids(frame_ids, count=20)
prepare_encoder_input(image_path, image_size=(344, 616), device="cpu")
```

The event reader selects rows whose stage is `motion_filter_feature_encoder`, preserves order, removes duplicates, and rejects fewer than 20 frames. Selection must include the first and last admitted ID and distribute the remaining IDs monotonically.

The input helper must reproduce `HierarchicalDataset` plus `MotionFilter` exactly: OpenCV BGR resize to `616x344`, float32 CHW, divide by 255, then subtract/divide the existing ImageNet mean/std channel vectors without an extra RGB swap. Expected output is contiguous `[1,3,344,616]` float32.

- [ ] **Step 5: Verify RED then implement GREEN**

Run the focused tests before and after implementation. Expected final result: all helper tests pass without requiring CUDA, ONNX, or TensorRT imports during collection.

### Task 3: ONNX wrapper, parity metrics, and report gate

**Files:**
- Modify: `scripts/acceleration/export_droid_encoders_onnx.py`
- Modify: `tests/test_droid_encoder_export.py`

- [ ] **Step 1: Write RED wrapper tests**

Define:

```python
class EncoderCoreExportWrapper(torch.nn.Module):
    def __init__(self, encoder): ...
    def forward(self, image): ...
```

Require 4D input, call `forward_core`, and return `[1,C,43,77]`. Test both output-channel variants and finite outputs. The export path must preserve FP16 output under CUDA autocast while accepting normalized float32 input.

- [ ] **Step 2: Implement the wrapper minimally**

The wrapper contains only the selected encoder and its autocast/core call. It must not include normalization, five-dimensional reshape, fnet/cnet splitting, motion gating, or checkpoint logic.

- [ ] **Step 3: Write RED parity/report tests**

Implement generic `parity_metrics` and `build_report`. Acceptance requires exactly 20 selected admitted frames, CUDAExecutionProvider, exact shapes, finite outputs, minimum cosine `>=0.999`, and maximum mean relative error `<=0.01`.

The report records encoder name, ONNX path, provider, frame IDs, hashes, thresholds, per-frame metrics, aggregates, and `accepted`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_droid_encoder_export
python -m py_compile scripts/acceleration/export_droid_encoders_onnx.py
```

Expected: all tests and syntax checks pass.

### Task 4: Export and validate real fnet/cnet ONNX graphs

**Files:**
- Generated: `engines/tensorrt/droid/droid_fnet_b1_344x616.onnx`
- Generated: `engines/tensorrt/droid/droid_cnet_b1_344x616.onnx`
- Generated reports under: `reports/tensorrt_fp16/`

- [ ] **Step 1: Preflight**

Verify no `scripts/run.py`, exporter, `trtexec`, or `tegrastats` process remains; both ONNX paths are unused; checkpoint hash is unchanged; the PyTorch SmallCity-200 runtime events and all selected images exist.

- [ ] **Step 2: Export fnet with opset 17**

Run the exporter with encoder `fnet`, checkpoint `ckpts/droid.pth`, the accepted PyTorch SmallCity-200 runtime events, and dataset `data/smallcity_subset_200/small_city`. Capture stdout/status/`tegrastats` independently.

The exporter must run checker, shape inference, and ORT CUDA parity before writing an accepted report. If export fails, preserve the first unsupported operation and stop fnet before TensorRT.

- [ ] **Step 3: Export cnet independently**

Use separate output, log, status, and resource names. A fnet result must not authorize cnet. Preserve any failure without changing opset or graph until the exact failing node is identified.

- [ ] **Step 4: Inspect artifacts**

Record opset, graph input/output dtype and shape, node counts/types, ONNX hashes/sizes, selected IDs, provider assignment, minimum cosine, maximum MRE, and resource peaks.

### Task 5: Guarded TensorRT build script

**Files:**
- Create: `reports/tensorrt_fp16/build_droid_encoders_fp16.sh`
- Create: `tests/test_droid_encoder_trt_build.py`

- [ ] **Step 1: Write RED script tests**

Require independent `build_encoder fnet` and `build_encoder cnet` calls, exact ONNX/plan/log/status/resource paths, `--fp16`, 1024 MiB workspace, `--buildOnly`, verbose layer dump, process rejection, ONNX existence checks, plan non-overwrite checks, and trapped `tegrastats` cleanup.

The script must stop after a failed fnet build rather than claiming cnet success; already existing plan files are never overwritten.

- [ ] **Step 2: Verify RED**

Run the test module. Expected: missing-script failure.

- [ ] **Step 3: Implement and verify GREEN**

Use `/usr/src/tensorrt/bin/trtexec` and write separate status files. Run:

```bash
python -m unittest tests.test_droid_encoder_trt_build
bash -n reports/tensorrt_fp16/build_droid_encoders_fp16.sh
```

Expected: pass without starting a build.

### Task 6: Build engines and strict sidecars

**Files:**
- Generated plans and JSON sidecars under `engines/tensorrt/droid/`
- Generated build evidence under `reports/tensorrt_fp16/`

- [ ] **Step 1: Build only ONNX-accepted encoders**

Run fnet and cnet builds independently. If TensorRT rejects `InstanceNormalization` or another node, record the first parser error and producer stack before changing anything. Do not add plugins or change precision speculatively.

- [ ] **Step 2: Generate strict metadata**

Use `build_engine_metadata` and `write_engine_metadata` with exact engine, ONNX, checkpoint, command, empty static profiles, precision `fp16`, and deserialized bindings.

- [ ] **Step 3: Strict load verification**

Instantiate `TensorRTEngine` for each successful plan. Require one input named `image`, one output named `features`, exact static shapes, and expected dtypes. Preserve hash, size, build time, layer summary, and resource peaks.

### Task 7: Module benchmark decision logic

**Files:**
- Create: `scripts/profiling/benchmark_droid_encoders_trt.py`
- Create: `tests/test_droid_encoder_trt.py`

- [ ] **Step 1: Write RED statistics and coverage tests**

Require mean/median/p90 calculations and exactly 20 admitted frames. A report fails if frame coverage, shape, finite, cosine, MRE, actual TensorRT execution, or latency improvement is missing.

- [ ] **Step 2: Implement the report gate**

For each encoder, require:

```text
minimum cosine >= 0.999
maximum MRE <= 0.01
mean latency improvement >= 0.20
```

Return independent recommendations so one encoder can pass while the other fails.

- [ ] **Step 3: Write RED input/dtype tests**

Require contiguous normalized float32 input and contiguous float16 outputs with exact static shapes. Test rejection of a CPU/non-contiguous/wrong-shape binding before real TensorRT inference.

- [ ] **Step 4: Implement the two-phase benchmark**

For each encoder separately:

1. Load PyTorch encoder and 20 inputs.
2. Run 10 warmups and 50 CUDA-event measurements per input under autocast.
3. Store CPU references/timings.
4. Delete PyTorch model, run GC/empty cache/synchronize.
5. Load strict TensorRT engine.
6. Run identical inputs and measurements.
7. Record actual parity, latency, and memory-after-release.

Never keep the PyTorch encoder and its TensorRT engine resident together during timing.

- [ ] **Step 5: Verify GREEN**

Run test module and syntax check. Expected: all tests pass without eager TensorRT import.

### Task 8: Run real module benchmarks and issue branch decisions

**Files:**
- Generated: `reports/tensorrt_fp16/droid_fnet_trt_module_report.{json,md}`
- Generated: `reports/tensorrt_fp16/droid_cnet_trt_module_report.{json,md}`

- [ ] **Step 1: Benchmark fnet**

Run with separate log/status/`tegrastats`. Inspect every per-frame metric and aggregate. Mark fnet online integration candidate only if every module gate passes.

- [ ] **Step 2: Benchmark cnet independently**

Use distinct evidence. Record its theoretical SmallCity-200 maximum contribution (`0.232%` of current frame total) and avoid claiming an end-to-end benefit from module throughput alone.

- [ ] **Step 3: Run full isolated regression set**

Add the four new test modules to the current 25-module isolated loop. Run each in a new Python process, then run `pip check`, strict successful-engine loads, `git diff --check`, evidence existence checks, and residual-process checks.

- [ ] **Step 4: Update handoff**

Record exact results and one of four decisions: neither passes, fnet only, cnet only, or both module candidates. Create a new online-integration plan only for passing encoders. Do not modify default backend or start paired SLAM runs in this plan.
