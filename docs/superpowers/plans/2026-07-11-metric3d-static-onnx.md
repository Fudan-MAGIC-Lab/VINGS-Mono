# Metric3D Static ONNX Export and Real-Input Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the complete Metric3D v2-S depth model at `1x3x448x784` and validate raw and final depth parity on ten evenly spaced real SmallCity frames with ONNX Runtime CUDA.

**Architecture:** One focused exporter module owns the export wrapper, fixed-scale model preparation, runtime-equivalent SmallCity image loading, parity metrics, provider enforcement, and reports. CPU-safe unit tests use fake models and arrays; the real Jetson command loads the checkpoint once, exports once, then validates all ten frames in the same process.

**Tech Stack:** Python 3.8, PyTorch 2.1 Jetson build, ONNX 1.14.1, ONNX Runtime GPU 1.15.1, OpenCV, NumPy, unittest.

**Worktree constraint:** Use the existing dirty Jetson worktree. Do not create a branch/worktree and do not stage, commit, reset, clean, or overwrite unrelated files.

---

### Task 1: CPU-safe sample, scale, parity, and provider contracts

**Files:**
- Create: `tests/test_metric3d_onnx_export.py`
- Create: `scripts/acceleration/export_metric3d_onnx.py`

- [ ] **Step 1: Write failing utility tests**

The tests import only the exporter and require:

```python
def test_sample_frames_are_evenly_spaced(self):
    self.assertEqual(sample_frame_ids(), (0, 5, 10, 15, 20, 25, 30, 35, 40, 45))

def test_scaled_forward_size_is_static_balanced_fast_shape(self):
    self.assertEqual(scaled_forward_size((616, 1064), 0.75), (448, 784))

def test_parity_metrics_use_positive_finite_expected_mask(self):
    expected = np.array([0.0, 1.0, 2.0, np.nan], dtype=np.float32)
    actual = np.array([9.0, 1.001, 1.998, 7.0], dtype=np.float32)
    result = parity_metrics(expected, actual)
    self.assertEqual(result["valid_count"], 2)
    self.assertTrue(result["finite"])
    self.assertGreaterEqual(result["cosine"], 0.999)
    self.assertLessEqual(result["mean_relative_error"], 0.02)

def test_parity_metrics_reject_shape_mismatch_and_empty_mask(self):
    with self.assertRaisesRegex(ValueError, "shape"):
        parity_metrics(np.zeros((2, 2)), np.zeros((4,)))
    with self.assertRaisesRegex(ValueError, "valid"):
        parity_metrics(np.zeros((2, 2)), np.zeros((2, 2)))

def test_cuda_provider_is_required(self):
    require_provider(["CPUExecutionProvider"], "CUDAExecutionProvider")
```

The provider test must expect a `RuntimeError` mentioning CUDA.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_metric3d_onnx_export
```

Expected: import failure because the exporter does not exist.

- [ ] **Step 3: Implement minimal CPU-safe utilities**

Implement these stable functions without importing Metric3D or ONNX Runtime at module import time:

```python
FRAME_IDS = tuple(range(0, 50, 5))

def sample_frame_ids():
    return FRAME_IDS

def scaled_forward_size(size, scale, multiple=28):
    h, w = map(int, size)
    return (
        max(multiple, int(round(h * scale / multiple)) * multiple),
        max(multiple, int(round(w * scale / multiple)) * multiple),
    )

def require_provider(providers, required="CUDAExecutionProvider"):
    if required not in providers:
        raise RuntimeError(f"required ONNX Runtime provider is unavailable: {required}")

def parity_metrics(expected, actual, relative_floor=1e-3):
    expected = np.asarray(expected)
    actual = np.asarray(actual)
    if expected.shape != actual.shape:
        raise ValueError(f"output shape mismatch: {expected.shape} != {actual.shape}")
    mask = np.isfinite(expected) & (expected > 0)
    if not mask.any():
        raise ValueError("expected depth has no valid positive values")
    expected_valid = expected[mask].astype(np.float64, copy=False)
    actual_valid = actual[mask].astype(np.float64, copy=False)
    finite = bool(np.isfinite(actual_valid).all())
    denominator = max(float(np.linalg.norm(expected_valid) * np.linalg.norm(actual_valid)), 1e-12)
    return {
        "shape": list(actual.shape),
        "valid_count": int(mask.sum()),
        "finite": finite,
        "cosine": float(np.dot(expected_valid, actual_valid) / denominator),
        "mean_relative_error": float(np.mean(
            np.abs(expected_valid - actual_valid)
            / np.maximum(np.abs(expected_valid), relative_floor)
        )),
    }
```

- [ ] **Step 4: Verify GREEN**

Run the focused unittest and require PASS.

### Task 2: Export wrapper and report decision logic

**Files:**
- Modify: `scripts/acceleration/export_metric3d_onnx.py`
- Modify: `tests/test_metric3d_onnx_export.py`

- [ ] **Step 1: Write failing wrapper/report tests**

Use a fake depth model whose forward returns `{"prediction": tensor, "confidence": tensor}` and require `Metric3DDepthExportWrapper` to return only `prediction`. Add report cases for all ten frame IDs with `raw` and `final` metrics, then require:

```python
report = build_report(cases, onnx_path="model.onnx", provider="CUDAExecutionProvider")
self.assertEqual(report["frame_ids"], list(sample_frame_ids()))
self.assertTrue(report["accepted"])
self.assertEqual(report["thresholds"]["minimum_cosine"], 0.999)
self.assertEqual(report["thresholds"]["maximum_mean_relative_error"], 0.02)
```

Change one final MRE to `0.0201` and require `accepted` to become false.

- [ ] **Step 2: Verify RED**

Run the focused unittest and require failure for missing wrapper/report APIs.

- [ ] **Step 3: Implement wrapper and strict report builder**

Implement:

```python
class Metric3DDepthExportWrapper(torch.nn.Module):
    def __init__(self, depth_model):
        super().__init__()
        self.depth_model = depth_model

    def forward(self, rgb):
        return self.depth_model(input=rgb)["prediction"]
```

`build_report()` must require exactly the ten configured frame IDs and both
`raw`/`final` results per frame. Acceptance requires shape records, finite true,
cosine at least 0.999, and MRE at most 0.02 for every boundary and frame. It
also records global minimum cosine and maximum MRE.

- [ ] **Step 4: Verify GREEN**

Run the focused unittest and require PASS.

### Task 3: Real predictor preparation and static export

**Files:**
- Modify: `scripts/acceleration/export_metric3d_onnx.py`
- Modify: `tests/test_metric3d_onnx_export.py`

- [ ] **Step 1: Write failing runtime-input tests**

Create a temporary BGR image with distinct channel values. Require
`load_runtime_rgb(path, image_size=(344, 616))` to return a `3x344x616` float32
RGB array/tensor whose channels are reversed from OpenCV BGR. Test
`smallcity_intrinsics(config)` returns `[fv, fu, cv, cu]`.

- [ ] **Step 2: Verify RED**

Run the focused unittest and require failure for the missing APIs.

- [ ] **Step 3: Implement lazy real-model loading and export**

The exporter must:

1. add only the repository `submodules` path before lazily importing
   `metric_modules.Metric`;
2. load v2-S from the requested checkpoint;
3. scale `cfg_.data_basic.crop_size` and `vit_size` to `(448, 784)`;
4. expose `load_runtime_rgb()` that performs `cv2.imread`, resize to the config
   frontend size, BGR-to-RGB conversion, CHW conversion, and float32 output;
5. expose `smallcity_intrinsics()` returning `[fv, fu, cv, cu]`;
6. preprocess frame 00000 through `predictor.preprocess()` and assert the model
   tensor is exactly `1x3x448x784`;
7. run the wrapper once under `torch.no_grad()` to create lazy buffers;
8. call `torch.onnx.export` with opset 17, `input_names=["rgb"]`,
   `output_names=["depth"]`, static axes, and constant folding disabled because
   the Jetson PyTorch 2.1 fold pass mixes CUDA tensors with CPU indices;
9. run ONNX checker and shape inference, then save the inferred model.

The output path is created only after confirming it does not already exist.

- [ ] **Step 4: Verify CPU tests and Python syntax**

Run the focused unittest and `python -m py_compile` without loading the real
checkpoint.

### Task 4: Ten-frame ONNX Runtime CUDA validation

**Files:**
- Modify: `scripts/acceleration/export_metric3d_onnx.py`
- Generate: `engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx`
- Generate: `reports/tensorrt_fp16/metric3d_onnx_validation.json`
- Generate: `reports/tensorrt_fp16/metric3d_onnx_validation.md`

- [ ] **Step 1: Implement the one-process validation loop**

For every frame ID, load the runtime-equivalent RGB image, call the existing
`predictor.preprocess()`, compare wrapper output with
`session.run(["depth"], {"rgb": rgb_input.cpu().numpy()})`, then apply the same
`predictor.postprocess(d_max=300, d_min=0)` and `cv2.resize` to both outputs.
Record `raw` and `final` parity metrics.

Before session construction, require CUDA in `ort.get_available_providers()`.
After construction, require CUDA in `session.get_providers()`.

- [ ] **Step 2: Implement JSON and Markdown output**

The JSON contains artifact path/SHA256, provider lists, fixed input/output
metadata, checkpoint/dataset/config paths, frame cases, thresholds, aggregates,
and `accepted`. Markdown shows a row for each frame and boundary plus the
explicit go/no-go decision.

- [ ] **Step 3: Run the real export and validation once**

With no SLAM or `tegrastats` process active, run the exporter in
`vings_jetson`, capture stdout/stderr to
`reports/tensorrt_fp16/metric3d_onnx_export_validation.log`, and collect a
dedicated `tegrastats` log. Do not retry with code changes until the first raw
failure has been inspected.

- [ ] **Step 4: Apply the acceptance gate**

Proceed only if ONNX checker/shape inference pass, CUDA is enabled, all ten
frames exist, and `accepted=true`. Otherwise retain logs and stop before any
TensorRT build.

### Task 5: Regression, evidence, and handoff decision

**Files:**
- Modify: `reports/tensorrt_fp16/metric3d_onnx_validation.md`

- [ ] **Step 1: Run focused and preserved tests in isolated processes**

Run `tests.test_metric3d_onnx_export`, `tests.test_metric_model_depth_scale`,
`tests.test_runtime_profiler`, the DROID update/export/runner tests, and the 11
preserved handoff regression modules. Require all to pass.

- [ ] **Step 2: Run final artifact checks**

Verify Python compilation, ONNX checker, input/output names and static shapes,
artifact SHA256, `pip check`, report decision, and no residual Python,
ONNX Runtime, or `tegrastats` process.

- [ ] **Step 3: Complete the report**

Add export/validation elapsed time, peak RAM/swap, model size/hash, provider
evidence, global and per-frame parity, and the explicit decision about whether
the next TensorRT build phase is authorized.

- [ ] **Step 4: Inspect the worktree without mutation**

Record HEAD and relevant modified/untracked files. Do not stage, commit, reset,
clean, or delete any existing user content.
