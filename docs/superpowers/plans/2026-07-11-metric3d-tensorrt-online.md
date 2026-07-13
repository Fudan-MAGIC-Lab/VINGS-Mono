# Metric3D TensorRT Online Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strictly reportable, opt-in Metric3D TensorRT online backend and validate it against a fresh paired SmallCity-50 PyTorch control.

**Architecture:** Keep `Metric_Model` as the public facade and delegate to one of two backend classes. The TensorRT backend constructs only Metric3D preprocessing configuration plus the existing `TensorRTEngine`; strict mode never falls back, while non-strict mode releases TensorRT and lazily constructs PyTorch once. Runtime-profiler metadata records the backend that actually produced depth.

**Tech Stack:** Python 3.8, PyTorch 2.1, TensorRT 8.5.2, OpenCV, unittest, YAML, Jetson `tegrastats`.

---

## File Map

- Create `scripts/metric/metric3d_backends.py`: isolated PyTorch and TensorRT backend implementations with no CLI or scheduling knowledge.
- Modify `scripts/metric/metric_model.py`: backend-selection facade, strict/fallback transition, existing preprocessing/model/postprocessing timing boundaries, and backend-status reporting.
- Modify `scripts/profiling/runtime_profiler.py`: small metadata API and `runtime_profile_metadata.json` output.
- Modify `scripts/run.py`: CLI overrides, default inference configuration, and initial backend metadata registration.
- Modify `configs/hierarchical/smallcity.yaml`: explicit PyTorch defaults only.
- Create `tests/test_metric3d_trt_backend.py`: backend construction, strict/fallback, output-contract, logging, and profiling tests.
- Modify `tests/test_metric_model_depth_scale.py`: preserve the existing PyTorch facade contract after delegation.
- Modify `tests/test_runtime_profiler.py`: metadata report behavior.
- Modify `tests/test_jetson_runtime_overrides.py`: CLI/config override behavior.
- Create `reports/tensorrt_fp16/run_metric3d_smallcity50_pair.sh`: guarded, reproducible paired-run driver that preserves separate logs and resource samples.
- Update `docs/HANDOFF_TENSORRT_FP16_DROID_METRIC3D_2026-07-10.md`: measured result and SmallCity-200 go/no-go decision.

### Task 1: Runtime profiler backend metadata

**Files:**
- Modify: `scripts/profiling/runtime_profiler.py`
- Modify: `tests/test_runtime_profiler.py`

- [ ] **Step 1: Write the failing metadata test**

Add a test that creates an enabled profiler, calls:

```python
profiler.set_metadata(
    "metric_depth",
    {
        "requested_backend": "tensorrt",
        "actual_backend": "torch",
        "strict": False,
        "fallback_reason": "engine load failed",
    },
)
profiler.write_reports()
```

Assert that `runtime_profile_metadata.json` contains the same nested object and that updating `metric_depth` replaces the prior value rather than creating duplicate records.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_runtime_profiler.RuntimeProfilerTests.test_write_reports_includes_mutable_metadata
```

Expected: fail because `RuntimeProfiler` has no `set_metadata` method.

- [ ] **Step 3: Implement the minimal metadata API**

In `RuntimeProfiler.__init__`, add `self.metadata = {}`. Add:

```python
def set_metadata(self, key, value):
    if not self.enabled:
        return
    self.metadata[str(key)] = value
```

In `write_reports`, write stable JSON after creating the output directory:

```python
with (self.output_dir / "runtime_profile_metadata.json").open("w") as handle:
    json.dump(self.metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
```

Import `json` at module scope. Do not change timing or CUDA synchronization behavior.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m unittest tests.test_runtime_profiler
```

Expected: all runtime-profiler tests pass.

### Task 2: Backend boundary and lightweight TensorRT preprocessing

**Files:**
- Create: `scripts/metric/metric3d_backends.py`
- Create: `tests/test_metric3d_trt_backend.py`

- [ ] **Step 1: Write failing construction tests**

Use injected `metric_class` and `runner_factory` dependencies so the tests do not import TensorRT. Cover these wished-for APIs:

```python
torch_backend = Metric3DPyTorchBackend(
    checkpoint="ckpts/metric_depth_vit_small_800k.pth",
    depth_scale=0.75,
    metric_class=FakeMetric,
)

trt_backend = Metric3DTensorRTBackend(
    engine_path="engine.plan",
    metadata_path="engine.json",
    checkpoint="ckpts/metric_depth_vit_small_800k.pth",
    depth_scale=0.75,
    metric_class=FakeMetric,
    runner_factory=FakeRunner,
)
```

Assert that the PyTorch backend calls `FakeMetric.__init__` once, while the TensorRT backend never calls it and creates configuration through `FakeMetric.__new__` plus `_load_config_`. Assert that scale `0.75` yields crop and ViT size `(448, 784)`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_metric3d_trt_backend.Metric3DBackendConstructionTests
```

Expected: import failure because `metric.metric3d_backends` does not exist.

- [ ] **Step 3: Implement the two backend classes**

Create focused classes with these responsibilities:

```python
class Metric3DPyTorchBackend:
    name = "torch"

    def preprocess(self, image, intrinsic):
        return self.predictor.preprocess(image, intrinsic)

    def infer(self, prepared):
        return self.predictor.forward_depth(*prepared)

    def postprocess(self, prediction, d_max, d_min):
        return self.predictor.postprocess(prediction, d_max=d_max, d_min=d_min)

    def close(self):
        self.predictor = None
```

```python
class Metric3DTensorRTBackend:
    name = "tensorrt"
    expected_shape = (1, 3, 448, 784)

    def infer(self, prepared):
        rgb = prepared[0].to(dtype=torch.float32).contiguous()
        if tuple(rgb.shape) != self.expected_shape:
            raise ValueError(
                f"Metric3D TensorRT input must have shape {self.expected_shape}, "
                f"got {tuple(rgb.shape)}"
            )
        return self.runner.infer({"rgb": rgb})["depth"]

    def close(self):
        if self.runner is not None:
            self.runner.output_cache.clear()
        self.runner = None
        self.predictor = None
```

The TensorRT constructor must lazily import `acceleration.trt_engine.TensorRTEngine` only when no test factory is supplied. Reuse one shared scale helper so PyTorch and TensorRT apply the same crop-size mutation. Reject TensorRT depth scales whose resulting shape is not `(448, 784)`.

- [ ] **Step 4: Add RED tests for inference contract**

Add tests requiring:

- `[1,3,448,784]` float32 contiguous input reaches `FakeRunner`;
- an incorrect shape raises before `FakeRunner.infer`;
- output is selected only from binding name `depth`;
- `close()` clears cached outputs and drops the runner.

Run the new tests and confirm they fail for missing validation or cleanup before implementing each behavior.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
python -m unittest tests.test_metric3d_trt_backend
```

Expected: all backend-boundary tests pass without importing NVIDIA TensorRT.

### Task 3: Metric_Model facade, strict mode, and lazy fallback

**Files:**
- Modify: `scripts/metric/metric_model.py`
- Modify: `tests/test_metric3d_trt_backend.py`
- Modify: `tests/test_metric_model_depth_scale.py`

- [ ] **Step 1: Write RED tests for backend selection**

Inject backend factories into a test-loaded `Metric_Model` and assert:

```python
cfg["inference"] = {
    "metric3d_backend": "torch",
    "metric3d_engine": None,
    "tensorrt_strict": False,
}
```

constructs only the PyTorch backend, while `metric3d_backend=tensorrt` constructs only TensorRT. Verify the default for a missing `inference` block is PyTorch.

- [ ] **Step 2: Verify RED**

Run the new selection tests. Expected: fail because `Metric_Model` always constructs `Metric` directly.

- [ ] **Step 3: Implement the facade selection**

Replace direct predictor ownership with:

```python
self.requested_backend = inference_cfg.get("metric3d_backend", "torch")
self.strict = bool(inference_cfg.get("tensorrt_strict", False))
self.engine_path = inference_cfg.get("metric3d_engine")
self.actual_backend = None
self.fallback_reason = None
self._fallback_logged = False
self.backend = self._create_requested_backend()
self.actual_backend = self.backend.name
self._log_backend_status()
```

Validate the requested backend against `{"torch", "tensorrt"}`. A TensorRT request without an engine path raises in strict mode and enters the same non-strict fallback path otherwise.

- [ ] **Step 4: Write RED strict/fallback tests**

Cover four independent cases:

1. strict TensorRT construction failure raises and never constructs PyTorch;
2. strict TensorRT inference failure raises and never constructs PyTorch;
3. non-strict construction failure creates PyTorch once;
4. non-strict inference failure closes TensorRT, clears CUDA cache through an injected cleanup callback, creates PyTorch once, retries the current prediction, and uses PyTorch on later predictions without retrying TensorRT.

Capture stdout and assert the initial status and fallback transition are each printed once.

- [ ] **Step 5: Implement the minimal fallback transition**

Add one method that is shared by initialization and inference failures:

```python
def _fallback_to_pytorch(self, exc):
    if self.strict:
        raise RuntimeError("Metric3D TensorRT backend failed in strict mode") from exc
    if self.backend is not None:
        self.backend.close()
    self.backend = None
    self._cleanup_cuda()
    self.fallback_reason = f"{type(exc).__name__}: {exc}"
    self.backend = self._make_pytorch_backend()
    self.actual_backend = "torch"
    self._log_fallback_once()
```

Use exception chaining so the original failure remains inspectable. Do not retry TensorRT.

- [ ] **Step 6: Preserve predict behavior and profiler stages**

Keep image conversion and the three existing profiler contexts in `Metric_Model.predict`. Replace predictor calls only:

```python
prepared = self.backend.preprocess(img_numpy, self.intr)
prediction = self._infer_with_fallback(img_numpy, prepared)
depth = self.backend.postprocess(prediction, d_max=self.d_max, d_min=0)
```

`_infer_with_fallback` receives both the original NumPy image and the prepared TensorRT tuple. If fallback occurs after TensorRT preprocessing, it calls `self.backend.preprocess(img_numpy, self.intr)` again after installing PyTorch, then retries with that new tuple. This makes the retry entirely owned by the active backend. Preserve original output size, float32 dtype, and output device.

- [ ] **Step 7: Add backend-status API and dynamic profiler update**

Expose only serializable values:

```python
def backend_status(self):
    return {
        "requested_backend": self.requested_backend,
        "actual_backend": self.actual_backend,
        "strict": self.strict,
        "engine_path": str(self.engine_path) if self.engine_path else None,
        "fallback_reason": self.fallback_reason,
    }
```

At the end of `predict`, call `profiler.set_metadata("metric_depth", self.backend_status())` when the method exists. This updates the report after a runtime fallback.

- [ ] **Step 8: Verify GREEN**

Run each module separately:

```bash
python -m unittest tests.test_metric3d_trt_backend
python -m unittest tests.test_metric_model_depth_scale
```

Expected: all tests pass and the existing PyTorch output/profiling contract remains unchanged.

### Task 4: CLI, configuration defaults, and initial backend reporting

**Files:**
- Modify: `scripts/run.py`
- Modify: `configs/hierarchical/smallcity.yaml`
- Modify: `tests/test_jetson_runtime_overrides.py`

- [ ] **Step 1: Write RED CLI override tests**

Load `scripts/run.py` with:

```text
--metric-depth-backend tensorrt
--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan
--tensorrt-strict
```

Assert that `apply_overrides` produces:

```python
{
    "metric3d_backend": "tensorrt",
    "metric3d_engine": "engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan",
    "tensorrt_strict": True,
}
```

Also assert that no flags leave existing values untouched.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_jetson_runtime_overrides.JetsonRuntimeOverrideTests.test_run_py_can_select_metric_depth_tensorrt
```

Expected: argparse rejects the new flags.

- [ ] **Step 3: Implement CLI and overrides**

Add the three approved arguments, call `cfg.setdefault("inference", {})`, and apply only non-`None` backend/engine values plus the true strict flag. Do not add a CLI flag that implicitly disables strict mode.

- [ ] **Step 4: Register initial runtime metadata**

After constructing `RuntimeProfiler`, add:

```python
if hasattr(self, "metric_predictor"):
    self.profiler.set_metadata(
        "metric_depth", self.metric_predictor.backend_status()
    )
```

Do not reorder tracker, mapper, or dataset construction in this task.

- [ ] **Step 5: Add explicit YAML defaults**

Add only:

```yaml
inference:
  metric3d_backend: torch
  metric3d_engine: null
  tensorrt_strict: false
```

Do not add DROID controls in this phase.

- [ ] **Step 6: Verify GREEN**

Run separately:

```bash
python -m unittest tests.test_jetson_runtime_overrides
python -m unittest tests.test_runtime_profiler
```

Expected: all tests pass.

### Task 5: Jetson real-engine online backend smoke

**Files:**
- Modify: `tests/test_metric3d_trt_backend.py`

- [ ] **Step 1: Add a guarded Jetson integration test**

Skip unless CUDA, TensorRT, the plan, metadata, checkpoint, and SmallCity frame 0 are present. Construct `Metric_Model` in strict TensorRT mode, predict frame 0, and assert:

- `actual_backend == "tensorrt"`;
- no PyTorch Metric3D model was constructed;
- output shape is `(344, 616)`;
- output dtype is float32 and all values are finite;
- backend status has no fallback reason.

- [ ] **Step 2: Verify the test catches a wrong engine path**

Run the strict test once with a temporary missing plan path and confirm it fails before prediction without constructing PyTorch. Restore the accepted path afterward; do not alter or delete engine artifacts.

- [ ] **Step 3: Run the accepted engine smoke**

Run:

```bash
python -m unittest tests.test_metric3d_trt_backend.Metric3DRealTensorRTIntegrationTest
```

Expected: pass on Jetson with the actual backend reported as TensorRT.

### Task 6: Full isolated regression verification

**Files:**
- No production changes

- [ ] **Step 1: Run focused syntax checks**

```bash
python -m py_compile \
  scripts/metric/metric3d_backends.py \
  scripts/metric/metric_model.py \
  scripts/profiling/runtime_profiler.py \
  scripts/run.py
```

- [ ] **Step 2: Run every preserved test module in a separate Python process**

Use the 22-module isolated loop from the completed module-gate report, then add `tests.test_metric3d_trt_backend`. Record the module and test counts. Do not run all modules in one interpreter because existing temporary `frontend.*` test doubles leak across modules.

- [ ] **Step 3: Validate environment and patch format**

Run `python -m pip check`, `git diff --check` on touched tracked files, strict engine deserialization, and process checks for `scripts/run.py`, `trtexec`, and `tegrastats`.

Expected: no broken requirements, no patch-format errors, valid bindings, and no residual processes.

### Task 7: Guarded paired SmallCity-50 driver

**Files:**
- Create: `reports/tensorrt_fp16/run_metric3d_smallcity50_pair.sh`
- Create: `tests/test_metric3d_smallcity50_pair_script.py`

- [ ] **Step 1: Write RED script-content tests**

Require the script to:

- use `set -u` and refuse to start if either unique output prefix already exists;
- capture git status, submodule status, versions, `pip freeze`, and process snapshot;
- reject residual `scripts/run.py` or `tegrastats` before each run;
- use dataset `data/smallcity_subset_50/small_city`;
- use `training_iters=10`, `metric_depth_scale=0.75`, and the accepted balanced-fast flags;
- add only the three TensorRT flags to the candidate command;
- save separate stdout, status, elapsed time, and `tegrastats` files;
- stop `tegrastats` through a shell trap;
- never remove or overwrite prior results.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m unittest tests.test_metric3d_smallcity50_pair_script
```

Expected: fail because the driver does not exist.

- [ ] **Step 3: Implement the guarded driver**

Use these immutable prefixes:

```text
jetson_smallcity_gt50_pytorch_metric3d_online_t10
jetson_smallcity_gt50_tensorrt_metric3d_online_t10
```

The common command must match the approved balanced-fast SmallCity command. The candidate command appends exactly:

```bash
--metric-depth-backend tensorrt \
--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan \
--tensorrt-strict
```

- [ ] **Step 4: Verify GREEN and shell syntax**

```bash
python -m unittest tests.test_metric3d_smallcity50_pair_script
bash -n reports/tensorrt_fp16/run_metric3d_smallcity50_pair.sh
```

Expected: pass without starting a run.

### Task 8: Execute and evaluate the paired runs

**Files:**
- Generated evidence under `reports/tensorrt_fp16/`
- Generated run directories under a unique `output/smallcity_metric3d_online_20260711/`

- [ ] **Step 1: Run the preserved preflight**

Confirm the dataset has 50 color frames, the engine hash is `fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9`, both prefixes are unused, and no target process remains.

- [ ] **Step 2: Execute PyTorch control and TensorRT candidate**

Run the guarded driver. Monitor status and resource logs at intervals shorter than 60 seconds. If the control fails, stop before TensorRT. If strict TensorRT fails or reports fallback, preserve evidence and stop without rerunning under a looser policy.

- [ ] **Step 3: Evaluate each run**

For each exact run directory, run:

```bash
python scripts/profiling/evaluate_smallcity_run.py \
  --run-dir RUN_DIR \
  --dataset-root data/smallcity_subset_50/small_city \
  --output RUN_DIR/smallcity_eval_metrics.md
```

- [ ] **Step 4: Compare tracking and profiles**

Run `compare_tracking_runs.py` with the PyTorch run as baseline and TensorRT as candidate. Calculate peak RAM from each `tegrastats` file and explicitly compare `metric_depth_model` and `frame_total` mean latency.

- [ ] **Step 5: Apply every acceptance gate**

Produce a table containing tracking status, actual backend, ATE, PSNR, SSIM, keyframe count, export count, Metric3D latency, frame-total latency, peak RAM, and shutdown status. Mark SmallCity-200 `go` only if every threshold in the approved spec passes.

### Task 9: Final evidence and handoff

**Files:**
- Modify: `docs/HANDOFF_TENSORRT_FP16_DROID_METRIC3D_2026-07-10.md`
- Create: `reports/tensorrt_fp16/metric3d_smallcity50_online_decision.md`

- [ ] **Step 1: Write the paired decision report**

Record exact commands, run paths, hashes, backend metadata, evaluation numbers, runtime deltas, resource peaks, and any failure. Distinguish module speedup from end-to-end speedup.

- [ ] **Step 2: Update the handoff**

State what was implemented, what tests ran, whether SmallCity-50 passed, and whether SmallCity-200 is authorized. Do not claim DROID readiness.

- [ ] **Step 3: Fresh final verification**

Re-run focused tests, the full isolated regression loop, `pip check`, strict engine loading, report-existence checks, and residual-process checks. Read the outputs before making any completion claim.

- [ ] **Step 4: Preserve dirty-worktree policy**

Do not reset, clean, checkout, stage, commit, or delete existing or anomalous untracked files. Generated `.onnx` and `.plan` artifacts remain uncommitted.
