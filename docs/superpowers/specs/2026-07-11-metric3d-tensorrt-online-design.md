# Metric3D TensorRT Online Backend Design

**Date:** 2026-07-11

**Status:** Approved design, awaiting written-spec review

## Objective

Add an explicit opt-in Metric3D TensorRT backend to the online SLAM path, report the backend that actually executes, preserve PyTorch as the default, and validate the integration with paired SmallCity-50 PyTorch and TensorRT runs.

This phase does not make TensorRT the default, start SmallCity-200 automatically, convert DROID encoders, change the frontend resolution, change OFA/mapping behavior, or build another engine.

## Existing Evidence

The accepted static engine is:

```text
engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan
```

Its input and output bindings are fixed:

```text
rgb   float32 [1, 3, 448, 784]
depth float32 [1, 1, 448, 784]
```

The offline ten-frame module gate passed with a `2.484x` model speedup, minimum raw cosine `0.999975135`, and maximum raw mean relative error `0.008364315`. The engine and its sidecar already pass strict deserialization and metadata validation.

## Architecture

`Metric_Model` remains the single interface used by `scripts/run.py`. It delegates inference to exactly one backend instance:

- `Metric3DPyTorchBackend` owns the existing `metric_modules.Metric` predictor and checkpoint.
- `Metric3DTensorRTBackend` owns a `TensorRTEngine` and only the lightweight Metric3D configuration required by canonical preprocessing and postprocessing. It must not construct the Metric3D neural network or load checkpoint weights.

Backend selection occurs during `Metric_Model` construction. A TensorRT request must not construct the PyTorch backend unless a non-strict failure requires fallback. The two full inference backends must never be resident together.

The TensorRT preprocessing helper may reuse Metric3D's existing configuration loader and preprocessing methods without calling `Metric3D.__init__`. This preserves canonical resize, normalization, padding, and depth-range configuration while avoiding model construction. The implementation must prove with tests that the heavyweight constructor is not called.

## Configuration and CLI

Configuration defaults remain PyTorch:

```yaml
inference:
  metric3d_backend: torch
  metric3d_engine: null
  tensorrt_strict: false
```

`scripts/run.py` adds these overrides:

```text
--metric-depth-backend {torch,tensorrt}
--metric-depth-engine PATH
--tensorrt-strict
```

The override layer creates `inference` when absent and changes only explicitly supplied values. Existing configurations without `inference` continue to use PyTorch.

TensorRT requires an engine path. The paired benchmark command must pass the engine explicitly and enable strict mode.

## Data Flow

The public call remains:

```python
Metric_Model.predict(img, profiler=None, frame_idx=None)
```

Both backends preserve the current behavior:

1. Convert a CHW PyTorch tensor to an HWC NumPy image when necessary and remember the original output device.
2. Apply Metric3D canonical preprocessing using the current intrinsic order `[fv, fu, cv, cu]`.
3. Execute either the PyTorch depth model or TensorRT engine.
4. Apply the existing depth clipping and CPU NumPy/OpenCV resize to the original image size.
5. Return a float32 PyTorch depth tensor on the original tensor device, or on `device.tracker` for NumPy input.

The static TensorRT path verifies that preprocessing produces exactly `[1, 3, 448, 784]`. Any other shape is an error rather than an implicit resize outside the accepted Metric3D preprocessing path.

The existing profiler boundaries remain unchanged:

```text
metric_depth_preprocess
metric_depth_model
metric_depth_postprocess
```

## Backend Reporting

Initialization logs one backend record:

```text
[metric_depth] requested_backend=tensorrt actual_backend=tensorrt strict=true
```

The runtime profile/report metadata records:

- requested backend;
- actual backend;
- strict-mode state;
- engine path when TensorRT is active;
- fallback reason when fallback occurs.

Reporting must use the backend that actually produced depth. A PyTorch fallback must never be counted or labeled as TensorRT.

## Failure and Fallback Semantics

Strict mode is mandatory for paired evaluation.

In strict mode, engine absence, sidecar mismatch, binding mismatch, static input-shape mismatch, deserialization failure, or inference failure raises the original error with Metric3D backend context. The run stops and no PyTorch model is constructed.

In non-strict mode:

1. Record the original TensorRT error once.
2. Drop the TensorRT runner and its cached output tensors.
3. Run garbage collection and clear the CUDA allocator cache when CUDA is available.
4. Construct `Metric3DPyTorchBackend` lazily.
5. Retry the current prediction once with PyTorch.
6. Log one transition record with `actual_backend=torch` and `fallback_from=tensorrt`.

Subsequent predictions use the single PyTorch backend without retrying TensorRT or repeating the fallback log.

## Testing Strategy

All implementation follows red-green TDD. Focused tests cover:

- existing configurations default to PyTorch and do not import TensorRT;
- CLI overrides populate the `inference` configuration correctly;
- a TensorRT request never calls the heavyweight Metric3D constructor;
- strict initialization and inference failures raise without fallback;
- non-strict failure releases TensorRT and constructs PyTorch exactly once;
- requested and actual backend reporting is correct and emitted once;
- TensorRT input shape, dtype, and contiguity are enforced;
- output shape, device, dtype, depth clipping, and three profiling stages match the current contract;
- runtime report metadata reflects fallback instead of the requested backend.

TensorRT imports remain lazy so CPU-only test discovery does not require NVIDIA libraries.

After focused tests, run the existing regression modules in isolated Python processes because some older tests install temporary `frontend.*` modules in `sys.modules`.

## Paired SmallCity-50 Evaluation

Before either run, capture process state and ensure no previous `scripts/run.py` or `tegrastats` process remains. Both runs use the same worktree and the same balanced-fast arguments, including `training_iters=10`, `metric_depth_scale=0.75`, mapping/pruning/pixel budgets, metric-depth scheduling, VPI C++ motion gate, runtime profiling, and evaluation export.

Use unique prefixes and output directories:

```text
jetson_smallcity_gt50_pytorch_metric3d_online_t10
jetson_smallcity_gt50_tensorrt_metric3d_online_t10
```

The TensorRT command differs only by adding:

```text
--metric-depth-backend tensorrt
--metric-depth-engine engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan
--tensorrt-strict
```

Each run preserves its command, stdout/stderr, `tegrastats`, exit status, runtime profile, evaluator output, and exported frame data. Existing results are never overwritten.

Run `evaluate_smallcity_run.py` separately for each output and then use `compare_tracking_runs.py` for the paired comparison.

## SmallCity-50 Acceptance Gate

The integration is accepted for a later SmallCity-200 decision run only when all conditions hold:

- neither run has a tracking failure or shutdown hang;
- the recorded actual backends are PyTorch and TensorRT respectively, with no fallback in the TensorRT run;
- TensorRT ATE increase is at most `max(15%, 0.10 m)` relative to PyTorch;
- mean PSNR drop is at most `0.5 dB`;
- mean SSIM drop is at most `0.02`;
- keyframe and exported sample count changes are at most `10%`;
- mean `metric_depth_model` latency improves by at least `20%`;
- mean `frame_total` improves by a measurable positive amount;
- TensorRT peak system RAM is no more than `500 MB` above PyTorch;
- no persistent idle or futex wait remains after shutdown.

If any gate fails, keep PyTorch as the only recommended online backend, preserve all evidence, and stop before SmallCity-200 or DROID encoder work.

## Deliverables

- Opt-in backend implementation and CLI/configuration wiring.
- Focused backend, fallback, reporting, profiling, and override tests.
- Paired SmallCity-50 run artifacts and resource logs.
- Evaluator and tracking-comparison reports.
- A handoff update stating the measured result and the go/no-go decision for SmallCity-200.

Generated `.onnx` and `.plan` files remain uncommitted. Existing user changes and anomalous untracked files remain untouched.
