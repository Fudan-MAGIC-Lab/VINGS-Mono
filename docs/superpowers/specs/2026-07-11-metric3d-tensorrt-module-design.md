# Metric3D TensorRT FP16 Engine and Module Benchmark Design

Date: 2026-07-11

## Goal

Build a reproducible TensorRT FP16 engine from the accepted static Metric3D
ONNX graph, then measure module parity and latency on the same ten real
SmallCity frames. This phase ends with an integrate/do-not-integrate module
decision and does not change the online Metric3D backend.

## Scope

- Source ONNX:
  `engines/tensorrt/metric3d/metric3d_v2s_448x784_opset16.onnx`.
- Static input/output: `rgb[1,3,448,784] -> depth[1,1,448,784]`.
- TensorRT 8.5.2.2 FP16, workspace 1024 MiB, built on the current Jetson.
- Use the existing checkpoint, preprocessing, postprocessing, ten frame IDs,
  and parity formulas from the accepted ONNX phase.
- Do not modify `Metric_Model.predict()`, runtime backend configuration,
  frontend resolution, DROID, OFA, mapping, or Gaussian rendering.
- Do not run SmallCity SLAM in this phase.
- Do not stage, commit, reset, clean, or remove unrelated worktree content.

## Build Artifacts

- `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan`
- `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.json`
- dedicated build, status, controller, and `tegrastats` logs under
  `reports/tensorrt_fp16/`

The build uses `trtexec --fp16 --memPoolSize=workspace:1024MiB --buildOnly
--verbose --dumpLayerInfo`. It refuses to overwrite an existing plan and
refuses to start while SLAM, `trtexec`, or `tegrastats` is active.

The initially accepted opset-17 ONNX is retained unchanged as parity evidence,
but TensorRT 8.5.2 cannot parse its 43 `LayerNormalization` nodes and has no
matching plugin. PyTorch's opset-16 symbolic decomposes LayerNorm into older
primitive ONNX operations. A separate opset-16 export must pass the identical
ten-frame raw/final ORT CUDA parity gate before it can become the engine source.

The JSON sidecar records engine/ONNX/checkpoint hashes, TensorRT/CUDA/device
versions, precision, exact build command, static binding metadata, and an empty
profile object because this engine has no dynamic dimensions.

## Build Acceptance

The engine is accepted for benchmarking only when:

- `trtexec` exits zero and prints its PASSED marker;
- the plan is nonempty;
- the parser/build log contains no unsupported-node or fatal error;
- bindings are exactly float32 `rgb[1,3,448,784]` and float32
  `depth[1,1,448,784]`;
- sidecar hashes match current artifacts;
- strict `TensorRTEngine` deserialization succeeds;
- the build's `tegrastats` process is cleaned up.

If parsing or building fails, preserve the first raw error and stop. Do not
change opset, precision, ONNX graph, or add plugins before root-cause analysis.

## Benchmark Memory Architecture

The Jetson must not hold the complete PyTorch Metric3D model and TensorRT
engine for the whole benchmark.

Phase A loads PyTorch once and, for frames `0,5,...,45`:

- recreates the runtime RGB and existing Metric3D preprocessing;
- stores the preprocessed float32 CUDA input as a CPU NumPy array;
- stores raw and final PyTorch depth references on CPU;
- measures preprocess, PyTorch neural forward, and postprocess timing.

After Phase A, the wrapper and `predictor.model_` are deleted and CUDA cache is
released. The lightweight predictor object remains only for its configuration
and postprocess method.

Phase B loads only the TensorRT engine, uploads each saved input, runs direct
CUDA pointer binding through `TensorRTEngine`, stores raw/final parity metrics,
and measures TensorRT neural latency. This preserves identical inputs without
long-lived dual model residency.

## Timing

For every real frame:

- five warmup iterations;
- twenty measured iterations;
- PyTorch and TensorRT neural forward timed with CUDA events;
- synchronization occurs only around benchmark measurements and correctness
  transfers;
- report per-frame latency plus mean, median, p90, speedup, and improvement.

The primary 20% performance gate applies only to neural forward, the exact
engine replacement boundary. The report also records preprocess and
postprocess timing and an estimated full pipeline total, but those values do
not change the module decision.

## Correctness and Decision Gates

For raw and final depth on all ten frames:

- shapes match exactly;
- all valid values are finite;
- cosine similarity is at least 0.999;
- mean relative error over positive finite PyTorch depth is at most 0.02.

The engine becomes an online-backend integration candidate only when every
correctness gate passes and mean TensorRT neural latency improves by at least
20% over mean PyTorch neural latency. Otherwise the report records a no-go and
runtime behavior remains unchanged.

## Testing and Reporting

- CPU-safe tests cover build command isolation, strict sidecar fields,
  benchmark statistics, complete frame coverage, parity gates, latency gates,
  and report decisions without importing TensorRT or loading Metric3D.
- The real benchmark writes JSON and Markdown reports containing build cost,
  engine size/hash, bindings, resource peaks, all parity metrics, all latency
  statistics, and the explicit module decision.
- Existing Metric3D ONNX/depth-scale and DROID/TensorRT tests run in isolated
  processes before completion.
