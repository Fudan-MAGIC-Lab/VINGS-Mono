# DROID Update-Core Delta-Head FP32 Design

Date: 2026-07-11

## Goal

Build and validate a mixed-precision TensorRT engine in which the DROID update
core remains FP16 except for the neural `delta` head, whose compute and
intermediate tensors are constrained to FP32. The experiment succeeds only if
it preserves the existing module-parity thresholds while retaining at least a
20% latency improvement at E=16.

## Scope

- Keep the existing PyTorch runtime as the default backend.
- Do not integrate the engine into live SLAM in this experiment.
- Do not change the ONNX model, frontend resolution, OFA, mapping, or checkpoint.
- Do not overwrite the existing pure-FP16 engine or its sidecar.
- Do not stage, commit, reset, clean, or remove unrelated worktree content.

## Precision Boundary

The source ONNX graph contains these `delta` neural nodes:

- `/delta/delta.0/Conv`
- `/delta/delta.1/Relu`
- `/delta/delta.2/Conv`

The mixed engine will use TensorRT `precisionConstraints=obey`. Both layer
precision and layer output type will be constrained to FP32 for these three
nodes. All other eligible layers remain available for FP16 selection through
`--fp16`.

Strict `obey` is required so that TensorRT fails the build instead of silently
ignoring a precision request. The verbose build log must demonstrate that the
delta path is FP32 and the weight path remains FP16. TensorRT may otherwise
fuse the first delta and weight convolutions in an unconstrained FP16 build;
applying constraints to the original ONNX node names is intended to prevent
that cross-precision fusion.

## Artifacts

New artifacts use a distinct `deltafp32` suffix:

- `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.plan`
- `engines/tensorrt/droid/droid_update_core_43x77_deltafp32.json`
- dedicated build, resource, parity, and benchmark logs under
  `reports/tensorrt_fp16/`

The sidecar uses the existing strict metadata schema and records the full build
command, hashes, bindings, profiles, runtime versions, compute capability, and
the precision label `fp16_delta_fp32`.

## Build and Validation Flow

1. Add a testable build-command description containing the three exact delta
   node names, strict `obey`, FP32 layer precisions, FP32 layer output types,
   and the unchanged E=1/16/48 profiles.
2. Confirm no SLAM, `trtexec`, or `tegrastats` process is running, then build
   the new engine with a dedicated resource log.
3. Require `trtexec` success and inspect the verbose layer report. A build that
   succeeds but does not preserve the requested precision boundary is a
   failure.
4. Generate the strict sidecar and load the engine through `TensorRTEngine`.
5. Run the existing seeded benchmark at E=1,4,16,32,48 with 10 warmup and 50
   measured iterations.
6. Compare the new mixed engine against both PyTorch for correctness and the
   pure-FP16 engine for latency context.

## Acceptance Criteria

For every output and every tested edge count:

- shapes match exactly;
- all values are finite;
- cosine similarity is at least 0.999;
- mean relative error is at most 1%.

At E=16, TensorRT latency must improve by at least 20% relative to PyTorch.
Online integration is recommended only if all correctness and performance
criteria pass. Otherwise the report records a no-go decision and the runtime
remains unchanged.

## Failure Handling

- If strict precision constraints are unsupported or unsatisfiable, record the
  raw TensorRT error and stop. Do not retry with `prefer` automatically.
- If the build log shows ignored constraints, unexpected delta/weight fusion,
  or FP16 delta tactics, reject the artifact even if the build exits zero.
- If parity still exceeds 1%, retain the engine only as diagnostic evidence and
  do not integrate it.
- If parity passes but the E=16 improvement is below 20%, do not integrate it.

## Testing

- Unit tests verify the exact mixed-precision constraint specification and
  distinct output paths before the build script is changed.
- Existing metadata, runner, benchmark, DROID update-core, and profiling tests
  remain CPU-safe and are run in isolated processes.
- Final verification includes hashes, strict deserialization, dependency
  health, process cleanup, build-log precision evidence, and the complete
  module report.
