# DROID fnet/cnet TensorRT Design

**Date:** 2026-07-11

**Status:** Approved design, awaiting written-spec review

## Objective

Add independently selectable TensorRT FP16 backends for the DROID feature encoder (`fnet`) and context encoder (`cnet`) on the static SmallCity frontend shape. Each encoder must pass its own module and paired SmallCity gates. A passing fnet may be integrated while cnet remains on PyTorch, or vice versa.

This phase does not change the default PyTorch backend, DROID update core, frontend resolution, Metric3D behavior, OFA, mapping, or checkpoint normalization. It does not start INT8 or claim that SmallCity plans support Hotel.

## Authorization and Existing Evidence

The user explicitly accepted the Metric3D SmallCity-200 SSIM exception as a conditional GO to continue the DROID fnet/cnet phase. The original Metric3D strict gate result remains preserved. This exception does not relax any DROID encoder or update-core threshold.

Current SmallCity-200 PyTorch profiling reports:

| Stage | Calls | Mean | Total | % frame total |
|---|---:|---:|---:|---:|
| fnet | 69 | 50.341 ms | 3.474 s | 1.107% |
| cnet | 69 | 10.544 ms | 0.728 s | 0.232% |
| combined | 69 each | 60.884 ms | 4.201 s | 1.339% |

The maximum possible end-to-end gain is limited. Module acceleration alone is insufficient; each online branch must produce a measurable positive `frame_total` improvement.

## Audited Runtime Contract

`MotionFilter` is the only online consumer of `net.fnet` and `net.cnet`. It supplies normalized RGB under CUDA autocast:

```text
5D input:  [1, 1, 3, 344, 616], float32, contiguous
fnet:      [1, 1, 128, 43, 77], float16, contiguous
cnet:      [1, 1, 256, 43, 77], float16, contiguous
cnet split:[1, 1, 128, 43, 77] net/inp
```

The checkpoint contains 32 fnet keys, 32 cnet keys, and 38 update keys after removing the `module.` prefix and cropping the existing update heads. FP32 parameter sizes are approximately 2.73 MiB for fnet and 2.79 MiB for cnet.

The accepted SmallCity admitted-frame IDs are available in the PyTorch SmallCity-200 `runtime_profile_events.csv`. Both encoders ran on the same 69 frame IDs. Module validation selects at least 20 IDs distributed across that sequence.

## Encoder Core Refactor

Add a stable 4D method to `BasicEncoder`:

```python
forward_core(x)
```

Its contract is:

```text
input:  [E, 3, H, W]
output: [E, output_dim, H/8, W/8]
```

The existing `forward(x)` retains its 5D API. It reshapes `[B,N,C,H,W]` to `[B*N,C,H,W]`, calls `forward_core`, and restores `[B,N,C2,H2,W2]`. Tests cover multiple B/N combinations and require exact or near-exact equivalence with the original computation order.

The online TensorRT plans are static at `E=1`, `H=344`, and `W=616`.

## Export Boundary and Artifacts

Build separate engines because fnet runs for every frame admitted by the motion gate while cnet runs only when context is required:

```text
engines/tensorrt/droid/droid_fnet_b1_344x616.onnx
engines/tensorrt/droid/droid_fnet_b1_344x616_fp16.plan
engines/tensorrt/droid/droid_cnet_b1_344x616.onnx
engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan
```

Use opset 17 first, fixed input/output shapes, and no dynamic dimensions. Export wrappers receive normalized float32 RGB and execute the encoder core under CUDA autocast so the graph output preserves the online FP16 contract.

Export loading must exactly mirror `DBAFusion.load_weights()`:

1. Load `ckpts/droid.pth` on CPU.
2. Remove `module.` prefixes.
3. Apply the existing update weight/delta head cropping.
4. Strictly load `DroidNet`.
5. Select `fnet` or `cnet`, move only the export model to CUDA, and set evaluation mode.

Each ONNX file is checked with ONNX checker, shape inference, and ONNX Runtime CUDA on at least 20 real admitted inputs before TensorRT build authorization.

Generated ONNX and plan files remain uncommitted.

## Runtime Adapter Architecture

Install an independent 5D-compatible adapter in place of `net.fnet` or `net.cnet`. Existing call sites remain unchanged.

The adapter:

1. Validates `[B,N,3,H,W]`.
2. Flattens to `[B*N,3,H,W]`.
3. Requires `[1,3,344,616]` for TensorRT.
4. Runs the corresponding engine.
5. Restores the 5D output.
6. Returns contiguous FP16 output.

Checkpoint loading remains on CPU. When an encoder uses TensorRT, its PyTorch module is removed before the remaining network is moved to CUDA. The adapter is installed after checkpoint validation. This prevents selected PyTorch encoder weights and the TensorRT engine from coexisting on CUDA.

The two adapters are independent. Supported configurations include:

```text
fnet=torch,    cnet=torch
fnet=tensorrt, cnet=torch
fnet=torch,    cnet=tensorrt
fnet=tensorrt, cnet=tensorrt
```

## Configuration and CLI

Configuration defaults remain:

```yaml
inference:
  droid_fnet_backend: torch
  droid_fnet_engine: null
  droid_cnet_backend: torch
  droid_cnet_engine: null
  tensorrt_strict: false
```

Add independent CLI overrides:

```text
--droid-fnet-backend {torch,tensorrt}
--droid-fnet-engine PATH
--droid-cnet-backend {torch,tensorrt}
--droid-cnet-engine PATH
```

The existing `--tensorrt-strict` flag is shared. Missing flags do not change configuration values.

## Failure and Fallback Semantics

Strict mode is mandatory for module benchmarks and paired evaluations.

In strict mode, a missing engine, invalid sidecar, binding mismatch, wrong dtype, wrong static shape, deserialization failure, or inference failure terminates the run without constructing a PyTorch fallback for that encoder.

In non-strict mode, only the failed encoder transitions:

1. Preserve and log the original error once.
2. Clear the engine output cache and release its runner.
3. Clear CUDA allocator cache when available.
4. Lazily construct the correct `BasicEncoder` configuration.
5. Reload only the corresponding checkpoint prefix.
6. Move it to CUDA in evaluation mode.
7. Retry the current inference once.
8. Use PyTorch for later calls without retrying TensorRT.

The other encoder backend is unchanged. A 5D request with `B*N != 1` follows the same strict/fallback policy because the accepted plans are static.

## Backend Reporting

Log one initialization record per encoder and one transition record if fallback occurs. Runtime metadata records separate objects for `droid_fnet` and `droid_cnet`:

- requested backend;
- actual backend;
- strict state;
- engine path;
- fallback reason.

Reports and benchmarks use the actual backend. A PyTorch fallback must never be labeled TensorRT.

## Module Validation

Validate fnet and cnet independently across at least 20 admitted SmallCity inputs:

- exact input and output shapes;
- output dtype matches online FP16 behavior;
- all outputs finite;
- cosine similarity at least `0.999`;
- mean relative error at most `1%`;
- mean module latency improvement at least `20%`.

Record mean, median, p90, engine/checkpoint/ONNX hashes, bindings, TensorRT/CUDA/compute capability, and resource peaks. An encoder that fails any module gate is not integrated online.

## Online Evaluation Sequence

Run branches independently:

1. fnet module export, parity, engine, and benchmark.
2. cnet module export, parity, engine, and benchmark.
3. fnet-only SmallCity-50, then SmallCity-200 if SmallCity-50 passes.
4. cnet-only SmallCity-50, then SmallCity-200 if SmallCity-50 passes.
5. Combined fnet+cnet only after both branches independently pass.

Every candidate receives a fresh paired PyTorch control from the same worktree and identical balanced-fast options. Use unique prefixes and output directories. Preserve stdout, status, `tegrastats`, runtime metadata/profile, evaluator output, and tracking comparison.

Quality/resource gates remain:

- no tracking failure or shutdown hang;
- ATE increase at most `max(15%, 0.10 m)`;
- mean PSNR drop at most `0.5 dB`;
- mean SSIM drop at most `0.02`;
- keyframe/export sample count change at most `10%`;
- target encoder mean latency improvement at least `20%`;
- measurable positive `frame_total` improvement;
- peak RAM increase at most `500 MB`;
- actual requested TensorRT backend with no strict fallback.

The Metric3D SSIM exception is specific to its reviewed SmallCity-200 run and does not automatically waive an encoder result.

## Testing

All implementation follows red-green TDD. Tests cover:

- `forward_core` equivalence and finite outputs for multiple B/N combinations;
- 4D/5D shape restoration, contiguity, and FP16 output;
- checkpoint grouping and exact loading;
- selected encoder removal before CUDA transfer;
- independent and mixed backend configuration;
- lazy TensorRT imports;
- strict loading, shape, dtype, and inference failures;
- independent non-strict lazy fallback;
- one-time logs and actual-backend metadata;
- real fnet and cnet engine smoke tests;
- exporter, validation-report, builder, and benchmark decision logic;
- all existing isolated regression modules.

## Stop Conditions

- Stop an encoder branch if ONNX or TensorRT parity fails after identifying the first divergent operation.
- Stop online integration if module speedup is below 20%.
- Stop cnet after module validation if its end-to-end contribution cannot be measured reliably.
- Keep PyTorch for any failed encoder.
- Do not revisit update-core integration unless its separate 1% parity policy is explicitly changed.
- Do not begin Hotel engines; they require separate static `256x448` plans.
