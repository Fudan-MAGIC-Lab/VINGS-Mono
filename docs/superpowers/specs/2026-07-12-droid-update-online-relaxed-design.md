# DROID Update-Core TensorRT Online Design with Relaxed Accuracy Gates

Date: 2026-07-12

## Decision

Integrate the existing mixed-precision `droid_update_core_43x77_deltafp32.plan` as an explicit opt-in backend. The delta neural head remains FP32 and the rest of the TensorRT graph remains eligible for FP16. The pure-FP16 engine is not used online.

## Runtime boundary

`UpdateModule.forward()` remains responsible for the current 5D reshape, optional zero flow, output permutations, tuple shape, and PyTorch `GraphAgg`. Only the stable flattened 4D boundary is delegated:

```text
forward_core(net, inp, corr, flow)
  -> updated_net, delta, weight
```

The online interface accepts CUDA FP16 (AMP runtime) or FP32 tensors independently; `flow=None` legitimately produces FP32 flow beside FP16 recurrent state, while CovisibleGraph legitimately produces strided flow. The adapter normalizes every input to contiguous FP32 for TensorRT, validates FP32 engine outputs, then restores the incoming `net` dtype before returning. Spatial dimensions remain `43x77`; edge count `E` must be between 1 and 48. Correlation construction, `ii/jj`, GraphAgg, reprojection, BA, and upsample remain unchanged.

## Backend selection and lifecycle

Add independent controls:

```yaml
inference:
  droid_update_backend: torch
  droid_update_engine: null
```

CLI overrides are `--droid-update-backend {torch,tensorrt}` and `--droid-update-engine PATH`. PyTorch remains the default.

Strict TensorRT mode detaches the PyTorch-only update-core modules (`corr_encoder`, `flow_encoder`, `gru`, `delta`, and `weight`), clears their CUDA allocation, and then loads the engine. `agg` remains resident. Non-strict mode retains the core modules for one logged fallback. Runtime metadata records requested/actual backend, strict state, engine path/hash, precision label, and fallback reason.

## Failure behavior

- Invalid names, dtypes, devices, contiguity, E range, or spatial shapes fail before enqueue.
- Strict load/inference failures propagate and cannot report TensorRT success.
- Non-strict failure closes TensorRT, switches once to `forward_core_torch`, and records actual backend `torch`.
- No runtime CUDA synchronization is added outside smoke tests and benchmarks.

## Online evaluation sequence

1. Real-engine smoke at E=1,4,16,32,48.
2. Paired SmallCity-50 with PyTorch Metric3D and PyTorch fnet/cnet in both variants; update backend is the only variable.
3. If the relaxed gate passes, paired SmallCity-50 combining Metric3D TensorRT and update TensorRT.
4. SmallCity-200 is conditional on the combined SmallCity-50 result.

## Relaxed acceptance gates

Module evidence is accepted with cosine at least `0.999`, all finite outputs, exact shapes, and delta MRE at most `1.5%`.

The online SmallCity-50 update-only pair requires:

- no tracking failure, fallback, hang, or residual process;
- ATE increase no more than `20%`;
- mean PSNR drop no more than `0.5 dB`;
- mean SSIM drop no more than `0.03`;
- exported sample-count change no more than `10%`;
- `frame_total` improvement at least `5%`.

Keyframe/export IDs are recorded but are no longer required to be identical. Performance is the primary objective. If the speed gate fails, stop even when quality passes.

## Guardrails

Do not change resolution, OFA, mapping, motion-gate parameters, checkpoint, TensorRT engine precision, or default backend. Do not enable cnet/fnet TensorRT. Preserve the dirty worktree and generated evidence without staging, committing, resetting, cleaning, or overwriting prior runs.
