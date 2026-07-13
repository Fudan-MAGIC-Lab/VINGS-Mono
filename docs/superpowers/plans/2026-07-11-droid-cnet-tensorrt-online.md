# DROID cnet-Only TensorRT Online Integration Plan

**Goal:** Add an explicit opt-in TensorRT backend for DROID cnet only, prove the actual backend and fallback state, and evaluate it against an otherwise identical PyTorch control. fnet remains PyTorch because its TensorRT module parity failed.

**Scope:** Preserve the default PyTorch behavior, input resolution, Metric3D selection, motion gate, OFA, mapping, and update operator. Do not add a fnet TensorRT path and do not run TensorRT and PyTorch cnet concurrently in strict mode.

## Task 1: Stable cnet adapter contract

- Add a small adapter around `TensorRTEngine` that accepts the current cnet 5D input `[1,1,3,344,616]`, validates the static `B*N=1` constraint, calls binding `image`, and restores `features [1,1,256,43,77]`.
- Keep the existing split into `net/inp` plus `tanh/relu` in `MotionFilter`; the adapter replaces only the encoder computation.
- Write tests first for exact shape, CUDA float32 contiguous input, CUDA float16 output, strict binding validation, and no eager TensorRT import.

## Task 2: Explicit backend selection and lifecycle

- Add independent configuration/CLI controls `droid_cnet_backend: torch|tensorrt` and `droid_cnet_engine`; keep defaults `torch` and `null`.
- Wire the adapter after normal checkpoint loading. In strict TensorRT mode, detach and release the PyTorch cnet module, clear its CUDA allocation, and only then load the engine before frame 0; retain fnet and update in PyTorch.
- In non-strict mode, allow one logged lazy fallback to the original cnet. Record requested backend, actual backend, strict state, engine path/hash, and fallback reason in runtime profile metadata.
- Add tests proving strict load/inference failures stop without silently reporting TensorRT, while non-strict fallback reports actual backend `torch` exactly once.

## Task 3: Real-engine smoke and regression

- Run a real SmallCity frame through the accepted cnet plan and require exact `[1,1,256,43,77]` output, float16 dtype, finite values, and actual backend `tensorrt`.
- Verify `MotionFilter.context_encoder` still returns the same `[1,128,43,77]` `net/inp` shapes and transformations.
- Run the isolated regression suite, `pip check`, strict engine/hash checks, `git diff --check`, and residual-process checks.

## Task 4: Paired SmallCity-50 gate

- Create guarded unique PyTorch-control and cnet-TensorRT candidate runs. Hold Metric3D and every balanced-fast parameter identical so cnet is the only backend difference.
- Require strict TensorRT, no fallback, identical keyframe/export frame IDs and sample count, no tracking failure, ATE difference at most `0.01 m`, PSNR difference at most `0.05 dB`, and SSIM difference at most `0.005`.
- Record `motion_filter_context_encoder`, `frame_total`, peak RAM/swap, and shutdown state. Because cnet's theoretical total contribution is only about `0.232%`, report end-to-end timing but do not invent a minimum total-speedup gate.

## Task 5: Conditional SmallCity-200 decision

- Proceed only if SmallCity-50 passes every behavior/backend gate.
- Run fresh paired SmallCity-200 evaluations with unique prefixes and the same single-variable design.
- Keep cnet explicit opt-in unless the paired quality, backend, memory, and shutdown gates pass. A module-level `2.046x` speedup alone is not authorization to change defaults.
