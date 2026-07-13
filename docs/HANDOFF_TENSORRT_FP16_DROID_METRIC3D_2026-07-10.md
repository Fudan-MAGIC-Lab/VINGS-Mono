# Handoff: TensorRT FP16 Acceleration for Metric3D and DROID

Date: 2026-07-10

Jetson repository: `/home/jetson/VINGS-Mono`

Windows workspace mirror: `J:/VINGS-Mono`

SSH alias: `jetson-codex`

## New Conversation Goal

Add TensorRT FP16 inference paths for Metric3D and the neural parts of DROID on Jetson Orin. The objective remains balanced-fast: accept a limited quality loss when it produces a meaningful end-to-end speed gain, while keeping tracking stable and measurable on SmallCity ground truth.

Recommended opening prompt:

```text
Continue TensorRT FP16 acceleration in /home/jetson/VINGS-Mono. First read docs/HANDOFF_TENSORRT_FP16_DROID_METRIC3D_2026-07-10.md and inspect the dirty worktree without reverting anything. Follow the order in the handoff: establish paired PyTorch baselines and finer inference profiling, implement a reusable TensorRT runner, accelerate Metric3D at the static scale=0.75 shape, then accelerate DROID fnet/cnet. Only attempt the DROID update core if profiling proves it is worth converting. Keep PyTorch as the default, do not start INT8, do not change frontend image size, and do not touch the OFA/mapping work while establishing the TensorRT comparison.
```

## Scope and Guardrails

In scope:

- TensorRT 8.5.2 FP16 engines built on this Jetson.
- Direct binding of PyTorch CUDA tensor addresses to TensorRT inputs and outputs.
- Metric3D first, using the current balanced-fast `metric_depth_scale=0.75` static shape.
- DROID `fnet` and `cnet` second.
- DROID update-core conversion only after internal profiling and an explicit go/no-go check.
- PyTorch/TensorRT backend selection, strict benchmark mode, clear logging, tests, and paired quality/runtime evaluation.

Out of scope for this pass:

- INT8 calibration or quantization-aware training.
- Changing frontend image size to manufacture a speedup.
- Converting `CorrBlock`, `droid_backends`, dense BA, mapping, or Gaussian rendering to TensorRT.
- DLA deployment.
- Reworking the OFA/VPI/NVMM motion-gate path.
- Loading both full PyTorch and TensorRT copies of Metric3D for the whole run.
- Running all 877 SmallCity frames before 50- and 200-frame acceptance passes.

## Current Git State

Current HEAD:

```text
b7c68bb chore: default jetson hotel depth scale to 0.75
```

The worktree is intentionally dirty. Relevant modified/untracked work includes tracking profiling, adaptive DBA settings, mapping budgets, evaluation export, the OFA/VPI C++ extension, GStreamer NVMM support, tests, reports, and data symlinks. The Metric3D submodule is also dirty.

Before editing, run:

```bash
cd /home/jetson/VINGS-Mono
git status --short
```

Do not reset, clean, checkout, delete, or overwrite existing changes. In particular, preserve:

```text
scripts/frontend/dbaf.py
scripts/frontend/dbaf_frontend.py
scripts/frontend/motion_filter.py
scripts/frontend/motion_gate.py
scripts/frontend/vpi_ofa_gate_cpp.cpp
scripts/gaussian/gaussian_base.py
scripts/metric/metric_model.py
scripts/run.py
submodules/metric_modules
tests/
```

There are odd untracked names such as `2`, `\`, and ` \`. Ignore them; do not remove them as part of this task.

## Jetson Environment

Verified on 2026-07-10:

| Item | Value |
|---|---|
| Device | Jetson Orin, CUDA capability 8.7 |
| Architecture | aarch64 |
| L4T | R35.6.3 |
| Conda environment | `vings_jetson`, Python 3.8 |
| PyTorch | `2.1.0a0+41361538.nv23.06` |
| PyTorch CUDA runtime | 11.4 |
| TensorRT runtime/dev | 8.5.2.2 |
| `trtexec` | `/usr/src/tensorrt/bin/trtexec` |
| Conda NumPy | 1.24.4 |
| Conda Protobuf | 5.29.6 |
| ONNX Runtime GPU | 1.15.1 |
| ONNX | not installed |
| Torch-TensorRT | not installed |
| PyCUDA / cuda-python | not installed |

The system TensorRT Python binding is under `/usr/lib/python3.8/dist-packages`, not on the conda environment's normal path. This command works without shadowing conda NumPy:

```bash
source /home/jetson/miniconda3/etc/profile.d/conda.sh
conda activate vings_jetson
PYTHONPATH=/usr/lib/python3.8/dist-packages \
python -c "import tensorrt as trt; print(trt.__version__)"
```

Expected output:

```text
8.5.2.2
```

Do not add `/usr/lib/python3/dist-packages` globally. It shadows conda packages and previously changed the visible NumPy from 1.24.4 to the system 1.17.4.

Before adding ONNX, capture the environment:

```bash
mkdir -p reports/tensorrt_fp16
python -m pip freeze > reports/tensorrt_fp16/pip_freeze_before.txt
```

The first conservative ONNX candidate to try is `onnx==1.14.1`, installed without dependency upgrades:

```bash
python -m pip install --no-deps onnx==1.14.1
python -c "import onnx; print(onnx.__version__)"
python -m pip check
```

If that import fails, stop and record the exact error. Do not solve it by upgrading PyTorch, NumPy, Protobuf, CUDA, or TensorRT in place. `onnxruntime-gpu` is not required for the initial export/build/runtime path.

## Existing Performance and Quality Reference

The best current ground-truth reference is the SmallCity-200 fast run:

```text
/home/jetson/VINGS-Mono/output/smallcity_gt_eval_20260709_111713/
  07-10-02-17-hierarchical-smallcit-jetson_smallcity_gt200_fast_t10
```

Configuration highlights:

```text
frames: 200
training_iters: 10
metric_depth_scale: 0.75
metric depth schedule: keyframe, warmup 8, min interval 3, force interval 10
motion gate: vpi_cpp, threshold 12, force interval 8, resize 96x160
mapping/pruning/pixel budgets: enabled
```

Runtime:

| Stage | Calls | Total | Mean |
|---|---:|---:|---:|
| frame total | 200 | 321.348 s | 1606.74 ms/frame |
| tracking | 200 | 210.120 s | 1050.60 ms/frame |
| DBA update | 69 | 179.458 s | 2600.84 ms/call |
| mapping | 160 | 102.900 s | 643.12 ms/call |
| Metric3D | 44 | 20.765 s | 471.94 ms/call |
| DROID feature encoder | 69 | 6.405 s | 92.82 ms/call |
| DROID context encoder | 69 | 0.835 s | 12.10 ms/call |
| motion-filter corr/update | 68 | 1.000 s | 14.71 ms/call |
| OFA motion gate | 200 | 1.325 s | 6.63 ms/frame |

Quality from the lightweight evaluator:

```text
ATE Sim3 RMSE: 0.525308 m
mean PSNR:     13.811639
mean SSIM:     0.670030
exported pose/render samples: 39
```

Resources:

```text
wall time: about 339 s
RAM average: 14360 MB
RAM peak:    15072 MB of 15503 MB
swap peak:    5706 MB
GR3D average: 69.6%
GR3D max:     99%
GPU temp max: 74.5 C
```

This run is a reference, not the formal TensorRT baseline. Before each TensorRT comparison, run a fresh PyTorch control from the same worktree and with exactly the same command/options. Thermal state, mapping trajectory, and keyframe decisions can move end-to-end times.

The existing SmallCity-50 run used `training_iters=50`, so it must not be compared directly with a new `training_iters=10` TensorRT run. Produce a paired SmallCity-50 PyTorch control first.

## Why the Work Is Ordered This Way

Metric3D is the cleanest first target: it is isolated inference with a fixed balanced-fast input size and currently costs about 472 ms per prediction. Even a 2x Metric3D speedup saves only about 10 seconds in the 200-frame run, so report the end-to-end improvement honestly.

DROID encoders are technically easier than DROID's recurrent update path, but their measured combined cost is only about 105 ms per accepted frame. The 2.6-second DBA update includes graph work, the neural update operator, upsampling, and custom dense BA. TensorRT cannot accelerate all of that.

Therefore, do not attribute all DBA time to the DROID neural network. Add finer profiling before converting the update operator.

## Target Architecture

### Shared TensorRT Runtime

Create a focused acceleration package; do not name it `tensorrt`, because that would shadow NVIDIA's Python package.

Recommended files:

```text
scripts/acceleration/__init__.py
scripts/acceleration/trt_engine.py
scripts/acceleration/engine_metadata.py
scripts/acceleration/export_metric3d_onnx.py
scripts/acceleration/export_droid_onnx.py
scripts/profiling/benchmark_inference.py
tests/test_trt_engine.py
tests/test_metric3d_trt_backend.py
tests/test_droid_trt_backend.py
```

`TensorRTEngine` should:

- import TensorRT lazily; if the normal import fails, add only `/usr/lib/python3.8/dist-packages` to `sys.path` and retry;
- deserialize one `.plan` file and create one execution context;
- validate binding names, dtypes, device, and shapes;
- support static shapes first and TensorRT optimization profiles where explicitly needed;
- allocate/reuse output tensors with `torch.empty(..., device="cuda")`;
- bind `tensor.data_ptr()` directly, with no NumPy/PyCUDA staging copy;
- call TensorRT 8.5 `execute_async_v2` on `torch.cuda.current_stream().cuda_stream`;
- avoid `torch.cuda.synchronize()` in the normal inference path;
- expose synchronization only in correctness tests and benchmarks;
- fail loudly in strict benchmark mode;
- log the actual backend once so a fallback cannot be mistaken for TensorRT.

Each plan must have a JSON sidecar containing at least:

```text
engine SHA256
source ONNX SHA256
checkpoint SHA256
TensorRT version
CUDA version
compute capability
input/output names, shapes, and dtypes
precision
build command
```

Serialized plans are Jetson/TensorRT/shape specific. Store them under `engines/tensorrt/` and do not commit `.onnx` or `.plan` artifacts.

### Backend Configuration

Keep all defaults on PyTorch. Add independent controls rather than one global switch:

```yaml
inference:
  metric3d_backend: torch
  metric3d_engine: null
  droid_encoder_backend: torch
  droid_fnet_engine: null
  droid_cnet_engine: null
  droid_update_backend: torch
  droid_update_engine: null
  tensorrt_strict: false
```

Recommended CLI overrides in `scripts/run.py`:

```text
--metric-depth-backend {torch,tensorrt}
--metric-depth-engine PATH
--droid-encoder-backend {torch,tensorrt}
--droid-fnet-engine PATH
--droid-cnet-engine PATH
--droid-update-backend {torch,tensorrt}
--droid-update-engine PATH
--tensorrt-strict
```

Benchmark runs must use `--tensorrt-strict`. Production fallback may be allowed, but it must instantiate the PyTorch model lazily after an engine failure. Do not keep both complete backends resident just to make fallback instant.

## Implementation Sequence

### Phase 0: Freeze a Paired Baseline and Add Missing Profiling

1. Capture `git status`, package versions, `tegrastats`, and a fresh SmallCity-50 PyTorch run with `training_iters=10`.
2. Split Metric3D timing into preprocessing, neural forward, and postprocessing. The current 471.94 ms includes CPU/NumPy/OpenCV transfers around the model.
3. Split `CovisibleGraph.update()` timing into at least correlation lookup, `update_op`, GraphAgg/upsample, and `video.ba`.
4. Run the same 50-frame control again and confirm profiling alone does not change keyframe count or evaluation outputs.

Suggested stage names:

```text
metric_depth_preprocess
metric_depth_model
metric_depth_postprocess
covisible_graph_corr
covisible_graph_update_op
covisible_graph_graph_agg
covisible_graph_ba
covisible_graph_upsample
```

Go/no-go rule for DROID update conversion:

```text
Only continue to the update-core engine if covisible_graph_update_op is at least
15% of DBA update time or at least 10% of total frame time on SmallCity-50/200.
```

If BA/graph management dominates, document that result and do not force TensorRT into the wrong boundary.

### Phase 1: Build and Test the Shared TensorRT Runner

Use TDD. Start with metadata/shape/dtype tests that do not require a real engine, then add a Jetson-only integration test using a tiny ONNX convolution model.

Required behavior:

- missing plan: clear exception in strict mode;
- metadata mismatch: reject the engine;
- unexpected shape or dtype: reject before enqueue;
- dynamic binding: set all binding shapes before allocation/enqueue;
- same CUDA stream: downstream PyTorch operations consume outputs without a global synchronize;
- persistent output buffers: repeated inference does not allocate a new tensor each call when shape is unchanged.

Run CPU-safe tests first:

```bash
python -m unittest tests.test_trt_engine
```

Run the Jetson integration test explicitly after TensorRT import is available:

```bash
PYTHONPATH=/usr/lib/python3.8/dist-packages \
python -m unittest tests.test_trt_engine.TensorRTIntegrationTest
```

### Phase 2: Metric3D FP16

Current entry points:

```text
scripts/metric/metric_model.py
submodules/metric_modules/metric.py
submodules/metric_modules/metric3d/mono/utils/do_test.py
```

Current data flow:

```text
PyTorch image -> CPU NumPy -> canonical resize/normalize -> CUDA tensor
-> Metric3D PyTorch model -> CPU NumPy -> OpenCV resize -> CUDA depth tensor
```

Preserve preprocessing and postprocessing first. Replace only the neural forward, then optimize transfers separately if profiling shows they still matter.

For `metric_depth_scale=0.75`, the existing scaling code rounds the native `(616, 1064)` crop to multiples of 28:

```text
static engine input: 1 x 3 x 448 x 784
```

Use a depth-only export wrapper around:

```python
predictor.model_.module.depth_model(input=rgb_input)["prediction"]
```

The current dense model does not consume `cam_model` in its model path. Do not add unused camera-model tensors to the TensorRT engine, but prove parity against the existing PyTorch call before relying on this simplification.

Before export:

- load the existing checkpoint from `ckpts/metric_depth_vit_small_800k.pth`;
- apply `metric_depth_scale=0.75`;
- call the PyTorch model once so lazy buffers such as `depth_expectation_anchor` exist;
- use `eval()` and `torch.no_grad()`;
- export one static batch-1 shape; do not start with dynamic spatial dimensions.

Initial ONNX settings:

```text
opset: 17
input name: rgb
output name: depth
static shape: 1x3x448x784
constant folding: enabled
```

Build on the Jetson, with no SLAM process running:

```bash
mkdir -p engines/tensorrt/metric3d
/usr/src/tensorrt/bin/trtexec \
  --onnx=engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx \
  --saveEngine=engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan \
  --fp16 \
  --memPoolSize=workspace:1024MiB \
  --buildOnly \
  --verbose
```

If parsing fails, record the first unsupported ONNX node and its producer stack. Likely trouble spots include traced diagnostic branches, lazy buffer creation, interpolation variants, and the recurrent decoder. Remove inference-only diagnostics from the export wrapper only when output-parity tests prove the change is an identity. Do not jump directly to a custom TensorRT plugin.

If full-model export remains blocked, test one partition at a time and measure its share. Do not declare success from converting only the ViT encoder unless end-to-end Metric3D latency improves materially.

Memory rule: in TensorRT runtime mode, do not construct `Metric(...)` and load its complete PyTorch weights before loading the engine. Backend selection must happen before model construction. Otherwise the engine and PyTorch model will coexist near an existing 15 GB RAM peak.

### Phase 3: DROID fnet/cnet FP16

Relevant files:

```text
scripts/frontend/droid_net.py
scripts/frontend/modules/extractor.py
scripts/frontend/motion_filter.py
scripts/frontend/dbaf.py
```

`BasicEncoder.forward()` accepts `[B, N, 3, H, W]`, flattens `B*N`, runs 2D convolutions, and reshapes the result. Export a simple 4D core (`[B*N, 3, H, W]`) and make the runtime adapter preserve the existing 5D interface.

Start with the current SmallCity spatial size from `configs/hierarchical/smallcity.yaml`:

```text
input: 1 x 3 x 344 x 616
fnet output: 1 x 128 x 43 x 77
cnet output: 1 x 256 x 43 x 77
```

Build separate engines:

```text
droid_fnet_b1_344x616_fp16.plan
droid_cnet_b1_344x616_fp16.plan
```

Keep them separate because `fnet` runs on every frame admitted by the motion gate, while `cnet` is needed when a frame becomes a keyframe. A fused engine would run unnecessary context inference.

Do not replace the DROID checkpoint or change normalization. The adapter must preserve existing AMP-compatible output dtype and the 5D shapes expected by `MotionFilter`, `DepthVideo`, and `CovisibleGraph`.

Hotel uses a different frontend shape (`256x448`) and therefore needs separate plans. Do not claim one static SmallCity plan supports both.

### Phase 4: DROID Update Core, Conditional

Relevant files:

```text
scripts/frontend/droid_net.py
scripts/frontend/covisible_graph.py
scripts/frontend/modules/gru.py
scripts/frontend/modules/clipping.py
```

Do not export `UpdateModule.forward()` unchanged. Its graph path includes `GraphAgg`, `torch.unique`, `torch_scatter.scatter_mean`, and `ii/jj` bookkeeping, which are poor TensorRT boundaries.

If Phase 0 profiling passes the go/no-go threshold, refactor without changing behavior:

```text
forward_core(net, inp, corr, flow)
  -> updated_net, delta, weight

forward(..., ii, jj, upsample)
  -> calls forward_core
  -> runs GraphAgg in PyTorch when requested
  -> preserves the existing return tuple
```

Export only `forward_core`. Flatten factor count `E` into the TensorRT batch dimension:

```text
minimum E: 1
optimum E: 16
maximum E: 48 (current max_factors)
spatial shape for SmallCity: 43x77
```

The engine should consume the flattened 4D tensors for `net`, `inp`, `corr`, and `flow`, and return flattened `updated_net`, `delta`, and `weight`. `ii`, `jj`, GraphAgg, correlation construction, reprojection, and dense BA remain in PyTorch/custom CUDA.

`GradientClip` is an inference identity. If ONNX export cannot handle its custom autograd function, replace it with an identity only inside the export wrapper and require exact/near-exact PyTorch output comparison.

## Correctness and Acceptance Gates

### Module-Level Parity

Metric3D, over at least 10 real SmallCity inputs:

```text
output shape exactly matches
all outputs finite
cosine similarity >= 0.999
mean relative error on valid PyTorch depth <= 2%
```

DROID fnet/cnet, over at least 20 admitted frames:

```text
output shape exactly matches
all outputs finite
cosine similarity >= 0.999
mean relative activation error <= 1%
```

DROID update core, if implemented:

```text
test E = 1, 4, 16, 32, 48
delta/weight shapes exactly match
no NaN/Inf
motion-filter keyframe decision agreement >= 95% on the sampled frames
```

### End-to-End Gates

Run paired PyTorch and TensorRT variants from the same worktree.

SmallCity-50 is the smoke/quality gate. SmallCity-200 is the decision gate.

For SmallCity-200, compared with its paired PyTorch control:

```text
no tracking failure
ATE increase <= max(15%, 0.10 m)
mean PSNR drop <= 0.5 dB
mean SSIM drop <= 0.02
keyframe/export sample count change <= 10%
target component mean latency improves by at least 20%
end-to-end frame_total must improve measurably, not only trtexec throughput
peak RAM increase <= 500 MB
no persistent idle/futex hang during shutdown or late mapping
```

Do not enable a TensorRT backend by default if it misses the 20% component-speed threshold or if its engine memory erases the end-to-end benefit.

Evaluate each completed run with:

```bash
RUN_DIR=$(find /home/jetson/VINGS-Mono/output/smallcity_tensorrt_eval_20260710 \
  -mindepth 1 -maxdepth 1 -type d \
  -name '*jetson_smallcity_gt200_tensorrt_eval*' \
  -print | sort | tail -n 1)
test -n "$RUN_DIR"
python scripts/profiling/evaluate_smallcity_run.py \
  --run-dir "$RUN_DIR" \
  --dataset-root /home/jetson/VINGS-Mono/data/smallcity_subset_200/small_city \
  --output "$RUN_DIR/smallcity_eval_metrics.md"
```

The evaluator uses Sim3 ATE and lightweight PSNR/simple-SSIM on exported samples. It is not the paper's official full-877 rendering evaluation, so do not compare these numbers as if they were identical to the paper metrics.

## Reproducible SmallCity-200 Shape of Run

Use this as the base command and add only the backend flags being tested:

```bash
cd /home/jetson/VINGS-Mono
source /home/jetson/miniconda3/etc/profile.d/conda.sh
conda activate vings_jetson

python scripts/run.py configs/hierarchical/smallcity.yaml \
  --prefix jetson_smallcity_gt200_tensorrt_eval \
  --dataset-root data/smallcity_subset_200/small_city \
  --output-dir output/smallcity_tensorrt_eval_20260710 \
  --no-vis \
  --profile-runtime \
  --export-eval \
  --export-eval-interval 1 \
  --skip-save-ply \
  --training-iters 10 \
  --enable-mapping-budget \
  --enable-jetson-pruning \
  --enable-pixel-budget \
  --enable-metric-depth-schedule \
  --metric-depth-mode keyframe \
  --metric-depth-warmup 8 \
  --metric-depth-keyframe-min-interval 3 \
  --metric-depth-keyframe-force-interval 10 \
  --metric-depth-high-motion-ratio 3.0 \
  --metric-depth-scale 0.75 \
  --enable-jetson-motion-gate \
  --motion-gate-backend vpi_cpp \
  --motion-gate-threshold 12.0 \
  --motion-gate-force-interval 8 \
  --motion-gate-resize 96,160 \
  --motion-gate-grid-size 4 \
  --motion-gate-vpi-levels 1 \
  --motion-gate-vpi-quality low
```

Use unique prefixes/output directories for PyTorch, Metric3D TensorRT, DROID encoder TensorRT, and combined TensorRT runs. Never overwrite a control run.

Monitor resources throughout engine builds and SLAM runs. SmallCity is already close to physical RAM capacity. Stop unrelated Python jobs before building or benchmarking, and verify no previous `scripts/run.py` or `tegrastats` process remains afterward.

## Failure and Fallback Policy

- ONNX export failure: capture the first unsupported operation and a minimal reproducer; do not broadly rewrite the model.
- TensorRT parser failure: inspect the exact node and parser message before changing opset or precision.
- FP16 NaN/Inf: find the first divergent layer; constrain only that layer to FP32 if TensorRT supports it.
- Engine metadata mismatch: rebuild; do not ignore the mismatch.
- Engine load failure in strict mode: fail the run before processing frame 0.
- Production fallback: log once, instantiate PyTorch lazily, and record the actual backend in profiling output.
- Memory pressure: avoid dual model residency, reuse bindings/buffers, and unload export/build models before starting SLAM.
- DROID update export blocked by GraphAgg/scatter: keep GraphAgg in PyTorch and export only the convolutional core.
- DROID update profile too small: stop that branch and report BA/graph as the real bottleneck.

## Tests to Preserve

Before TensorRT changes, and again after each integration phase:

```bash
python -m unittest \
  tests.test_motion_gate \
  tests.test_dbaf_motion_gate_packet \
  tests.test_dbaf_runtime_profiling \
  tests.test_dbaf_frontend_runtime_profiling \
  tests.test_dbaf_frontend_iteration_config \
  tests.test_metric_model_depth_scale \
  tests.test_metric_depth_schedule \
  tests.test_motion_filter_lazy_depth \
  tests.test_eval_export \
  tests.test_smallcity_evaluator \
  tests.test_jetson_runtime_overrides
```

Add focused TensorRT tests without making CPU-only test collection import NVIDIA TensorRT eagerly. TensorRT imports must be lazy or cleanly skipped outside Jetson.

## Completion Definition

The TensorRT FP16 task is complete only when:

1. PyTorch remains the default and existing tests pass.
2. Engine build/export commands and sidecar metadata are reproducible.
3. Metric3D TensorRT passes module parity plus paired SmallCity-50 and SmallCity-200 evaluation.
4. DROID fnet/cnet TensorRT passes module parity plus the same paired evaluations.
5. DROID update is either implemented after passing the profiling gate or explicitly rejected with profiling evidence.
6. Runtime reports identify actual backends and cannot silently count a PyTorch fallback as TensorRT.
7. Resource logs show no new memory/hang regression.
8. The final report separates component speedup, end-to-end speedup, quality change, and memory cost.

## Relevant Existing Files

```text
docs/HANDOFF_TRACKING_OPT_2026-07-08.md
reports/before_after_optimization_metrics_20260709.md
scripts/frontend/dbaf.py
scripts/frontend/dbaf_frontend.py
scripts/frontend/covisible_graph.py
scripts/frontend/droid_net.py
scripts/frontend/motion_filter.py
scripts/frontend/modules/extractor.py
scripts/frontend/modules/gru.py
scripts/metric/metric_model.py
scripts/profiling/runtime_profiler.py
scripts/profiling/evaluate_smallcity_run.py
scripts/run.py
submodules/metric_modules/metric.py
submodules/metric_modules/metric3d/mono/model/
submodules/metric_modules/metric3d/mono/utils/do_test.py
configs/hierarchical/smallcity.yaml
ckpts/droid.pth
ckpts/metric_depth_vit_small_800k.pth
```

Final practical reminder: TensorRT is a means, not the metric. The current OFA gate costs only about 6.6 ms/frame, while DBA update and mapping dominate. Keep each conversion only when a paired run shows a real end-to-end gain on Jetson.

## Execution Update: Metric3D Static FP16 Module Gate (2026-07-11)

The offline Metric3D module gate is complete. This does not change the default backend and does not yet integrate TensorRT into the online SLAM path.

- The original opset-17 export remains the accepted ONNX Runtime parity artifact, but TensorRT 8.5.2 rejected its 43 `LayerNormalization` nodes. The complete parser failure is preserved in `reports/tensorrt_fp16/metric3d_trtexec_build.log`.
- An independent opset-16 export decomposed layer normalization into supported operators. Its ten-frame ONNX Runtime CUDA check passed with minimum cosine `0.999997518` and maximum mean relative error `0.000566827`.
- TensorRT built `engines/tensorrt/metric3d/metric3d_v2s_448x784_fp16.plan` successfully in 1,254 seconds. Engine SHA-256 is `fd08a2dd7f976ad6f23f0c0e3c5ed17b9622490e780091b15a6139ef5ea9b8d9`; size is 80,266,391 bytes.
- Strict engine loading verified exactly two static float32 bindings: input `rgb [1,3,448,784]` and output `depth [1,1,448,784]`.
- The first module benchmark exposed a non-contiguous NumPy-to-PyTorch input. The runner correctly rejected it; the original failure is preserved. A focused contiguous-input regression test was added before retrying.
- Retry1 passed all ten real SmallCity inputs. Worst raw cosine was `0.999975135`; worst raw mean relative error was `0.008364315`. All outputs were finite and shapes matched.
- Mean model latency improved from `409.0013 ms` in PyTorch to `164.6583 ms` in TensorRT: `2.484x` speedup and `59.74%` improvement. Estimated preprocessing/model/postprocessing pipeline latency improved from `432.0261 ms` to `187.2378 ms`.
- Build peak system RAM was 12,816 MB. The two-phase module benchmark, which releases the PyTorch model before loading TensorRT, peaked at 5,043 MB; allocated CUDA memory after release was 8,519,680 bytes.
- The isolated regression set passed 22 modules and 114 tests. Running every module in one Python process is not a supported isolation mode because earlier tests install temporary `frontend.*` module doubles that leak into later imports.

Primary evidence:

```text
reports/tensorrt_fp16/metric3d_opset16_onnx_validation.json
reports/tensorrt_fp16/metric3d_opset16_trtexec_build.log
reports/tensorrt_fp16/metric3d_opset16_trtexec_status.txt
reports/tensorrt_fp16/metric3d_trt_module_report.json
reports/tensorrt_fp16/metric3d_trt_module_report.md
reports/tensorrt_fp16/metric3d_trt_benchmark_retry1.log
reports/tensorrt_fp16/metric3d_trt_benchmark_status_retry1.txt
```

Decision: Metric3D FP16 is an online-integration candidate at the module gate. The next step is to add an explicit opt-in Metric3D TensorRT backend with strict backend reporting and lazy PyTorch fallback, then run paired SmallCity-50 PyTorch/TensorRT evaluation. Do not begin DROID encoder conversion or make TensorRT the default until that paired run is accepted.

## Execution Update: Metric3D Online SmallCity-50 Gate (2026-07-11)

The explicit online Metric3D TensorRT backend and paired SmallCity-50 gate are complete.

- `Metric_Model` now selects one backend. PyTorch remains the default. TensorRT constructs only lightweight Metric3D preprocessing configuration plus the engine; it does not construct the neural checkpoint model.
- CLI controls are `--metric-depth-backend`, `--metric-depth-engine`, and `--tensorrt-strict`. Runtime profile metadata records requested backend, actual backend, strict state, engine path, and fallback reason.
- Strict mode raises on engine or inference failure without constructing PyTorch. Non-strict mode releases TensorRT, logs the original failure once, and lazily constructs PyTorch once.
- The real-engine online smoke passed on a real SmallCity frame with CUDA float32 output shape `[344,616]` while `Metric3D.__init__` was patched to fail if called.
- The initial paired control attempt is preserved. It omitted the established `frontend_save_buffer=64` balanced-fast override, reverted to the 2500-frame default, reached 13,622 MB RAM, and failed allocation before TensorRT began. A script regression test was added before the retry.
- Retry1 completed both variants and both evaluators. Actual backends were `torch` and strict `tensorrt`; the TensorRT run had no fallback.
- ATE RMSE changed from 0.056609 m to 0.059765 m (+5.575%); mean PSNR changed by -0.063613 dB; mean SSIM changed by +0.000878. Keyframe lists were byte-identical and exported sample count changed from 12 to 11 (-8.333%).
- Mean `metric_depth_model` latency improved from 474.322 ms to 174.613 ms (63.187%). `frame_total` improved from 113.626 s to 107.729 s (5.190%). Peak RAM changed from 10,007 MB to 10,026 MB (+19 MB).
- Both runs shut down cleanly with no residual `scripts/run.py` or `tegrastats` process.

Decision: **GO for a new paired SmallCity-200 Metric3D TensorRT decision run.** PyTorch remains the default, and this result does not authorize DROID encoder work. Full evidence is in `reports/tensorrt_fp16/metric3d_smallcity50_online_decision.md`.

## Execution Update: Metric3D Online SmallCity-200 Decision (2026-07-11)

The fresh paired SmallCity-200 decision run is complete. Both runs and evaluators exited successfully, the tracking keyframe lists were byte-identical, and the strict TensorRT run used the actual TensorRT backend without fallback.

- ATE Sim3 RMSE changed from 0.535978 m to 0.568415 m (+6.052%): pass.
- Mean PSNR changed from 14.133403 dB to 13.796926 dB (-0.336477 dB): pass.
- Mean SSIM changed from 0.685839 to 0.663496 (-0.022343): **fail**, because the allowed drop is 0.02.
- Exported sample count changed from 39 to 38 (-2.564%): pass.
- Mean `metric_depth_model` latency improved from 438.087 ms to 173.890 ms (60.307%): pass.
- `frame_total` improved from 313.649 s to 304.151 s (3.028%): pass.
- Peak RAM decreased from 10,943 MB to 10,592 MB (-351 MB): pass.
- Both variants shut down cleanly with no residual `scripts/run.py` or `tegrastats` process.

Decision: **NO-GO for recommending or enabling Metric3D TensorRT by default.** PyTorch remains the default. Keep TensorRT explicit opt-in only and stop before DROID encoder conversion. The strict decision report is `reports/tensorrt_fp16/metric3d_smallcity200_online_decision.md`.

### User-approved SSIM exception

After reviewing the SmallCity-200 evidence, the user explicitly accepted the `0.022343` mean SSIM drop. Record the result as a **conditional GO for continuing to the DROID fnet/cnet phase**, while preserving the original strict `NO-GO` above.

- The original 0.02 threshold and measured failure remain in the report; this is an explicit human risk acceptance, not a retroactive numerical pass.
- On the 26 common exported frames, quick mean SSIM dropped by 0.0142, within 0.02.
- All ATE, PSNR, keyframe, sample-count, component-latency, `frame_total`, RAM, backend, and shutdown gates passed.
- PyTorch remains the default. Metric3D TensorRT remains explicit opt-in.
- The exception authorizes only the DROID fnet/cnet audit and gated implementation. It does not relax the DROID update-core 1% parity threshold and does not authorize INT8, resolution changes, OFA changes, or mapping changes.

## Execution Update: DROID fnet/cnet Static FP16 Module Gate (2026-07-11)

The independent fnet/cnet offline module gate is complete. PyTorch remains the default, no online adapter was added, and the two encoder decisions are intentionally independent.

- `BasicEncoder.forward_core(x4d)` now contains the unchanged convolutional body. The existing 5D `forward()` contract only flattens/restores batch and frame dimensions. Exact legacy-order tests passed for fnet and cnet at `(B,N)=(1,1),(1,4),(2,3)`.
- Static opset-17 ONNX graphs use input `image [1,3,344,616]` float32 and output `features` float16. The 20 admitted-frame ONNX Runtime CUDA gate passed for both encoders. fnet reached minimum cosine `0.999999649` and maximum MRE `0.004627710`; cnet reached cosine approximately `1.0` and MRE `0.0`.
- TensorRT 8.5 Python deserialization initially failed only for fnet because `InstanceNormalization_TRT` had not been registered. The shared runtime and metadata paths now call `init_libnvinfer_plugins` before constructing `Runtime`; focused tests require this ordering. Both plans now pass strict hash, version, binding, and deserialization checks.
- fnet plan: `engines/tensorrt/droid/droid_fnet_b1_344x616_fp16.plan`, SHA-256 `f4e167f52bb69382fd9e716272ba367084c90b179f511ce1f49fe667bc90abb3`, 1,902,249 bytes, build time 141 seconds.
- cnet plan: `engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan`, SHA-256 `6b90bb5c1d7ccd9d730f587df69adf269a8e1e10d71576cbe93c57008f90218e`, 1,844,481 bytes, build time 28 seconds.
- fnet TensorRT latency improved from `11.4291 ms` to `8.2197 ms` (`1.390x`, `28.08%`), but parity failed on every real input: minimum cosine `0.007816976` and maximum MRE `1.160341`. The failure is isolated to the TensorRT graph containing 15 `InstanceNormalization` nodes; the same ONNX graph passes ONNX Runtime, while the shared runner and non-InstanceNorm cnet graph pass. Do not integrate fnet or relax its parity threshold.
- cnet passed all 20 real inputs with minimum cosine `0.999999341`, maximum MRE `0.004004960`, and finite float16 outputs of exact shape `[1,256,43,77]`. Mean latency improved from `8.6458 ms` to `4.2257 ms` (`2.046x`, `51.12%`). This is a module result only; the previously measured theoretical SmallCity-200 maximum contribution is about `0.232%` of `frame_total`.
- Two-phase benchmarks released the PyTorch encoder before loading TensorRT and reported zero allocated CUDA bytes after release. Peak system RAM was 6,042 MB for fnet and 5,714 MB for cnet; peak swap was 312 MB for both.
- The final isolated regression set passed 29 modules and 172 tests. `pip check`, Python syntax, patch-format checks, artifact existence, strict loading of both plans, and residual-process checks all passed.

Decision: **cnet only is an online-integration candidate. fnet is NO-GO.** The next phase may add an explicit opt-in cnet-only TensorRT adapter and paired SmallCity gates. fnet must remain PyTorch, and no paired SLAM run has been authorized or executed by this offline module plan.

Primary evidence:

```text
reports/tensorrt_fp16/droid_fnet_onnx_validation.json
reports/tensorrt_fp16/droid_cnet_onnx_validation.json
reports/tensorrt_fp16/droid_fnet_trtexec_build.log
reports/tensorrt_fp16/droid_cnet_trtexec_build.log
reports/tensorrt_fp16/droid_fnet_trt_module_report.json
reports/tensorrt_fp16/droid_fnet_trt_module_report.md
reports/tensorrt_fp16/droid_cnet_trt_module_report.json
reports/tensorrt_fp16/droid_cnet_trt_module_report.md
reports/tensorrt_fp16/droid_encoder_isolated_regression.log
reports/tensorrt_fp16/droid_encoder_isolated_regression_status.txt
```

## Execution Update: DROID cnet Online SmallCity-50 Gate (2026-07-12)

The explicit opt-in cnet-only TensorRT adapter and paired SmallCity-50 gate are complete. The pair kept Metric3D on PyTorch in both variants, so cnet was the only backend difference.

- New controls are `--droid-cnet-backend {torch,tensorrt}` and `--droid-cnet-engine`; defaults remain `torch` and `null`. The shared `--tensorrt-strict` policy applies.
- The adapter preserves the existing `[1,1,3,344,616] -> [1,1,256,43,77]` contract and leaves `MotionFilter` split/tanh/relu behavior unchanged.
- Strict mode detaches and releases the PyTorch cnet before loading TensorRT. Non-strict mode retains one lazy PyTorch fallback, logs the actual backend once, and updates runtime profile metadata on fallback.
- The real-engine smoke passed with finite `[1,128,43,77]` context `net/inp` tensors and actual strict TensorRT execution.
- The isolated regression set passed 30 modules and 187 tests before the paired run.
- Both SmallCity-50 runs, backend checks, evaluators, and tracking comparison exited successfully. The candidate recorded strict actual backend `tensorrt`, engine SHA-256 `6b90bb5c1d7ccd9d730f587df69adf269a8e1e10d71576cbe93c57008f90218e`, and no fallback.
- The 16-entry keyframe lists were byte-identical, but exported frame IDs diverged and sample count changed from 10 to 9.
- ATE Sim3 RMSE changed from `0.050925 m` to `0.080243 m` (`+0.029318 m`), exceeding the `0.01 m` limit.
- Mean PSNR changed by `-0.394835 dB`, exceeding the `0.05 dB` limit. Mean SSIM changed by `-0.013848`, exceeding the `0.005` limit.
- Mean `motion_filter_context_encoder` latency improved from `11.4876 ms` to `6.1374 ms` (`46.57%`), but `frame_total` improved by only `0.072%` (`118.9955 s` to `118.9100 s`). Peak RAM increased by 148 MB to 11,630 MB.

Decision: **NO-GO for cnet TensorRT online integration.** Keep cnet and fnet on PyTorch, retain TensorRT as offline evidence only, and stop before SmallCity-200. The strict decision report is `reports/tensorrt_fp16/droid_cnet_smallcity50_online_decision.md`.

## Execution Update: DROID Update-Core Online Speed-First Gate (2026-07-12)

The existing delta-head-FP32 update-core engine was integrated behind explicit `--droid-update-backend` and `--droid-update-engine` controls. PyTorch remains the default. `UpdateModule.forward()` and PyTorch GraphAgg remain unchanged; only flattened `forward_core` dispatches to TensorRT.

- Strict mode releases `corr_encoder`, `flow_encoder`, GRU, delta, and weight modules before loading TensorRT while retaining `agg`. Non-strict mode retains one logged PyTorch fallback.
- Runtime metadata records requested/actual backend, strict state, engine hash, precision `fp16_delta_fp32`, and fallback reason.
- Real-engine smoke passed at E=1,4,16,32,48. Online AMP and strided CovisibleGraph inputs are normalized to contiguous FP32 bindings, and outputs restore the incoming recurrent dtype.
- The first paired attempt failed strictly on AMP FP16 input. Retry1 failed strictly on non-contiguous CovisibleGraph flow. Both failures are preserved, and each received a regression test before retrying.
- Retry2 completed both variants, evaluators, backend verification, and tracking comparison. The candidate used actual strict TensorRT with no fallback.
- `covisible_graph_update_op` improved by `44.18%` (`32.5726 s` to `18.1823 s`). `frame_total` improved by `12.22%` (`119.1256 s` to `104.5651 s`), passing the required 5% speed gate.
- ATE Sim3 RMSE improved by `12.76%`, but mean PSNR dropped `1.109833 dB`, mean SSIM dropped `0.071798`, and exported sample count changed from 12 to 10 (`-16.67%`). These exceed the approved relaxed limits of `0.5 dB`, `0.03`, and `10%`.
- Peak RAM increased from 11,379 MB to 11,604 MB; swap remained 313 MB. Keyframe lists were byte-identical, while export IDs diverged.

Decision: **NO-GO under the approved relaxed accuracy policy.** Keep update-core on PyTorch by default and stop before the Metric3D+update combined pair and SmallCity-200. The decision report is `reports/tensorrt_fp16/droid_update_smallcity50_online_decision.md`.

## Execution Update: User-Authorized Metric3D + Update-Core Trial (2026-07-12)

After the update-only NO-GO, the user explicitly authorized trying the combined branch. This override authorized one evidence-gathering SmallCity-50 pair; it did not change defaults or retroactively pass the update-only gate.

- Both variants, backend checks, evaluators, and comparison exited successfully. Runtime metadata confirmed actual strict TensorRT for Metric3D and delta-FP32 update core with no fallback; fnet and cnet remained PyTorch.
- `metric_depth_model` improved by `62.746%` (`9.011 s` to `3.357 s`) and `covisible_graph_update_op` improved by `43.157%` (`31.972 s` to `18.174 s`).
- `frame_total` improved by `14.785%` (`115.677 s` to `98.574 s`); driver elapsed improved by `11.811%` (`127 s` to `112 s`).
- Mean PSNR dropped `0.147702 dB`, mean SSIM dropped `0.008214`, and exported sample count changed from 10 to 11 (`+10.0%`), all within the previous relaxed limits.
- ATE Sim3 RMSE changed from `0.057683 m` to `0.072073 m` (`+24.947%`), exceeding the previous 20% relaxed limit.
- The 16-entry keyframe lists were byte-identical, but export IDs diverged. Peak RAM increased from 10,491 MB to 11,576 MB; swap remained 313 MB.

Decision: the combination is technically viable and is the strongest speed-first online TensorRT candidate so far, but remains **NO-GO under the previous relaxed gate** due to ATE. It may be considered only as explicit opt-in if the user explicitly accepts about 25% ATE regression and roughly 1.1 GB additional peak RAM. Defaults remain PyTorch. Full evidence is in `reports/tensorrt_fp16/metric3d_update_smallcity50_online_decision.md`.
