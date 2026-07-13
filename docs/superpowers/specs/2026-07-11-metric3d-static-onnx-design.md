# Metric3D Static ONNX Export and Real-Input Parity Design

Date: 2026-07-11

## Goal

Export the Metric3D v2-S neural depth model at the balanced-fast static input
shape and verify it with ONNX Runtime CUDA on ten real SmallCity images. This
phase establishes ONNX correctness only; it does not build TensorRT or change
the online Metric3D backend.

## Scope

- Use `ckpts/metric_depth_vit_small_800k.pth` and the existing Metric3D v2-S
  configuration.
- Apply `metric_depth_scale=0.75`, producing the fixed model input shape
  `1x3x448x784`.
- Preserve the current Metric3D preprocessing and postprocessing exactly.
- Do not modify the Metric3D submodule, runtime backend selection, frontend
  resolution, OFA, mapping, or DROID behavior.
- Do not build a TensorRT engine in this phase.
- Do not stage, commit, reset, clean, or remove unrelated worktree content.

## Export Boundary

The export wrapper accepts only the normalized RGB tensor and returns only the
depth prediction:

```python
predictor.model_.module.depth_model(input=rgb_input)["prediction"]
```

The dense pipeline's `forward(input, **kwargs)` uses the RGB input directly;
the current v2-S depth path does not consume the camera-model stacks. The ONNX
graph therefore has:

- input name `rgb`, shape `1x3x448x784`;
- output name `depth`;
- opset 17;
- static batch and spatial dimensions;
- constant folding disabled for this Jetson PyTorch 2.1 exporter.

The initial real export proved that PyTorch's ONNX constant-fold pass mixes a
CUDA tensor with a CPU `index_select` index and aborts after successful model
tracing. An otherwise identical no-fold probe exported the complete model and
passed ONNX checker and shape inference. Disabling this exporter optimization
does not change model computation; ten-frame output parity remains the required
proof of graph equivalence.

Before export, one PyTorch forward pass creates inference-time lazy buffers,
including `depth_expectation_anchor`. The runtime model is not modified. Any
export-only wrapper is a separate `nn.Module` around the loaded model.

## Real-Input Dataset

Use these ten SmallCity-50 color frames:

```text
00000, 00005, 00010, 00015, 00020,
00025, 00030, 00035, 00040, 00045
```

Images come from `data/smallcity_subset_50/small_city/color/`. Preprocessing
uses the same `Metric.preprocess()` implementation and SmallCity intrinsics as
`Metric_Model`, so PyTorch and ONNX Runtime receive identical model tensors.

## Validation Layers

Validation occurs at two boundaries for every selected frame:

1. **Raw neural output:** compare the PyTorch export wrapper with ONNX Runtime
   CUDA before clamp or resize.
2. **Final depth:** send both raw outputs through the same existing clamp and
   OpenCV resize path, then compare the resulting original-resolution depth.

For each comparison:

- output shapes must match exactly;
- all compared values must be finite;
- the valid mask is finite PyTorch depth greater than zero;
- the valid mask must contain at least one value;
- cosine similarity over valid values must be at least 0.999;
- mean relative error over valid values must be at most 0.02, using
  `max(abs(expected), 1e-3)` as the denominator floor.

The report records per-frame raw and final metrics plus global minimum cosine
and maximum mean relative error. ONNX Runtime must actually enable
`CUDAExecutionProvider`; silent CPU-only validation is rejected.

## Artifacts

- `engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx`
- `reports/tensorrt_fp16/metric3d_onnx_validation.json`
- `reports/tensorrt_fp16/metric3d_onnx_validation.md`
- a raw export/validation log under `reports/tensorrt_fp16/`

The generated ONNX artifact is not staged or committed.

## Failure Handling

- If checkpoint loading fails, record the exact path/error and stop.
- If export fails, record the first unsupported operation and producer stack.
  Do not broadly rewrite the model or change opset by trial and error.
- If the failure is caused by inference-only diagnostics or lazy buffer
  creation, isolate the smallest export-only identity change and require a new
  PyTorch equivalence test before retrying.
- If the complete depth model remains blocked, partition experiments may be
  used only for diagnosis. A partial encoder export does not satisfy this
  phase.
- If ONNX checker, shape inference, CUDA provider selection, shape, finite, or
  parity checks fail, retain diagnostic evidence and do not proceed to the
  TensorRT build phase.

## Testing

- CPU-safe tests cover wrapper output selection, fixed export names/shapes,
  parity calculations, valid-mask handling, provider enforcement, sample frame
  selection, and report decision logic without loading Metric3D.
- Jetson validation loads the real checkpoint once, exports once, and evaluates
  all ten frames in one process to avoid repeated model memory cost.
- Existing Metric3D depth-scale/profiling and DROID/TensorRT regression modules
  remain isolated and unchanged.
