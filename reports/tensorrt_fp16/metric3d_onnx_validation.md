# Metric3D Static ONNX Real-Input Parity

- ONNX: `engines/tensorrt/metric3d/metric3d_v2s_448x784.onnx`
- SHA256: `fd805c344ec73305b658bc568cacd62832551da479a53d89134a72bc6c2bc911`
- Provider: `CUDAExecutionProvider`
- Static input: `[1, 3, 448, 784]`
- Static output: `[1, 1, 448, 784]`
- Opset: `17`
- ONNX size: `151,143,881` bytes (144.1 MiB)
- Accepted: `True`

## Export and resource evidence

- Export: 13.46 s; ORT session plus ten-frame validation: 69.99 s; total
  process time: 96.96 s (controller wall time 101 s).
- ONNX Runtime available providers included TensorRT, CUDA, and CPU; the session
  enabled `CUDAExecutionProvider` followed by `CPUExecutionProvider`.
- Peak system RAM was 11,726 MB and peak system swap was 311 MB across 101
  `tegrastats` samples.
- The initial export with constant folding enabled failed inside the Jetson
  PyTorch 2.1 ONNX fold pass because `index_select` received a CUDA tensor and
  CPU index. An isolated no-fold probe exported the complete graph and passed
  ONNX checker/shape inference, so the final export disables only this exporter
  optimization. Model computation and runtime behavior were not changed.

| Frame | Boundary | Cosine | Mean relative error | Valid | Finite |
|---:|:---|---:|---:|---:|:---:|
| 0 | raw | 0.999999870 | 0.000566 | 351232 | True |
| 0 | final | 0.999999904 | 0.000566 | 211904 | True |
| 5 | raw | 0.999999589 | 0.000338 | 351232 | True |
| 5 | final | 0.999999690 | 0.000337 | 211904 | True |
| 10 | raw | 0.999999945 | 0.000299 | 351232 | True |
| 10 | final | 0.999999950 | 0.000298 | 211904 | True |
| 15 | raw | 0.999999911 | 0.000357 | 351232 | True |
| 15 | final | 0.999999933 | 0.000357 | 211904 | True |
| 20 | raw | 0.999999922 | 0.000234 | 351232 | True |
| 20 | final | 0.999999939 | 0.000234 | 211904 | True |
| 25 | raw | 0.999999918 | 0.000134 | 351232 | True |
| 25 | final | 0.999999934 | 0.000134 | 211904 | True |
| 30 | raw | 0.999999834 | 0.000203 | 351232 | True |
| 30 | final | 0.999999876 | 0.000203 | 211904 | True |
| 35 | raw | 0.999999953 | 0.000227 | 351232 | True |
| 35 | final | 0.999999957 | 0.000227 | 211904 | True |
| 40 | raw | 0.999999765 | 0.000239 | 351232 | True |
| 40 | final | 0.999999821 | 0.000238 | 211904 | True |
| 45 | raw | 0.999997506 | 0.000374 | 351232 | True |
| 45 | final | 0.999997819 | 0.000373 | 211904 | True |

## Aggregates

- Minimum cosine: `0.999997506`
- Maximum mean relative error: `0.000566`

## Decision

Static Metric3D ONNX parity passed; TensorRT build is authorized.
