# Metric3D Static ONNX Real-Input Parity

- ONNX: `engines/tensorrt/metric3d/metric3d_v2s_448x784_opset16.onnx`
- SHA256: `601152c515e80fc4d1a92ee7d9f13c6305ede6d4d88d2f15d1f560a4aa574d2b`
- Provider: `CUDAExecutionProvider`
- Static input: `[1, 3, 448, 784]`
- Accepted: `True`

| Frame | Boundary | Cosine | Mean relative error | Valid | Finite |
|---:|:---|---:|---:|---:|:---:|
| 0 | raw | 0.999999866 | 0.000567 | 351232 | True |
| 0 | final | 0.999999903 | 0.000567 | 211904 | True |
| 5 | raw | 0.999999583 | 0.000339 | 351232 | True |
| 5 | final | 0.999999685 | 0.000339 | 211904 | True |
| 10 | raw | 0.999999945 | 0.000298 | 351232 | True |
| 10 | final | 0.999999950 | 0.000298 | 211904 | True |
| 15 | raw | 0.999999912 | 0.000355 | 351232 | True |
| 15 | final | 0.999999933 | 0.000355 | 211904 | True |
| 20 | raw | 0.999999918 | 0.000236 | 351232 | True |
| 20 | final | 0.999999938 | 0.000235 | 211904 | True |
| 25 | raw | 0.999999918 | 0.000136 | 351232 | True |
| 25 | final | 0.999999935 | 0.000135 | 211904 | True |
| 30 | raw | 0.999999831 | 0.000205 | 351232 | True |
| 30 | final | 0.999999875 | 0.000205 | 211904 | True |
| 35 | raw | 0.999999953 | 0.000229 | 351232 | True |
| 35 | final | 0.999999957 | 0.000228 | 211904 | True |
| 40 | raw | 0.999999769 | 0.000240 | 351232 | True |
| 40 | final | 0.999999824 | 0.000239 | 211904 | True |
| 45 | raw | 0.999997518 | 0.000374 | 351232 | True |
| 45 | final | 0.999997830 | 0.000374 | 211904 | True |

## Aggregates

- Minimum cosine: `0.999997518`
- Maximum mean relative error: `0.000567`

## Decision

Static Metric3D ONNX parity passed; TensorRT build is authorized.
