# DROID fnet Static ONNX Parity

- ONNX: `engines/tensorrt/droid/droid_fnet_b1_344x616.onnx`
- ONNX SHA-256: `0e27080125cbe8943f1d90a1570d0143c090fd0c94b40d16384ee20f5b212e1b`
- Checkpoint SHA-256: `46476ef64cde45a97504910d6f3de2eef7b398ec1c6e4e668815c29076024526`
- Provider: `CUDAExecutionProvider`
- Accepted: `true`

| Frame | Shape | Dtype | Cosine | Mean relative error | Finite |
|---:|---|---|---:|---:|:---:|
| 0 | 1x128x43x77 | float16 | 0.999999678 | 0.003886 | yes |
| 8 | 1x128x43x77 | float16 | 0.999999671 | 0.003863 | yes |
| 11 | 1x128x43x77 | float16 | 0.999999678 | 0.003789 | yes |
| 15 | 1x128x43x77 | float16 | 0.999999649 | 0.004118 | yes |
| 19 | 1x128x43x77 | float16 | 0.999999678 | 0.003779 | yes |
| 28 | 1x128x43x77 | float16 | 0.999999684 | 0.003953 | yes |
| 38 | 1x128x43x77 | float16 | 0.999999694 | 0.003707 | yes |
| 44 | 1x128x43x77 | float16 | 0.999999684 | 0.003908 | yes |
| 56 | 1x128x43x77 | float16 | 0.999999680 | 0.003919 | yes |
| 72 | 1x128x43x77 | float16 | 0.999999695 | 0.003726 | yes |
| 82 | 1x128x43x77 | float16 | 0.999999675 | 0.003961 | yes |
| 104 | 1x128x43x77 | float16 | 0.999999695 | 0.003755 | yes |
| 112 | 1x128x43x77 | float16 | 0.999999677 | 0.004628 | yes |
| 129 | 1x128x43x77 | float16 | 0.999999655 | 0.004523 | yes |
| 144 | 1x128x43x77 | float16 | 0.999999691 | 0.003685 | yes |
| 158 | 1x128x43x77 | float16 | 0.999999678 | 0.003983 | yes |
| 165 | 1x128x43x77 | float16 | 0.999999694 | 0.003777 | yes |
| 170 | 1x128x43x77 | float16 | 0.999999693 | 0.003756 | yes |
| 179 | 1x128x43x77 | float16 | 0.999999698 | 0.003796 | yes |
| 195 | 1x128x43x77 | float16 | 0.999999668 | 0.003927 | yes |

## Aggregates

- Minimum cosine: 0.999999649
- Maximum mean relative error: 0.004628

TensorRT build is authorized.
