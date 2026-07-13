# DROID cnet TensorRT FP16 Module Report

- Engine: `engines/tensorrt/droid/droid_cnet_b1_344x616_fp16.plan`
- Checkpoint: `ckpts/droid.pth`
- Timing: 10 warmup, 50 measured iterations per frame
- Backend separation: PyTorch encoder released before TensorRT engine load

| Frame | PyTorch (ms) | TensorRT (ms) | Cosine | MRE | Finite |
|---:|---:|---:|---:|---:|:---:|
| 0 | 9.6903 | 4.2253 | 0.999999423 | 0.003639 | yes |
| 8 | 9.0376 | 4.2259 | 0.999999417 | 0.003602 | yes |
| 11 | 9.8552 | 4.2255 | 0.999999437 | 0.003540 | yes |
| 15 | 8.4937 | 4.2287 | 0.999999437 | 0.003685 | yes |
| 19 | 8.4894 | 4.2269 | 0.999999402 | 0.003764 | yes |
| 28 | 8.4902 | 4.2259 | 0.999999372 | 0.003844 | yes |
| 38 | 8.4884 | 4.2259 | 0.999999341 | 0.004005 | yes |
| 44 | 8.4894 | 4.2256 | 0.999999395 | 0.003713 | yes |
| 56 | 8.4902 | 4.2244 | 0.999999380 | 0.003782 | yes |
| 72 | 8.4899 | 4.2256 | 0.999999418 | 0.003691 | yes |
| 82 | 8.4891 | 4.2244 | 0.999999355 | 0.003865 | yes |
| 104 | 8.4889 | 4.2243 | 0.999999415 | 0.003689 | yes |
| 112 | 8.4898 | 4.2262 | 0.999999379 | 0.003829 | yes |
| 129 | 8.4899 | 4.2256 | 0.999999427 | 0.003797 | yes |
| 144 | 8.4907 | 4.2254 | 0.999999390 | 0.003756 | yes |
| 158 | 8.4909 | 4.2259 | 0.999999395 | 0.003750 | yes |
| 165 | 8.4907 | 4.2264 | 0.999999373 | 0.003713 | yes |
| 170 | 8.4902 | 4.2271 | 0.999999370 | 0.003949 | yes |
| 179 | 8.4907 | 4.2249 | 0.999999369 | 0.003762 | yes |
| 195 | 8.4915 | 4.2244 | 0.999999366 | 0.003961 | yes |

## Aggregates

- PyTorch mean/median/p90: 8.6458 / 8.4902 / 9.1028 ms
- TensorRT mean/median/p90: 4.2257 / 4.2256 / 4.2269 ms
- Speedup: 2.046x
- Improvement: 51.12%
- Minimum cosine: 0.999999341
- Maximum mean relative error: 0.004005

## Decision

integration candidate: parity passed and mean latency improved by at least 20%
