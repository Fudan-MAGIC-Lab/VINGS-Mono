# DROID fnet TensorRT FP16 Module Report

- Engine: `engines/tensorrt/droid/droid_fnet_b1_344x616_fp16.plan`
- Checkpoint: `ckpts/droid.pth`
- Timing: 10 warmup, 50 measured iterations per frame
- Backend separation: PyTorch encoder released before TensorRT engine load

| Frame | PyTorch (ms) | TensorRT (ms) | Cosine | MRE | Finite |
|---:|---:|---:|---:|---:|:---:|
| 0 | 12.7211 | 8.2219 | 0.008613421 | 1.100136 | yes |
| 8 | 12.4072 | 8.2227 | 0.010026926 | 1.098500 | yes |
| 11 | 12.7795 | 8.2258 | 0.010374928 | 1.095483 | yes |
| 15 | 11.2373 | 8.2210 | 0.008940809 | 1.101931 | yes |
| 19 | 11.2120 | 8.2159 | 0.008374492 | 1.097476 | yes |
| 28 | 11.2128 | 8.2226 | 0.008912070 | 1.111495 | yes |
| 38 | 11.2142 | 8.2183 | 0.007816976 | 1.096291 | yes |
| 44 | 11.2114 | 8.2196 | 0.008427582 | 1.102866 | yes |
| 56 | 11.2122 | 8.2396 | 0.008514297 | 1.103758 | yes |
| 72 | 11.2121 | 8.2175 | 0.009145913 | 1.099326 | yes |
| 82 | 11.2111 | 8.2222 | 0.008433747 | 1.101146 | yes |
| 104 | 11.2135 | 8.2210 | 0.008210832 | 1.096234 | yes |
| 112 | 11.2138 | 8.2156 | 0.008709660 | 1.160341 | yes |
| 129 | 11.2164 | 8.2129 | 0.009310000 | 1.130560 | yes |
| 144 | 11.2176 | 8.2125 | 0.008680443 | 1.096740 | yes |
| 158 | 11.2127 | 8.2144 | 0.007838018 | 1.101538 | yes |
| 165 | 11.2146 | 8.2204 | 0.008187778 | 1.101408 | yes |
| 170 | 11.2149 | 8.2133 | 0.008162188 | 1.094592 | yes |
| 179 | 11.2352 | 8.2172 | 0.008815871 | 1.099186 | yes |
| 195 | 11.2125 | 8.2200 | 0.008022253 | 1.101047 | yes |

## Aggregates

- PyTorch mean/median/p90: 11.4291 / 11.2140 / 12.4386 ms
- TensorRT mean/median/p90: 8.2197 / 8.2198 / 8.2230 ms
- Speedup: 1.390x
- Improvement: 28.08%
- Minimum cosine: 0.007816976
- Maximum mean relative error: 1.160341

## Decision

do not integrate: module parity thresholds were not met
