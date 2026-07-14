# DBA Memory Calibration

## Environment and evidence

- Capture boundary: `corr_update_aggregation`
- Host platform: `Linux-5.10.216-tegra-aarch64-with-glibc2.17`
- Torch: `2.1.0a0+41361538.nv23.06`
- CUDA: `11.4`
- CUDA device: `Orin`
- Feature shape: `[43, 77]`
- Frontend image size: `[344, 616]`
- Source: `/home/jetson/VINGS-Mono/output/cuda_graph_dba_phase05_20260714/smallcity200/07-14-18-56-hierarchical-smallcit-cuda_graph_dba_phase05_smallcity200/runtime_profile_details.jsonl` (`9408f0e1db1fa48667627a537073a02910db86f9038c752790b32da5447f835b`)
- Source: `/home/jetson/VINGS-Mono/output/cuda_graph_dba_phase05_20260714/hotel344/07-14-18-58-rtgslam-hote-cuda_graph_dba_phase05_hotel344/runtime_profile_details.jsonl` (`7561852b196130164373dbce9a6cbe90cc87a9f397317002c2d3cdb709f73811`)

## Signature coverage

- Sampled signatures: 16
- Memory samples: 48
- Weighted observed-call coverage: 100.00%

## Boundary model

- `corr_pyramid_bytes`: 690383232 to 1380766464 bytes
- `sampled_corr_bytes`: 31149888 to 62299776 bytes
- `recurrent_bytes`: 40685568 to 81371136 bytes
- `motion_bytes`: 1271424 to 2542848 bytes
- `delta_bytes`: 635712 to 1271424 bytes
- `weight_bytes`: 635712 to 1271424 bytes
- `damping_bytes`: 33110 to 79464 bytes
- `upmask_bytes`: 19071360 to 45771264 bytes
- `aggregation_bytes`: 4238080 to 10171392 bytes
- `ba_inputs_bytes`: 0 to 0 bytes

## Coefficients and safety margin

- `intercept`: 6189352.914923
- `active_edges`: 15738452.193094
- `ba_edges`: 0.000000
- `source_poses`: 0.000000
- `pose_window`: 829676.297864
- Margin: max(67108864 bytes, 10% of modeled workspace)

## Validation

- Valid: **True**
- Weighted MAPE: 0.54%
- Training weighted MAPE: 0.54%
- Held-out weighted MAPE: 0.51%
- Underpredicted signatures: none
- Unstable signatures: none
