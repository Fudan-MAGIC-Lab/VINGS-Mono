# Jetson Gaussian Densification Design

## Goal

Optimize the 2D Gaussian mapping stage of VINGS-Mono for Jetson Orin NX by replacing the fixed per-keyframe Gaussian addition count with a resource-aware adaptive densification policy.

The project should remain small enough for an undergraduate project: one focused mapping change, clear before/after measurements, and no CUDA kernel rewrite.

## Motivation

Current Jetson runs show that mapping becomes slower and memory pressure grows as the Gaussian count increases. Existing code adds up to `40000` Gaussians for every new keyframe in `scripts/gaussian/gaussian_model.py`, regardless of map size, current coverage, or Jetson resource limits.

This fixed policy is simple, but it can overbuild the map. More Gaussians increase the cost of later rasterization, training, visualization, and export.

## Scope

In scope:

- Add an adaptive policy for the number of Gaussians sampled in `GaussianModel.add_new_frame`.
- Base the policy on current Gaussian count and image coverage from `pred_accum`.
- Add lightweight logging for each new keyframe: Gaussian count before, requested sample budget, actual added count, and Gaussian count after.
- Keep the policy configurable from YAML or CLI-friendly config fields.
- Compare baseline and optimized runs on the Jetson Hotel demo.

Out of scope:

- Rewriting the rasterizer CUDA kernels.
- Changing the visual frontend.
- Changing loop closure.
- Replacing dynamic object detection.
- Treating visualization disablement as the main contribution.

## Proposed Policy

The mapper will compute a per-keyframe budget before calling `get_pointcloud`.

Inputs:

- `current_gaussians`: current `self._xyz.shape[0]`.
- `uncovered_ratio`: ratio of pixels with low accumulation in `pred_accum`.
- Config thresholds and budgets.

Default budget schedule:

| Current Gaussian Count | Max New Gaussians |
| --- | ---: |
| `< 150k` | `40000` |
| `150k - 300k` | `25000` |
| `300k - 600k` | `12000` |
| `> 600k` | `6000` |

Coverage adjustment:

- If the image is mostly already covered, reduce the budget.
- If the low-accumulation area is large, keep more of the stage budget.
- Clamp the final budget to a minimum, such as `2000`, so mapping can still fill genuinely new areas.

## Data Flow

Existing flow:

```text
new keyframe
  -> render current map
  -> compute accumulation and errors
  -> get_pointcloud(..., 40000)
  -> concatenate new Gaussians
  -> train mapping iterations
```

New flow:

```text
new keyframe
  -> render current map
  -> compute accumulation and errors
  -> estimate uncovered_ratio
  -> choose adaptive Gaussian budget
  -> get_pointcloud(..., adaptive_budget)
  -> concatenate new Gaussians
  -> log densification statistics
  -> train mapping iterations
```

## Evaluation

Primary metrics:

- Total runtime on Hotel demo.
- Average seconds per frame.
- Peak RAM and swap from `tegrastats`.
- Final Gaussian count.
- Whether the run completes without OOM.

Secondary quality checks:

- Saved RGB/depth/normal/uncertainty preview images.
- Final PLY size and rough visual inspection.
- If available, compare PSNR or image statistics from existing reporting scripts.

## Expected Result

The optimized run should reduce Gaussian growth and memory pressure while preserving acceptable map quality. A successful undergraduate-level outcome is:

- lower final Gaussian count,
- lower or more stable memory use,
- faster runtime or fewer late-stage slowdowns,
- no major visual collapse in the Hotel demo.

## Implementation Notes

The first implementation should be conservative:

- Keep the old fixed `40000` behavior as the default unless `mapping_budget.enabled` is true.
- Add a small helper method on `GaussianModel`, such as `_select_new_gaussian_budget`.
- Avoid changing optimizer, loss, frontend, loop, metric, or rasterizer behavior.
- Add unit tests for the budget selection logic without requiring CUDA.
