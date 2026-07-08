#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
shared_root="$repo_root"

if [ "$(basename "$(dirname "$repo_root")")" = ".worktrees" ]; then
  shared_root="$(cd "$repo_root/../.." && pwd)"
fi

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

dataset_root="${VINGS_DATASET_ROOT:-$shared_root/data/hotel}"
output_dir="${VINGS_OUTPUT_DIR:-$shared_root/output}"
frontend_weight="${VINGS_DROID_WEIGHT:-$shared_root/ckpts/droid.pth}"
lightglue_weight_dir="${VINGS_LIGHTGLUE_DIR:-$shared_root/ckpts/lightglue}"
loop_onnx_provider="${VINGS_LOOP_ONNX_PROVIDER:-cpu}"
frontend_image_size="${VINGS_FRONTEND_IMAGE_SIZE:-256,448}"
training_iters="${VINGS_TRAINING_ITERS:-30}"
adaptive_runtime="${VINGS_ADAPTIVE_RUNTIME:-1}"
mapping_budget="${VINGS_MAPPING_BUDGET:-0}"
pruning_budget="${VINGS_PRUNING_BUDGET:-0}"
pixel_budget="${VINGS_PIXEL_BUDGET:-0}"
profile_runtime="${VINGS_PROFILE_RUNTIME:-0}"
metric_depth_schedule="${VINGS_METRIC_DEPTH_SCHEDULE:-0}"
metric_depth_warmup="${VINGS_METRIC_DEPTH_WARMUP:-30}"
metric_depth_interval="${VINGS_METRIC_DEPTH_INTERVAL:-5}"
metric_depth_mode="${VINGS_METRIC_DEPTH_MODE:-keyframe}"
metric_depth_keyframe_min_interval="${VINGS_METRIC_DEPTH_KEYFRAME_MIN_INTERVAL:-3}"
metric_depth_keyframe_force_interval="${VINGS_METRIC_DEPTH_KEYFRAME_FORCE_INTERVAL:-10}"
metric_depth_high_motion_ratio="${VINGS_METRIC_DEPTH_HIGH_MOTION_RATIO:-3.0}"
metric_depth_scale="${VINGS_METRIC_DEPTH_SCALE:-0.75}"
tracker_device="${VINGS_TRACKER_DEVICE:-cuda:0}"
mapper_device="${VINGS_MAPPER_DEVICE:-cuda:0}"

adaptive_runtime_args=()
if [ "$adaptive_runtime" != "0" ]; then
  adaptive_runtime_args+=(--adaptive-runtime)
fi

mapping_budget_args=()
if [ "$mapping_budget" != "0" ]; then
  mapping_budget_args+=(--enable-mapping-budget)
fi

pruning_budget_args=()
if [ "$pruning_budget" != "0" ]; then
  pruning_budget_args+=(--enable-jetson-pruning)
fi

pixel_budget_args=()
if [ "$pixel_budget" != "0" ]; then
  pixel_budget_args+=(--enable-pixel-budget)
fi

profile_runtime_args=()
if [ "$profile_runtime" != "0" ]; then
  profile_runtime_args+=(--profile-runtime)
fi

metric_depth_schedule_args=()
if [ "$metric_depth_schedule" != "0" ]; then
  metric_depth_schedule_args+=(--enable-metric-depth-schedule)
  metric_depth_schedule_args+=(--metric-depth-warmup "$metric_depth_warmup")
  metric_depth_schedule_args+=(--metric-depth-interval "$metric_depth_interval")
  metric_depth_schedule_args+=(--metric-depth-mode "$metric_depth_mode")
  metric_depth_schedule_args+=(--metric-depth-keyframe-min-interval "$metric_depth_keyframe_min_interval")
  metric_depth_schedule_args+=(--metric-depth-keyframe-force-interval "$metric_depth_keyframe_force_interval")
  metric_depth_schedule_args+=(--metric-depth-high-motion-ratio "$metric_depth_high_motion_ratio")
fi

if [ ! -d "$dataset_root/nosky_color" ]; then
  echo "Hotel dataset not ready: expected '$dataset_root/nosky_color'." >&2
  echo "Place the RTG-SLAM Hotel dataset at '$dataset_root' or override VINGS_DATASET_ROOT." >&2
  exit 1
fi

if [ ! -f "$frontend_weight" ]; then
  echo "Missing frontend weight: $frontend_weight" >&2
  exit 1
fi

if [ ! -f "$lightglue_weight_dir/superpoint.onnx" ] || [ ! -f "$lightglue_weight_dir/superpoint_lightglue.onnx" ]; then
  echo "Missing LightGlue ONNX weights under: $lightglue_weight_dir" >&2
  echo "Expected files: superpoint.onnx and superpoint_lightglue.onnx" >&2
  echo "Override with VINGS_LIGHTGLUE_DIR if they live elsewhere." >&2
  exit 1
fi

mkdir -p "$output_dir"

cd "$repo_root"

python scripts/run.py \
  configs/rtg/hotel.yaml \
  --prefix hotel \
  --dataset-root "$dataset_root" \
  --output-dir "$output_dir" \
  --frontend-weight "$frontend_weight" \
  --lightglue-weight-dir "$lightglue_weight_dir" \
  --loop-onnx-provider "$loop_onnx_provider" \
  --frontend-image-size "$frontend_image_size" \
  --training-iters "$training_iters" \
  --metric-depth-scale "$metric_depth_scale" \
  "${adaptive_runtime_args[@]}" \
  "${mapping_budget_args[@]}" \
  "${pruning_budget_args[@]}" \
  "${pixel_budget_args[@]}" \
  "${profile_runtime_args[@]}" \
  "${metric_depth_schedule_args[@]}" \
  --device-tracker "$tracker_device" \
  --device-mapper "$mapper_device"
