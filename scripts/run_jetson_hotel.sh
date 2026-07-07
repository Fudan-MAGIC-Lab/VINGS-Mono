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
  "${adaptive_runtime_args[@]}" \
  "${mapping_budget_args[@]}" \
  --device-tracker "$tracker_device" \
  --device-mapper "$mapper_device"
