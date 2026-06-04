#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

dataset_root="${VINGS_DATASET_ROOT:-$repo_root/data/smallcity/small_city}"
output_dir="${VINGS_OUTPUT_DIR:-$repo_root/output}"
frontend_weight="${VINGS_DROID_WEIGHT:-$repo_root/ckpts/droid.pth}"
tracker_device="${VINGS_TRACKER_DEVICE:-cuda:0}"
mapper_device="${VINGS_MAPPER_DEVICE:-cuda:0}"

cd "$repo_root"

python scripts/run.py \
  configs/hierarchical/smallcity.yaml \
  --prefix jetson \
  --dataset-root "$dataset_root" \
  --output-dir "$output_dir" \
  --frontend-weight "$frontend_weight" \
  --device-tracker "$tracker_device" \
  --device-mapper "$mapper_device" \
  --no-vis \
  --disable-metric